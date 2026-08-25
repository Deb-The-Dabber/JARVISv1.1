"""Phase 5: Unified Regression Gate

Orchestrates the full regression suite:
  1. Golden-set eval (intent → tools → response quality, mock or real)
  2. Classifier gate benchmarks (configs A-E, offline sweep)
  3. Coding-routing benchmark (optional, real providers)
  4. Unified report with pass/fail gates

Usage:
    # CI/fast mode (mock providers only):
    python benchmarks/regression_gate.py --mode mock

    # Local/full mode (real providers, requires API keys):
    python benchmarks/regression_gate.py --mode real --health-seed benchmarks/health_seeds/baseline.json

    # Just the golden set (mock):
    python benchmarks/regression_gate.py --mode mock --only golden

    # Just the gate benchmarks:
    python benchmarks/regression_gate.py --mode mock --only gates

Outputs:
    benchmarks/results/regression_gate_<timestamp>.json  (full report)
    benchmarks/results/latest_regression.json            (latest)
    benchmarks/results/gate_bench_summary.json           (gate configs)

Exit codes:
    0 = all gates passed
    1 = regression detected (gate failure)
    2 = infrastructure error (mock provider down, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "benchmarks" / "results"
GOLDEN_SET_PATH = ROOT / "tests" / "eval" / "golden_set.jsonl"
ROUTING_GOLDEN_PATH = ROOT / "tests" / "eval" / "routing_golden.jsonl"


# ──────────────────────────────────────────────────────────────────────
# Thresholds (tune these per release)
# ──────────────────────────────────────────────────────────────────────
# Mock mode: catches infrastructure regressions (crashes, 500s, timeouts, routing loops)
# Real mode: catches quality regressions (intent, tools, content, cost)
GATES = {
    "mock": {
        "golden_pass_rate": 0.0,         # eval_runner's 'passed' requires kw_recall≥0.5 — mock replies generic
        "golden_intent_acc": 0.80,       # keyword routing should work
        "golden_tool_acc": 0.40,         # tools may be called but mock replies generic
        "golden_kw_recall": 0.0,         # mock replies don't contain real content
        "golden_max_latency_s": 30.0,
        "gate_min_accuracy": 0.70,       # keyword routing accuracy baseline
        "gate_max_escalation": 1.0,      # all escalate when local_nn disabled
        "gate_min_accept_rate": 0.0,     # local_nn disabled in mock
        "gate_max_incorrect_accept": 1.0,
    },
    "real": {
        "golden_pass_rate": 0.75,
        "golden_intent_acc": 0.85,
        "golden_tool_acc": 0.6,
        "golden_kw_recall": 0.6,
        "golden_max_latency_s": 30.0,
        "gate_min_accuracy": 0.85,
        "gate_max_escalation": 0.30,
        "gate_min_accept_rate": 0.50,
        "gate_max_incorrect_accept": 0.05,
        "coding_pass_rate": 0.60,
        "coding_intent_acc": 0.80,
        "coding_class_match": 0.70,
        "coding_fallback_rate": 0.30,
        "coding_avg_latency_s": 180.0,
        "coding_cost_usd_per_req": 0.005,
    },
}


def _gates_for_mode(mode: str) -> dict:
    return GATES.get(mode, GATES["mock"])

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], env: dict | None = None, timeout: int = 300, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess with merged env, return CompletedProcess."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=full_env, cwd=cwd or ROOT)


def _start_mock_provider(port: int = 18889) -> subprocess.Popen:
    """Start the mock provider server in background."""
    env = os.environ.copy()
    env.update({
        "JARVIS_MOCK_PROVIDERS": "1",
        "MOCK_PROVIDER_PORT": str(port),
        "MOCK_PROVIDER_URL": f"http://127.0.0.1:{port}",
    })
    # Dummy keys so brain.py passes its import-time checks
    for k in ["NVIDIA_NEMOTRON_API_KEY", "NVIDIA_API_KEY", "NVIDIA_EMBEDDING_API_KEY",
              "DEEPSEEK_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "KIMI_API_KEY",
              "TAVILY_API_KEY", "ELEVENLABS_API_KEY"]:
        env.setdefault(k, "mock-key")

    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "tests.mock_provider:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    # Wait for health
    for _ in range(30):
        try:
            import requests
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("Mock provider failed to start")


def _stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ──────────────────────────────────────────────────────────────────────
# 1. Golden-set eval (imports eval_runner)
# ──────────────────────────────────────────────────────────────────────

def run_golden_eval(mode: str, require_no_api: bool = False) -> dict:
    """Run the golden set through eval_runner."""
    print(f"\n=== Golden-set eval ({mode}) ===")
    os.environ["JARVIS_EVAL_MODE"] = "1"
    if mode == "mock":
        os.environ["JARVIS_MOCK_PROVIDERS"] = "1"
        os.environ["MOCK_PROVIDER_URL"] = "http://127.0.0.1:18889"
        os.environ["MOCK_PROVIDER_PORT"] = "18889"

    # Import eval_runner (already wired with LLM judge scoring)
    from eval_runner import run_eval_suite

    report = run_eval_suite(require_no_api=require_no_api)
    return {
        "mode": mode,
        "total": report["total"],
        "passed": report["passed"],
        "failed": report["failed"],
        "skipped": report["skipped"],
        "pass_rate": report["pass_rate"],
        "avg_intent_accuracy": report["avg_intent_accuracy"],
        "avg_tool_accuracy": report["avg_tool_accuracy"],
        "avg_keyword_recall": report["avg_keyword_recall"],
        "avg_latency_seconds": report["avg_latency_seconds"],
        "results": report["results"],
    }


# ──────────────────────────────────────────────────────────────────────
# 2. Classifier gate benchmarks (configs A-E)
# ──────────────────────────────────────────────────────────────────────

def run_gate_benchmarks(mode: str) -> dict:
    """Run the classifier gate benchmarks (configs A-E)."""
    print(f"\n=== Classifier gate benchmarks ({mode}) ===")
    os.environ["JARVIS_EVAL_MODE"] = "1"
    if mode == "mock":
        os.environ["JARVIS_MOCK_PROVIDERS"] = "1"
        os.environ["MOCK_PROVIDER_URL"] = "http://127.0.0.1:18889"
        os.environ["MOCK_PROVIDER_PORT"] = "18889"

    # Reuse classifier_gate_bench.py's run_all
    from benchmarks.classifier_gate_bench import run_all, load_cases, CONFIGS

    cases = load_cases()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in sorted(CONFIGS):
        env = os.environ.copy()
        env["JARVIS_LLM_FIRST"] = "1"
        env["JARVIS_ROUTER_CHEAP"] = "1" if name == "E" else "0"
        env["JARVIS_ROUTER_POLICY"] = "0"
        env.update(CONFIGS[name])
        print(f"  Config {name}...", flush=True)

        proc = subprocess.run(
            [sys.executable, str(ROOT / "benchmarks" / "classifier_gate_bench.py"), "--config", name],
            capture_output=True, text=True, timeout=300, env=env, cwd=ROOT
        )
        if proc.returncode != 0:
            return {"error": f"Config {name} failed: {proc.stderr[-500:]}"}
        summary_file = RESULTS_DIR / f"gate_bench_{name}.json"
        if summary_file.exists():
            summaries.append(json.loads(summary_file.read_text()))

    if not summaries:
        return {"error": "No gate configs succeeded"}

    (RESULTS_DIR / "gate_bench_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    return {"configs": summaries}


# ──────────────────────────────────────────────────────────────────────
# 3. Coding routing benchmark (optional, real providers)
# ──────────────────────────────────────────────────────────────────────

def run_coding_routing(tag: str, health_seed: str | None, limit: int = 0) -> dict:
    """Run the coding routing benchmark (real providers)."""
    print(f"\n=== Coding routing benchmark ({tag}) ===")
    # Use the existing coding_routing_bench.py with --one per case
    from benchmarks.coding_routing_bench import load_cases, _spawn_case, _make_isolation, run_suite

    cases = load_cases(ROUTING_GOLDEN_PATH)
    if limit > 0:
        cases = cases[:limit]

    isolation = _make_isolation(health_seed, tag)
    report = run_suite(cases, tag, health_seed)
    return report


# ──────────────────────────────────────────────────────────────────────
# Gate evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_gates(golden: dict, gates: dict, coding: dict | None, mode: str) -> tuple[bool, list[str]]:
    """Evaluate all gates against thresholds. Returns (passed, violations)."""
    violations = []
    G = _gates_for_mode(mode)

    # Golden set gates
    if golden:
        if golden.get("pass_rate", 0) < G["golden_pass_rate"]:
            violations.append(f"golden_pass_rate: {golden['pass_rate']:.3f} < {G['golden_pass_rate']}")
        if golden.get("avg_intent_accuracy", 0) < G["golden_intent_acc"]:
            violations.append(f"golden_intent_acc: {golden['avg_intent_accuracy']:.3f} < {G['golden_intent_acc']}")
        if golden.get("avg_tool_accuracy", 0) < G["golden_tool_acc"]:
            violations.append(f"golden_tool_acc: {golden['avg_tool_accuracy']:.3f} < {G['golden_tool_acc']}")
        if golden.get("avg_keyword_recall", 0) < G["golden_kw_recall"]:
            violations.append(f"golden_kw_recall: {golden['avg_keyword_recall']:.3f} < {G['golden_kw_recall']}")
        if golden.get("avg_latency_seconds", 999) > G["golden_max_latency_s"]:
            violations.append(f"golden_latency: {golden['avg_latency_seconds']:.1f}s > {G['golden_max_latency_s']}s")

    # Gate benchmarks (configs A-D)
    for s in gates.get("configs", []):
        if s.get("config") in ("A", "B", "C", "D"):
            if s.get("accuracy", 0) < G["gate_min_accuracy"]:
                violations.append(f"gate {s['config']} accuracy: {s['accuracy']:.3f} < {G['gate_min_accuracy']}")
            if s.get("escalation_rate", 1) > G["gate_max_escalation"]:
                violations.append(f"gate {s['config']} escalation_rate: {s['escalation_rate']:.3f} > {G['gate_max_escalation']}")
            if s.get("local_nn_accepted", 0) / max(s.get("total_cases", 1), 1) < G["gate_min_accept_rate"]:
                violations.append(f"gate {s['config']} accept_rate too low")
            incorrect_accepts = s.get("local_nn_incorrect_accepts", 0)
            total_accepts = s.get("local_nn_accepted", 1)
            if incorrect_accepts / max(total_accepts, 1) > G["gate_max_incorrect_accept"]:
                violations.append(f"gate {s['config']} incorrect accepts: {incorrect_accepts}/{total_accepts}")

    # Coding routing (if run)
    if coding and not coding.get("error") and not coding.get("skipped"):
        if coding.get("pass_rate", 0) < G.get("coding_pass_rate", 0.6):
            violations.append(f"coding pass_rate: {coding['pass_rate']:.3f} < {G.get('coding_pass_rate', 0.6)}")
        if coding.get("intent_accuracy", 0) < G.get("coding_intent_acc", 0.8):
            violations.append(f"coding intent_acc: {coding['intent_accuracy']:.3f} < {G.get('coding_intent_acc', 0.8)}")
        if coding.get("provider_class_match_rate", 0) < G.get("coding_class_match", 0.7):
            violations.append(f"coding class_match: {coding['provider_class_match_rate']:.3f} < {G.get('coding_class_match', 0.7)}")
        if coding.get("fallback_rate", 1) > G.get("coding_fallback_rate", 0.3):
            violations.append(f"coding fallback_rate: {coding['fallback_rate']:.3f} > {G.get('coding_fallback_rate', 0.3)}")
        if coding.get("avg_total_latency_ms", 999999) > G.get("coding_avg_latency_s", 180) * 1000:
            violations.append(f"coding latency: {coding['avg_total_latency_ms']/1000:.1f}s > {G.get('coding_avg_latency_s', 180)}s")

    return len(violations) == 0, violations


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5: Unified Regression Gate")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock",
                    help="mock = mock providers (CI), real = real providers (local)")
    ap.add_argument("--only", choices=["golden", "gates", "coding", "all"], action="append", default=[],
                    help="which suites to run (can repeat)")
    ap.add_argument("--health-seed", default="", help="static provider-health JSON for coding benchmark")
    ap.add_argument("--limit", type=int, default=0, help="limit coding cases (0 = all)")
    ap.add_argument("--tag", default="regression", help="tag for coding benchmark output")
    ap.add_argument("--no-api", action="store_true", help="skip cases that need real API (golden eval)")
    args = ap.parse_args()

    # Normalize --only: if empty or contains "all", run everything
    only = set(args.only) if args.only else {"all"}
    if "all" in only:
        only = {"golden", "gates", "coding"}
    mode = args.mode
    start = time.time()

    # Mock provider setup (needed for both mock and real modes for eval/gates)
    mock_proc = None
    if args.mode == "mock" or args.only in ("golden", "gates", "all"):
        try:
            mock_proc = _start_mock_provider()
            print("Mock provider started on port 18889")
        except Exception as e:
            print(f"ERROR: Failed to start mock provider: {e}", file=sys.stderr)
            return 2

    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "run_duration_seconds": 0,
        "golden": {},
        "gates": {},
        "coding": {},
        "gates_passed": False,
        "violations": [],
    }

    try:
        # 1. Golden set eval
        if "golden" in only:
            results["golden"] = run_golden_eval(mode, require_no_api=args.no_api)

        # 2. Classifier gate benchmarks
        if "gates" in only:
            results["gates"] = run_gate_benchmarks(mode)

        # 3. Coding routing (real providers only, optional)
        if "coding" in only and mode == "real":
            results["coding"] = run_coding_routing(args.tag, args.health_seed or None, args.limit)
        elif "coding" in only and mode == "mock":
            results["coding"] = {"skipped": True, "reason": "coding benchmark requires --mode real"}

        # Evaluate gates
        passed, violations = evaluate_gates(results["golden"], results["gates"], results.get("coding"), mode)
        results["gates_passed"] = passed
        results["violations"] = violations

    finally:
        if mock_proc:
            _stop_proc(mock_proc)

    results["run_duration_seconds"] = round(time.time() - start, 1)

    # Save unified report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat()
    safe_ts = ts.replace(":", "-").replace(".", "-")
    report_path = RESULTS_DIR / f"regression_gate_{safe_ts}.json"
    with report_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    (RESULTS_DIR / "latest_regression.json").write_text(json.dumps(results, indent=2, default=str))

    # Print summary
    print("\n" + "=" * 60)
    print("REGRESSION GATE SUMMARY")
    print("=" * 60)
    print(f"Mode: {mode} | Duration: {results['run_duration_seconds']}s")
    print(f"Golden:  pass_rate={results['golden'].get('pass_rate', 'N/A'):.3f}  "
          f"intent={results['golden'].get('avg_intent_accuracy', 'N/A'):.3f}  "
          f"tool={results['golden'].get('avg_tool_accuracy', 'N/A'):.3f}  "
          f"kw={results['golden'].get('avg_keyword_recall', 'N/A'):.3f}")
    if "configs" in results.get("gates", {}):
        for s in results["gates"]["configs"]:
            print(f"  Gate {s['config']}: acc={s['accuracy']:.3f}  "
                  f"accept={s['local_nn_accepted']}/{s['total_cases']}  "
                  f"wrong={s['local_nn_incorrect_accepts']}  esc={s['escalation_rate']:.2f}")
    if results.get("coding"):
        c = results["coding"]
        if not c.get("skipped"):
            print(f"Coding:  pass_rate={c.get('pass_rate', 0):.3f}  "
                  f"intent={c.get('intent_accuracy', 0):.3f}  "
                  f"class={c.get('provider_class_match_rate', 0):.3f}  "
                  f"fallback={c.get('fallback_rate', 0):.3f}")
    print(f"\nGATES PASSED: {results['gates_passed']}")
    if results["violations"]:
        for v in results["violations"]:
            print(f"  VIOLATION: {v}")

    print(f"\nReport: {report_path}")

    return 0 if results["gates_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())