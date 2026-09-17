#!/usr/bin/env python3
"""E4 NIM slot health probe.

Pings every candidate model in the NIM slot ladder once per round with a
1-token request, recording {model, slot, status, latency_ms} to a JSONL log.
Determines which models deserve active routing slots (gate: <10s median).

Usage:
    python scripts/nim_probe.py [--once] [--interval SECONDS] [--log PATH]
    python scripts/nim_probe.py --check [--window HOURS] [--log PATH]

    --once          single round, print table, exit (default: daemon loop)
    --interval      seconds between rounds (default 1800 = 30 min)
    --check         summarize logged rounds: per-model median latency,
                    success rate, consecutive failures, promotion-gate verdicts
    --window        hours of history considered by --check (default 24)
    --log           JSONL path (default ~/.jarvis/nim_probe.jsonl)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

import httpx

DEFAULT_KEY_ENV = "NVIDIA_NEMOTRON_API_KEY"

ACTIVE_SLOTS = {
    "fast": {
        "models": [
            "deepseek-ai/deepseek-v4-flash-0731",
            "openai/gpt-oss-20b",
            "nvidia/nemotron-3-super-120b-a12b",
        ],
        "key_env": DEFAULT_KEY_ENV,
    },
    "coding": {
        "models": ["deepseek-ai/deepseek-v4-flash-0731", "openai/gpt-oss-20b"],
        "key_env": DEFAULT_KEY_ENV,
    },
    "frontier": {
        "models": ["nvidia/nemotron-3-ultra-550b-a55b"],
        "key_env": DEFAULT_KEY_ENV,
    },
    "active_vision": {
        "models": ["nvidia/llama-3.1-nemotron-nano-vl-8b-v1"],
        "key_env": "NVIDIA_API_KEY",
    },
}

E4_CANDIDATES = {
    "reasoning_gate": {
        "models": ["nvidia/nemotron-3-super-120b-a12b"],
        "key_env": DEFAULT_KEY_ENV,
    },
    "vision_gate": {
        "models": ["nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"],
        "key_env": DEFAULT_KEY_ENV,
    },
    "vision_gate_image": {
        "models": ["nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"],
        "key_env": "NVIDIA_API_KEY",
        "image": True,
    },
    "future": {
        "models": ["poolside/laguna-xs-2.1", "google/diffusiongemma-26b-a4b-it"],
        "key_env": DEFAULT_KEY_ENV,
    },
}

BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TIMEOUT = 45

# 1x1 transparent PNG (~70 bytes) for real vision-capability checks.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def group_key(key_env: str, fallback_key: str) -> str:
    value = (os.getenv(key_env) or "").strip()
    return value or fallback_key


def probe_model(key: str, model: str, with_image: bool = False) -> dict:
    t0 = time.time()
    content: list[dict] = []
    if with_image:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{TINY_PNG_B64}"},
            }
        )
    content.append({"type": "text", "text": "ping"})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content if with_image else "ping"}],
        "max_tokens": 1,
    }
    try:
        r = httpx.post(
            BASE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=TIMEOUT,
        )
        status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
        detail = "" if r.status_code == 200 else r.text[:80]
    except Exception as e:
        status = "error"
        detail = str(e)[:80]
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "slot": "image" if with_image else "",
        "status": status,
        "latency_ms": round((time.time() - t0) * 1000),
        "detail": detail,
    }


def run_round(fallback_key: str, log_path: Path) -> list[dict]:
    rows = []
    seen: set[tuple[str, bool]] = set()
    for group, spec in {**ACTIVE_SLOTS, **E4_CANDIDATES}.items():
        key = group_key(spec.get("key_env", DEFAULT_KEY_ENV), fallback_key) or fallback_key
        for model in spec["models"]:
            variant = bool(spec.get("image"))
            if (model, variant) in seen:
                continue
            seen.add((model, variant))
            row = probe_model(key, model, with_image=variant)
            row["slot"] = "vision_gate_image" if variant else group
            rows.append(row)
            print(f"{row['latency_ms']:6d}ms  {row['status']:10s}  [{row['slot']:16s}] {model}")
            with log_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
    return rows


def short_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def summarize(log_path: Path, window_hours: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    samples: dict[str, list[dict]] = {}
    if not log_path.exists():
        print(f"no log at {log_path}")
        return 0
    for line in log_path.open():
        try:
            d = json.loads(line)
            ts = datetime.fromisoformat(d["ts"])
        except Exception:
            continue
        if ts < cutoff:
            continue
        samples.setdefault(d["model"], []).append(d)

    print(f"--- nim probe --check (window {window_hours}h, {datetime.now(timezone.utc).isoformat()}) ---")
    print(f"{'model':38s} {'n':>3s} {'median':>8s} {'ok%':>5s} {'tailFail':>8s}  verdict")
    for model in sorted(samples):
        rows = samples[model]
        ms = sorted(r["latency_ms"] for r in rows)
        ok = [r for r in rows if r["status"] == "ok"]
        median = statistics.median(ms) if ms else None
        ok_rate = len(ok) / len(rows) if rows else 0.0
        tail = 0
        for r in reversed(rows):
            if r["status"] != "ok":
                tail += 1
            else:
                break
        verdict = "watch"
        gate = ""
        if model.endswith("nemotron-3-super-120b-a12b"):
            gate = "super-120b: >=10 ok, >=95%, <10s median"
            ok_rate_raw = ok_rate
            verdict = (
                "PASS"
                if len(ok) >= 10 and ok_rate_raw >= 0.95 and median is not None and median < 10000
                else "INSUFFICIENT" if len(rows) < 10
                else "FAIL"
            )
        elif model.endswith("nemotron-3-nano-omni-30b-a3b-reasoning"):
            gate = "omni: chat 200 x3 + vision 200"
            chat_ok = [r for r in rows if r["slot"] != "vision_gate_image" and r["status"] == "ok"]
            img_ok = [r for r in rows if r["slot"] == "vision_gate_image" and r["status"] == "ok"]
            verdict = (
                "PASS" if len(chat_ok) >= 3 and len(img_ok) >= 1
                else "INSUFFICIENT" if len(rows) < 3
                else "FAIL"
            )
        elif tail >= 2:
            verdict = "DEGRADED"
        median_s = f"{median / 1000:.1f}s" if median is not None else "-"
        print(
            f"{short_name(model):38s} {len(rows):3d} {median_s:>8s} "
            f"{ok_rate * 100:5.0f}% {tail:8d}  {verdict:12s} {gate}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="single round then exit")
    ap.add_argument("--interval", type=int, default=1800, help="seconds between rounds")
    ap.add_argument("--check", action="store_true", help="summarize logged rounds and print gate verdicts")
    ap.add_argument("--window", type=int, default=24, help="hours of history for --check (default 24)")
    ap.add_argument("--log", default=str(Path.home() / ".jarvis" / "nim_probe.jsonl"))
    args = ap.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.check:
        return summarize(log_path, args.window)

    key = os.environ.get("NVIDIA_NEMOTRON_API_KEY", "")
    if not key:
        print("NVIDIA_NEMOTRON_API_KEY not set", file=sys.stderr)
        return 1

    while True:
        print(f"--- nim probe round {datetime.now(timezone.utc).isoformat()} ---")
        run_round(key, log_path)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
