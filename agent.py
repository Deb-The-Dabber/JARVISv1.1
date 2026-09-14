import hashlib
import json
import os
import re
import threading
import time
import uuid
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import plans
from event_bus import publish

# Phase 2: ActiveTask integration
from task import ActiveTask, TaskPhase, TaskEventType

# ── Goal sanitization: strip bypass/PII before passing to sub-agents ──
_BYPASS_PATTERNS = re.compile(
    r"(?i)(bypass|skip|ignore|disable|circumvent|override|suppress)"
    r"\s*(permission|prompt|confirm|approval|check|gate|block|safety|restrict)"
    r"|"
    r"(don'?t|do not|never)\s*(wait|ask|prompt|confirm|check|halt)"
    r"|"
    r"(just|simply|automatically)\s*(go|do|proceed|execute|run)\s*(without|ahead|freely)"
    r"|"
    r"\bjust go ahead\b"
)
_PII_PATTERNS = re.compile(
    r"\b\d{13,19}\b"  # credit card / long numbers
    r"|\b\d{3}-\d{2}-\d{4}\b"  # SSN
    r"|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"  # email
)


def _sanitize_goal(goal: str) -> str:
    goal = _BYPASS_PATTERNS.sub("[redacted]", goal)
    goal = _PII_PATTERNS.sub("[redacted]", goal)
    return goal


AGENT_TRIGGERS = [
    "discord",
    "send a message",
    "send the chat",
    "open the chat",
    "read the most recent",
    "fix",
    "debug",
    "refactor",
    "rewrite",
    "build",
    "create and",
    "write a script",
    "write code",
    "implement",
    "edit the file",
    "update the file",
    "read and",
    "scan and",
    "find and fix",
    "look through",
    "go through",
    "check all",
    "analyse",
    "analyze",
    "step by step",
    "automatically",
    "keep trying",
    "until it works",
    "iterate",
]

PLANNER_TRIGGERS = [
    "complex",
    "multi-step",
    "research",
    "investigate",
    "compare",
    "find and",
    "gather",
    "collect",
    "organize",
    "prepare a report",
    "comprehensive",
    "thorough",
    "plan",
]

MAX_STEPS = 30

# ── Phase 2: agent-loop discipline budgets ──
# Per-request hard limits so a derailed loop degrades gracefully instead of
# burning unbounded time/model calls. All overridable via env vars.
MAX_AGENT_TOOL_CALLS = 10          # total executed tool calls per request
MAX_AGENT_EXPLORE_CALLS = 8        # read-only info-gathering calls before forced transition
MAX_AGENT_LOOP_WALL_CLOCK_S = 90.0  # hard wall-clock ceiling per request


def _env_budget(name: str, default: float) -> float:
    """Read a numeric env override, falling back to the default."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Phase 5F: Structured event logging for Thought/Action/Reality separation
@dataclass
class AgentEvent:
    """Structured event for Thought/Action/Reality separation."""
    event_type: str
    step_num: int
    timestamp: str
    agent_id: str
    goal: str
    phase: str
    # Model Thought fields
    model_thought: str = ""
    model_tool: str = ""
    model_args: dict = None
    # Tool Action fields
    tool_name: str = ""
    tool_args: dict = None
    # Verified Reality fields
    verified: bool = False
    verification_msg: str = ""
    tool_result: str = ""
    success: bool = False
    # State change fields
    old_status: str = ""
    new_status: str = ""
    # Progress fields
    verified_progress_count: int = 0
    false_progress_count: int = 0
    criteria_passed: int = 0
    criteria_total: int = 0
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "step_num": self.step_num,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "goal": self.goal,
            "phase": self.phase,
            "model_thought": self.model_thought,
            "model_tool": self.model_tool,
            "model_args": self.model_args,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "verified": self.verified,
            "verification_msg": self.verification_msg,
            "tool_result": self.tool_result,
            "success": self.success,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "verified_progress_count": self.verified_progress_count,
            "false_progress_count": self.false_progress_count,
            "criteria_passed": self.criteria_passed,
            "criteria_total": self.criteria_total,
        }


def _log_agent_event(event: AgentEvent) -> None:
    """Log a structured agent event."""
    try:
        from event_bus import publish
        publish("agent_event", event.to_dict())
    except Exception:
        pass  # Don't let event logging break the agent loop


# Tools that only gather information (read-only) — counted against the
# exploration budget and blocked once it is exhausted.
EXPLORE_TOOLS = frozenset({
    "read_file", "web_search", "search_web", "search_news", "search_shopping",
    "search_my_notes", "semantic_search_memory", "warwatch_news",
    "fetch_top_headlines", "fetch_everything", "find_recent_screenshot",
    "docs_search", "docs_get", "slides_search", "slides_get",
    "forms_get", "forms_get_responses", "gdrive_get", "gdrive_download",
    "inspect_tools", "list_agents", "get_agent_status",
    "browser_current_url", "spotify_current", "check_email", "disk_usage",
})

PHASE_EXPLORE = "explore"
PHASE_PLAN = "plan"
PHASE_IMPLEMENT = "implement"
_PHASE_ORDER = (PHASE_EXPLORE, PHASE_PLAN, PHASE_IMPLEMENT)


def _advance_phase(current: str, requested: str) -> str:
    """Monotonic phase transition: explore → plan → implement. Never regresses."""
    if requested not in _PHASE_ORDER:
        return current
    try:
        cur = _PHASE_ORDER.index(current)
    except ValueError:
        return PHASE_EXPLORE
    return _PHASE_ORDER[max(cur, _PHASE_ORDER.index(requested))]


def _validate_phase_transition(current: str, requested: str) -> bool:
    """Validate that a phase transition is allowed.
    
    Only allows: EXPLORE -> PLAN -> IMPLEMENT (monotonic forward).
    Returns True if transition is valid, False otherwise.
    """
    if requested not in _PHASE_ORDER:
        return False
    try:
        cur_idx = _PHASE_ORDER.index(current)
        req_idx = _PHASE_ORDER.index(requested)
    except ValueError:
        return False
    # Only allow forward transitions or staying in same phase
    return req_idx >= cur_idx

# ── Think → Act → Evaluate: Plan dataclasses ──


def _stable_step_id(goal: str, tool_hint: str, args: dict) -> str:
    """Generate a stable, deterministic step ID from step semantics.
    
    Uses content-addressing so the same logical step produces the same ID
    across planner regenerations, retries, and process restarts.
    """
    # Normalize args to sorted, canonical representation
    args_canonical = json.dumps(args, sort_keys=True, separators=(",", ":"))
    # Normalize goal and tool hint
    goal_norm = re.sub(r"\s+", " ", goal.strip().lower())
    tool_norm = tool_hint.strip().lower()
    # Create stable hash from semantic content
    content = f"{goal_norm}|{tool_norm}|{args_canonical}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class PlanStep:
    step_id: str = ""
    goal: str = ""
    tool_hint: str = ""
    args: dict = field(default_factory=dict)
    status: str = "pending"
    result: str = ""
    evaluation: str = ""


@dataclass
class Plan:
    plan_id: str = ""
    original_goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    current_step: int = 0
    status: str = "running"
    final_answer: str = ""
    executed_call_sigs: dict = field(default_factory=dict)  # sig -> {"status","result"}


def _call_signature(tool: str, args: dict) -> str:
    """Canonical (tool,args) signature for convergence/dedupe checks.

    Uses SHA-1 of the canonical json so it is stable for identical calls and
    small enough for logging."""
    try:
        c = json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        c = repr(args)
    return f"{tool}::{hashlib.sha1(c.encode()).hexdigest()}"


_plan_store: dict[str, Plan] = {}
_plan_lock = threading.Lock()


def _register_plan(p: Plan):
    with _plan_lock:
        _plan_store[p.plan_id] = p
    _persist_plan_for_execution(p)


def get_plan(plan_id: str) -> Plan | None:
    with _plan_lock:
        return _plan_store.get(plan_id)


def list_plans() -> list[dict]:
    with _plan_lock:
        return [
            {
                "plan_id": p.plan_id,
                "goal": p.original_goal[:100],
                "status": p.status,
                "steps": len(p.steps),
                "current_step": p.current_step,
            }
            for p in _plan_store.values()
        ]


def needs_planner(text: str) -> bool:
    """Check if the user's request is complex enough to warrant a planner agent."""
    t = (text or "").lower()
    return any(trigger in t for trigger in PLANNER_TRIGGERS)


def _strip_code_fences(text: str) -> str:
    """Remove markdown/code fence markers (```json, ```) from an LLM reply."""
    return re.sub(r"```(?:json)?", "", text or "").replace("```", "").strip()


def _scan_balanced(text: str, opening: str, closing: str) -> list[str]:
    """Return every balanced-bracket substring in ``text`` (string-aware).

    Handles quotes/escapes so '[x]' inside a JSON string value doesn't
    truncate the scan. Longest-balanced semantics: nested pairs collapse
    into their outermost match.
    """
    spans: list[tuple[int, int, int]] = []
    for m in re.finditer(re.escape(opening), text):
        depth = 0
        in_str = False
        escaped = False
        for i in range(m.start(), len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    spans.append((m.start(), i + 1, i + 1 - m.start()))
                    break
    # Shortest-first so the outermost (longest) match wins dedup
    spans.sort(key=lambda s: s[2])
    seen: set[tuple[int, int]] = set()
    out: list[str] = []
    for start, end, _ in spans:
        if any(start >= s and end <= e for s, e in seen):
            continue
        seen.add((start, end))
        out.append(text[start:end])
    out.sort(key=len, reverse=True)
    return out


def _extract_json_array(text: str) -> list | None:
    """Parse a JSON array out of a messy LLM reply (fences/prose/strict).

    Order: strict full-parse of the fence-stripped reply, then every
    balanced ``[...]`` block (longest first). Returns None when no array
    parses, so callers degrade gracefully instead of crashing.
    """
    stripped = _strip_code_fences(text)
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
        if isinstance(data, list):
            return data
    except ValueError:
        pass
    for cand in _scan_balanced(stripped, "[", "]"):
        try:
            data = json.loads(cand)
            if isinstance(data, list):
                return data
        except ValueError:
            continue
    return None


def _extract_json_object(text: str) -> dict | None:
    """Parse a JSON object out of a messy LLM reply (for retry decisions)."""
    stripped = _strip_code_fences(text)
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass
    for cand in _scan_balanced(stripped, "{", "}"):
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except ValueError:
            continue
    return None


def _step_from_dict(s: dict) -> PlanStep | None:
    """Coerce a raw model dict into a PlanStep, tolerating schema drift."""
    if not isinstance(s, dict):
        return None
    clean = {k: v for k, v in s.items() if k in {"step_id", "goal", "tool_hint", "args", "status", "result", "evaluation"}}
    if not isinstance(clean.get("args"), dict):
        clean["args"] = {}
    # Generate stable step_id from step semantics if not provided or if it's a sequential number
    provided_id = clean.get("step_id")
    if not provided_id or (isinstance(provided_id, str) and provided_id.isdigit()):
        clean["step_id"] = _stable_step_id(
            clean.get("goal", ""),
            clean.get("tool_hint", ""),
            clean.get("args", {})
        )
    clean.setdefault("goal", "")
    clean.setdefault("tool_hint", "")
    clean.setdefault("status", "pending")
    clean.setdefault("result", "")
    clean.setdefault("evaluation", "")
    return PlanStep(**clean)


def _tool_contract_text() -> str:
    """Compact authoritative argument contract for the planner model.

    Built from the live TOOL_DEFINITIONS registry once — eliminates the
    'planner invents argument names' class of bugs (e.g. read_file(file_path=...)
    when the contract requires 'path'). Lazy import because tools/__init__.
    imports agent.py.
    """
    try:
        from tools import TOOL_DEFINITIONS
    except Exception:
        return ""
    lines = []
    for entry in TOOL_DEFINITIONS:
        fn = entry.get("function", {})
        name = fn.get("name", "")
        if not name:
            continue
        params = (fn.get("parameters") or {}).get("properties", {})
        required = (fn.get("parameters") or {}).get("required", [])
        parts = []
        for pname, pmeta in params.items():
            marker = "" if pname in required else "?"
            parts.append(f"{pname}{marker}")
        lines.append(f"{name}({', '.join(parts)})")
        if len(lines) >= 120:  # cap to stay within prompt budget
            break
    return "\n".join(lines)


def _create_plan(goal: str, ask_llm_fn) -> Plan | None:
    """Use the LLM to decompose a goal into steps.

    Phase 4: robust against plain-completion replies — the model may wrap
    the JSON in prose or return it in a fence; any parseable array wins,
    and malformed entries are dropped instead of crashing the loop.
    """
    contract = _tool_contract_text()
    contract_block = f"\nAvailable tools (authoritative — use EXACTLY these names and arg keys):\n{contract}\n" if contract else ""
    prompt = (
        "You are Jarvis's planner. Given a user goal, break it into a sequence of 2-6 discrete steps. "
        "Each step should use exactly one tool. Respond ONLY with a JSON array of steps.\n\n"
        f"Goal: {goal}\n"
        f"{contract_block}\n"
        "Format:\n"
        "[\n"
        '  {"step_id": "1", "goal": "what to accomplish", "tool_hint": "suggested_tool_name",\n'
        '   "args": {"query": "search term", "app_name": "Safari"}},\n'
        "  ...\n"
        "]\n"
        "Rules:\n"
        "- Each step must be achievable with a single tool call\n"
        "- tool_hint MUST be one of the tool names listed above (no invented names)\n"
        "- args keys MUST match the parameter names listed for that tool (suffix '?' = optional)\n"
        "- If a tool's required params are unclear, pick the closest known tool instead of inventing args\n"
        "- Return NOTHING but the JSON array"
    )
    try:
        raw = ask_llm_fn(prompt)
    except Exception:
        return None
    steps_data = _extract_json_array(raw)
    if not steps_data:
        return None
    steps = [s for s in (_step_from_dict(x) for x in steps_data) if s is not None]
    if not steps:
        return None
    plan = Plan(
        plan_id=uuid.uuid4().hex[:8],
        original_goal=goal,
        steps=steps,
    )
    _register_plan(plan)
    return plan


def _evaluate_step(step: PlanStep, ask_llm_fn) -> str:
    """LLM-as-judge: evaluate whether a step succeeded."""
    prompt = (
        "You are Jarvis's step evaluator. Determine if this step succeeded.\n\n"
        f"Step goal: {step.goal}\n"
        f"Tool: {step.tool_hint}\n"
        f"Result: {step.result[:500]}\n\n"
        "Respond with a single word: SUCCESS or FAILURE. Then a brief reason.\n"
        "If Result asks the user for confirmation or approval (sandbox preview),"
        " respond with PENDING instead — nothing executed yet."
    )
    try:
        raw = ask_llm_fn(prompt)
        upper = raw.strip().upper()
        if upper.startswith("PENDING"):
            return "pending"
        if upper.startswith("SUCCESS"):
            return "success"
        return "failure"
    except Exception:
        return "failure" if "error" in step.result.lower() else "success"


def _generate_final_answer(plan: Plan, ask_llm_fn) -> str:
    """Synthesize a final answer from all completed steps."""
    steps_text = "\n".join(f"Step {s.step_id}: {s.goal} → {s.evaluation}\n  Result: {s.result[:200]}" for s in plan.steps if s.result)
    prompt = f"Summarize what was accomplished based on these step results.\nOriginal goal: {plan.original_goal}\n\n{steps_text}\n\nProvide a concise summary of what was done and key findings."
    try:
        return ask_llm_fn(prompt) or "Completed."
    except Exception:
        return "Completed."


# Phase 6E: Multi-step replanning
def _replan(goal: str, ask_llm_fn, failed_results: List[dict], context: str = "") -> Plan | None:
    """
    Phase 6E: Re-plan based on verification failures.
    Creates a new plan that addresses the failed criteria.
    """
    failed = [r for r in failed_results if not r.get("passed") and r.get("required", True)]
    if not failed:
        return _create_plan(goal, ask_llm_fn)
    
    # Build context from failures
    failure_context = "\n".join([
        f"- {r['type']}: {r['target']} - {r['message'][:200]}"
        for r in failed
    ])
    
    replan_prompt = (
        f"You are Jarvis's replanner. The previous plan failed to meet these goal criteria:\n"
        f"{failure_context}\n\n"
        f"Additional context: {context}\n\n"
        f"Original goal: {goal}\n\n"
        "Create a NEW plan that addresses these failures. Focus on fixing the failed criteria.\n"
        "Respond ONLY with a JSON array of steps (same format as _create_plan).\n\n"
        "Format:\n"
        "[\n"
        '  {"step_id": "1", "goal": "what to accomplish", "tool_hint": "suggested_tool_name",\n'
        '   "args": {"query": "search term", "app_name": "Safari"}},\n'
        "  ...\n"
        "]\n"
        "Rules:\n"
        "- Each step must be achievable with a single tool call\n"
        "- tool_hint must match a known tool name\n"
        "- Each step MUST include an 'args' object with ALL required parameters\n"
        "- Return NOTHING but the JSON array"
    )
    try:
        raw = ask_llm_fn(replan_prompt)
    except Exception:
        return None
    steps_data = _extract_json_array(raw)
    if not steps_data:
        return None
    steps = [s for s in (_step_from_dict(x) for x in steps_data) if s is not None]
    if not steps:
        return None
    plan = Plan(
        plan_id=uuid.uuid4().hex[:8],
        original_goal=goal,
        steps=steps,
    )
    _register_plan(plan)
    return plan


def _persist_plan_for_execution(plan: Plan, task: "ActiveTask | None" = None) -> object | None:
    """Persist the plan and its steps immediately before executing it.

    Persistence is deliberately best-effort: a database failure must not
    discard the in-memory planner path.  Importing here keeps the planner
    usable in installations that have not enabled the optional plan store.
    """
    try:
        import plans

        plans.save_plan(
            plan_id=plan.plan_id,
            original_goal=plan.original_goal,
            status=plan.status,
            current_step=plan.current_step,
            final_answer=plan.final_answer,
            active_task_id=task.task_id if task else None,
        )
        for step_index, step in enumerate(plan.steps):
            plans.save_plan_step(
                step_id=step.step_id,
                plan_id=plan.plan_id,
                step_index=step_index,
                goal=step.goal,
                tool_hint=step.tool_hint,
                args=step.args,
                status=step.status,
                result=step.result,
                evaluation=step.evaluation,
            )
        return plans
    except Exception:
        return None


def plan_from_persistence(plan_id: str, plans_module=None) -> Plan | None:
    """Rebuild a runtime Plan from the persisted store (post-restart resume).

    Returns None if the plan does not exist or the store is unavailable.
    Steps are ordered by their persisted numeric step_index; step IDs are
    content hashes with no ordering semantics.
    """
    try:
        pm = plans_module if plans_module is not None else __import__("plans")
        data = pm.load_full_plan(plan_id)
    except Exception:
        return None
    if not data:
        return None
    ordered = sorted(data.get("steps") or [], key=lambda s: s.get("step_index", 0))
    steps = [
        PlanStep(
            step_id=s.get("step_id", ""),
            goal=s.get("goal", ""),
            tool_hint=s.get("tool_hint", ""),
            args=s.get("args") if isinstance(s.get("args"), dict) else {},
            status=s.get("status", "pending"),
            result=s.get("result", ""),
            evaluation=s.get("evaluation", ""),
        )
        for s in ordered
    ]
    plan = Plan(
        plan_id=data.get("plan_id", plan_id),
        original_goal=data.get("original_goal", ""),
        steps=steps,
        current_step=data.get("current_step", 0),
        status=data.get("status", "running"),
        final_answer=data.get("final_answer", ""),
    )
    _register_plan(plan)
    return plan


def _persist_execution_start(plans_module, plan: Plan, step: PlanStep, step_index: int, tool: str, args: dict) -> str | None:
    """Durably mark an attempt before invoking its tool.

    A process crash after this write but before the outcome update leaves a
    record with ``success=False`` and no result, correctly identifying the
    attempt as having an unknown outcome rather than never having started.
    """
    if plans_module is None:
        return None
    try:
        attempts = plans_module.load_execution_records_for_step(plan.plan_id, step.step_id)
        execution_id = uuid.uuid4().hex
        plans_module.save_execution_record(
            execution_id=execution_id,
            plan_id=plan.plan_id,
            step_id=step.step_id,
            step_index=step_index,
            tool=tool,
            args=args,
            attempt=len(attempts) + 1,
        )
        return execution_id
    except Exception:
        return None


def _persist_execution_outcome(
    plans_module,
    execution_id: str | None,
    plan: Plan,
    step: PlanStep,
    step_index: int,
    error: str | None,
    duration_ms: int,
) -> None:
    """Persist the outcome that is available after one tool attempt."""
    if plans_module is None:
        return
    try:
        if execution_id:
            plans_module.update_execution_record(
                execution_id,
                result=step.result,
                evaluation=step.evaluation,
                success=step.evaluation == "success",
                error=error,
                duration_ms=duration_ms,
            )
        plans_module.update_step_status(
            plan.plan_id,
            step.step_id,
            step.status,
            result=step.result,
            evaluation=step.evaluation,
        )
        plans_module.update_plan_status(
            plan.plan_id,
            plan.status,
            current_step=step_index,
        )
    except Exception:
        pass


def _looks_like_pending_confirmation(result: str) -> bool:
    """Marker check for a tool that stopped at the sandbox preview / approval gate.

    When a tool requires confirmation (run_python, run_terminal_command, write
    file, etc.), brain._execute_tool returns the confirm-prompt text instead of
    real output. Evaluating that preview text as a failed outcome is what made
    preview paths look like project failures in Fallback; per the demo log the
    evaluator honestly answered FAILURE when the first preview landed, which led
    to a recovery-replan cycle asking for the same calls again.
    """
    r = (result or "").lower()
    if not r:
        return False
    return (
        ("say yes to run it for real" in r)
        or ("say yes to apply" in r)
        or ("say yes to proceed" in r)
        or ("awaiting approval" in r)
        or ("preview shown above" in r and "say yes" in r)
    )


def _execute_planned_step(
    plans_module,
    plan: Plan,
    step: PlanStep,
    step_index: int,
    tool: str,
    args: dict,
    execute_tool_fn,
    ask_llm_fn,
) -> None:
    """Execute and evaluate one planner attempt with incremental persistence."""
    execution_id = _persist_execution_start(
        plans_module, plan, step, step_index, tool, args
    )
    started_at = time.time()

    # Deterministic convergence: if this plan already executed the same
    # (tool, args) signature, reuse the recorded result. The tool-level memo
    # cache only covers the current turn; plan-level dedup covers repeats
    # across steps AND across recovery suggestions.
    sig = _call_signature(tool, args)
    prev_call = plan.executed_call_sigs.get(sig)
    if prev_call is not None:
        step.result = prev_call["result"]
        step.evaluation = "success" if prev_call["status"] == "completed" else "failure"
        step.status = "completed" if prev_call["status"] == "completed" else "failed"
        _persist_execution_outcome(
            plans_module, execution_id, plan, step, step_index,
            None, int((time.time() - started_at) * 1000),
        )
        return

    error = None
    try:
        result = execute_tool_fn(tool, args)
        step.result = str(result)
    except Exception as exc:
        error = str(exc)
        step.result = f"Tool error: {exc}"

    # A pending confirmation is NOT an execution outcome. The sandbox preview
    # may also have hit an OS-level PermissionError (sandboxed previews can't
    # write outside the sandbox); brief the evaluator on what actually happened
    # so it doesn't flag the sandbox as the project.
    if _looks_like_pending_confirmation(step.result):
        step.status = "awaiting_confirmation"
        step.evaluation = "pending"
        _persist_execution_outcome(
            plans_module,
            execution_id,
            plan,
            step,
            step_index,
            "needs user approval (preview only)",
            int((time.time() - started_at) * 1000),
        )
        return
    if "PermissionError" in step.result and "[OS sandboxed]" in step.result:
        # Sandboxed previews may not touch user files — the preview failure is
        # a sandbox inconvenience, not a project regression. Label and move on.
        step.status = "awaiting_confirmation"
        step.evaluation = "pending"
        step.result += "\n[Note] That's a preview-sandbox write block, not a real project failure."
        _persist_execution_outcome(
            plans_module,
            execution_id,
            plan,
            step,
            step_index,
            "sandbox preview blocked by OS write permission",
            int((time.time() - started_at) * 1000),
        )
        return

    step.evaluation = _evaluate_step(step, ask_llm_fn)
    step.status = "completed" if step.evaluation == "success" else "failed"

    # Track every executed (tool,args) attempt on the plan so the recovery
    # loop never re-executes an identical call (Issue 2: loop convergence).
    sig = _call_signature(tool, args)
    plan.executed_call_sigs[sig] = {"status": step.status, "result": step.result}

    _persist_execution_outcome(
        plans_module,
        execution_id,
        plan,
        step,
        step_index,
        error,
        int((time.time() - started_at) * 1000),
    )


def run_planner_loop_with_plan(
    plan: Plan | None = None,
    execute_tool_fn=None,
    ask_llm_fn=None,
    speak_fn=None,
    task: "ActiveTask | None" = None,
    resume_from_step_id: str | None = None,
    plan_id: str | None = None,
) -> str:
    """Execute an existing plan, optionally resuming at one of its steps.

    Provide exactly one of `plan` (in-memory Plan) or `plan_id` (persisted plan).
    Supplying both raises ValueError; supplying neither raises ValueError.
    """
    start_time = time.time()

    if plan is not None and plan_id is not None:
        raise ValueError("Provide either plan or plan_id, not both")
    if plan is None:
        if plan_id is None:
            raise ValueError("run_planner_loop_with_plan requires a plan or plan_id")
        plan = plan_from_persistence(plan_id)
        if plan is None:
            return f"I couldn't resume the plan because plan '{plan_id}' was not found."

    plans_module = _persist_plan_for_execution(plan, task)
    step_indexes = {step.step_id: index for index, step in enumerate(plan.steps)}
    if plans_module is not None:
        try:
            # The persisted ordering is authoritative.  Step IDs are content
            # hashes and cannot be compared to establish execution order.
            step_indexes = {
                stored_step["step_id"]: stored_step["step_index"]
                for stored_step in plans_module.load_plan_steps(plan.plan_id)
            }
        except Exception:
            pass
    ordered_steps = sorted(
        plan.steps,
        key=lambda step: step_indexes.get(step.step_id, len(plan.steps)),
    )

    resume_index = min(step_indexes.values(), default=0)
    if resume_from_step_id is not None:
        resume_index = step_indexes.get(resume_from_step_id)
        if resume_index is None:
            return (
                "I couldn't resume the plan because step "
                f"'{resume_from_step_id}' was not found."
            )

    # ACT → EVALUATE loop.  IDs are content hashes and deliberately have no
    # ordering semantics; only the persisted step_index controls resume order.
    for step in ordered_steps:
        step_idx = step_indexes.get(step.step_id, 0)
        if step_idx < resume_index:
            continue
        plan.current_step = step_idx
        step.status = "running"
        if plans_module is not None:
            try:
                plans_module.update_plan_status(plan.plan_id, "running", current_step=step_idx)
                plans_module.update_step_status(plan.plan_id, step.step_id, "running")
            except Exception:
                pass

        tool_name = step.tool_hint
        args = step.args if step.args else {}
        _execute_planned_step(
            plans_module, plan, step, step_idx, tool_name, args,
            execute_tool_fn, ask_llm_fn,
        )

        # Re-plan if step failed (max 2 retries per step)
        if step.status == "failed":
            retry_prompt = (
                f"Step '{step.goal}' failed. Result: {step.result[:300]}\n"
                "Suggest an alternative approach or tool to accomplish this goal. "
                f'Respond with: {{"tool": "tool_name", "args": {{"param": "value"}}, "reason": "why"}}\n'
                "Include ALL required arguments for the tool in the args field.\n"
                "Do NOT suggest the exact same tool+args that just failed."
            )
            try:
                raw = ask_llm_fn(retry_prompt)
                retry = _extract_json_object(raw) or {}
                alt_tool = retry.get("tool", "")
                alt_args = retry.get("args", {})
                if not isinstance(alt_args, dict):
                    alt_args = {}

                # Recovery-dedup: don't re-execute an identical (tool, args) call.
                # We already executed it (success or failure) once this plan —
                # re-running is wasted. Reuse the recorded result instead.
                sig = _call_signature(alt_tool, alt_args)
                if alt_tool and sig in plan.executed_call_sigs:
                    prev = plan.executed_call_sigs[sig]
                    if prev["status"] == "completed":
                        step.status = "completed"
                        step.result = prev["result"]
                        step.evaluation = "success"
                    # Identical failed calls: re-executing them never helps and
                    # wastes LLM budget. Don't re-execute the doomed call.
                elif alt_tool:
                    _execute_planned_step(
                        plans_module, plan, step, step_idx, alt_tool, alt_args,
                        execute_tool_fn, ask_llm_fn,
                    )
            except Exception:
                pass

        # Announce progress for multi-step plans
        if speak_fn and len(plan.steps) > 1:
            status_msg = f"Step {step_idx + 1} of {len(plan.steps)}: {step.evaluation}"
            try:
                speak_fn(status_msg)
            except Exception:
                pass

        # Phase 2: Record step to active task
        if task:
            step_data = {
                "tool": step.tool_hint,
                "args": step.args,
                "success": step.evaluation == "success",
                "result": step.result,
                "thought": step.goal,
            }
            task.record_step(step_data, step.tool_hint, step.args)
            task.save()

    # Synthesize final answer
    plan.status = "completed"
    plan.final_answer = _generate_final_answer(plan, ask_llm_fn)
    if plans_module is not None:
        try:
            plans_module.update_plan_status(
                plan.plan_id,
                plan.status,
                current_step=plan.current_step,
                final_answer=plan.final_answer,
            )
        except Exception:
            pass
    publish("subagent_completed", {
        "agent_id": "planner",  # identifier for planner
        "goal": plan.original_goal,
        "final_answer": plan.final_answer,
        "steps": [
            {"tool": s.tool_hint, "status": "success" if s.evaluation == "success" else "failed"}
            for s in plan.steps
        ],
        "duration": time.time() - start_time,
        "timestamp": time.time(),
    })
    return plan.final_answer


def run_planner_loop(goal: str, execute_tool_fn, ask_llm_fn, speak_fn=None, task: "ActiveTask | None" = None) -> str:
    """Think → Act → Evaluate loop with separate planner agent."""

    # Phase 2: Initialize with task context if provided
    if task:
        # Inject task context into goal
        if task.get_context_for_prompt():
            goal = f"{task.get_context_for_prompt()}\n\nCurrent Goal: {goal}"
        # Pre-populate already_called with task's tool history
        # (handled by the agent's internal logic if needed)

    if speak_fn:
        try:
            speak_fn("Let me think through the best approach for this.")
        except Exception:
            pass

    # THINK: Create a plan
    plan = _create_plan(goal, ask_llm_fn)
    if not plan:
        return "I couldn't create a plan for this task. Try being more specific or use the direct agent instead."

    if speak_fn:
        try:
            speak_fn(f"Okay, I have a {len(plan.steps)}-step plan. Let me start working through it.")
        except Exception:
            pass

    return run_planner_loop_with_plan(
        plan,
        execute_tool_fn,
        ask_llm_fn,
        speak_fn=speak_fn,
        task=task,
    )


# ── Success/Failure detection ──

# Agent success detection — scored terms
SUCCESS_TERMS = {
    "opened",
    "sent",
    "completed",
    "created",
    "started",
    "playing",
    "navigated",
    "remembered",
    "saved",
    "quit",
    "closed",
    "delivered",
    "found",
    "results",
    "added",
    "marked",
    "done",
    "success",
    "launched",
    "focused",
    "loaded",
    "navigating",
    "posted",
    "resumed",
    "skipped",
    "next",
    "timer set",
    "countdown",
    "goal added",
    "stored",
    "written",
    "edited",
    "updated",
    "modified",
    "exit code 0",
    "temperature",
    "humidity",
    "wind",
    "cpu",
    "ram",
    "disk",
    "percent",
    "killed",
}
FAILURE_TERMS = {
    "error",
    "failed",
    "could not",
    "timeout",
    "exception",
    "denied",
    "not found",
    "permission",
    "unavailable",
    "invalid",
    "missing",
    "traceback",
    "cannot",
    "unable",
    "refused",
    "aborted",
    "no such",
}


def _is_success_result(tool_name: str, result: str) -> bool:
    """Score-based success detection. Primary = positive terms; fallback = non-empty without failure."""
    r = result.lower()
    # Score positive and negative terms
    pos_score = sum(1 for t in SUCCESS_TERMS if t in r)
    neg_score = sum(1 for t in FAILURE_TERMS if t in r)
    if pos_score or neg_score:
        return pos_score > neg_score
    # Fallback: non-empty result is likely success
    return bool(r.strip())


def _verify_tool_execution(tool_name: str, args: dict, result_str: str, execute_tool_fn) -> tuple[bool, str]:
    """
    Phase 4: Verify tool execution actually accomplished what was intended.
    For write operations, re-read the file. For code changes, run tests.
    Returns (verified: bool, verification_message: str).
    Uses a verification-specific execute that doesn't count towards tool budget.
    """
    # Create a verification-only execute function that doesn't affect budgets
    def _verify_execute(tool, args):
        return execute_tool_fn(tool, args)
    
    try:
        # For write operations, verify by reading back
        if tool_name in ("write_file", "create_file", "append_file"):
            path = args.get("path", "")
            if path:
                verify_result = _verify_execute("read_file", {"path": path, "offset": 0})
                if verify_result and "Could not read file" not in verify_result:
                    return True, f"Verified: {tool_name} succeeded (file readable)"
                else:
                    return False, f"Verification failed: {tool_name} wrote but file not readable"
        
        # For code execution, check if tests were run
        if tool_name in ("run_python", "run_terminal_command"):
            # If result contains test output, check for pass/fail
            result_lower = result_str.lower()
            if any(kw in result_lower for kw in ["test", "pytest", "passed", "failed", "ok", "error"]):
                # Check for explicit failure indicators
                if re.search(r'\b(failed|error|traceback)\b', result_lower) and not re.search(r'\b0 failed\b', result_lower):
                    return False, f"Tests/commands indicated failure"
                # Check for explicit success indicators
                if re.search(r'\b(passed|ok)\b', result_lower):
                    return True, "Tests/commands passed"
        
        # For other tools, rely on result scoring
        success = _is_success_result(tool_name, "")
        return success, "Verified via result scoring"
    except Exception as e:
        return False, f"Verification error: {e}"


def _filter_narration(text: str) -> str:
    """
    Phase 4: Filter out verbose model narration from user-facing output.
    Strips common narration patterns like "Let me...", "Now I will...", "I need to..."
    but preserves the actual content after the narration.
    """
    if not text:
        return text
    
    # Patterns to remove from start of lines
    narration_patterns = [
        r"^\s*(let me|now i will|i will|i need to|let me|i'm going to|i am going to)\s+",
        r"^\s*(first,?|then,?|next,?|finally,?)\s+",
        r"^\s*(okay,?|alright,?|sure,?)\s+",
    ]
    
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.lstrip()
        # Check if line starts with narration pattern
        is_narration_line = False
        for pattern in narration_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_narration_line = True
                # Remove just the narration prefix, keep the rest
                stripped = re.sub(pattern, '', stripped, flags=re.IGNORECASE)
                break
        if stripped:
            filtered_lines.append(stripped)
    
    result = '\n'.join(filtered_lines).strip()
    # If everything was filtered, return a concise summary
    if not result:
        return "Action completed"
    return result


_agent_store: dict[str, "Agent"] = {}
_store_lock = threading.Lock()
AGENTS_DB = Path.home() / ".jarvis" / "agents.json"


def _save_agents():
    """Persist agent store to JSON file."""
    try:
        AGENTS_DB.parent.mkdir(parents=True, exist_ok=True)
        with _store_lock:
            data = {aid: a.to_dict() for aid, a in _agent_store.items()}
        with open(AGENTS_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _load_agents():
    """Load agent store from JSON file on startup."""
    if not AGENTS_DB.exists():
        return
    try:
        with open(AGENTS_DB, "r", encoding="utf-8") as f:
            data = json.load(f)
        with _store_lock:
            for aid, adict in data.items():
                # Create a minimal Agent-like object with just the dict data
                # We store as dict since full Agent reconstruction needs tools/execute_fn
                _agent_store[aid] = adict
    except Exception:
        pass


# Load persisted agents on module import
_load_agents()


class Agent:
    def __init__(self, goal: str, tools: dict = None, max_iterations: int = MAX_STEPS):
        self.id = uuid.uuid4().hex[:8]
        self.goal = goal
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.steps = []
        self.last_result = ""
        self.already_called: set[tuple[str, str]] = set()
        self.fail_counts = {}
        self.status = "running"
        self.error = None
        self.final_answer = ""
        self.parent_id = None

    def checkpoint(self):
        try:
            from procedural_memory import save_procedure

            summary = "; ".join(f"{s['tool']}:{'ok' if s['success'] else 'fail'}" for s in self.steps[-5:])
            save_procedure(
                trigger=f"agent_{self.id}",
                steps=[s["tool"] for s in self.steps if s.get("tool")],
                description=f"Agent #{self.id}: {self.goal[:80]} ({summary})",
            )
        except Exception:
            pass

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal[:200],
            "status": self.status,
            "step_count": len(self.steps),
            "successful_steps": sum(1 for s in self.steps if s.get("success")),
            "tools_used": sorted(set(s["tool"] for s in self.steps if s.get("tool"))),
            "final_answer": self.final_answer[:200] if self.final_answer else "",
            "error": self.error,
            "parent_id": self.parent_id,
        }


def _register_agent(a: Agent):
    with _store_lock:
        _agent_store[a.id] = a
    _save_agents()


def get_agent(agent_id: str) -> Agent | dict | None:
    with _store_lock:
        return _agent_store.get(agent_id)


def list_agents() -> list[dict]:
    with _store_lock:
        result = []
        for a in _agent_store.values():
            if hasattr(a, 'to_dict'):
                result.append(a.to_dict())
            elif isinstance(a, dict):
                result.append(a)
        return result


def stop_agent(agent_id: str) -> bool:
    with _store_lock:
        if agent_id in _agent_store:
            agent = _agent_store[agent_id]
            if hasattr(agent, 'status'):
                agent.status = "stopped"
            elif isinstance(agent, dict):
                agent["status"] = "stopped"
            _save_agents()
            return True
        return False


def needs_agent_loop(text: str) -> bool:
    t = (text or "").lower()
    return any(trigger in t for trigger in AGENT_TRIGGERS)


def _parse_decision(raw: str) -> dict:
    if not raw:
        # Empty model response — do NOT hard-stop with "No response from model."
        # as the final answer; the loop retries and aborts gracefully after budget.
        return {"thought": "No response from model.", "tool": "", "args": {},
                "done": False, "final_answer": "", "_no_response": True}
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {
        "thought": "Model returned non-JSON response.",
        "tool": "",
        "args": {},
        "done": False,
        "final_answer": "",
    }


def _build_prompt(goal: str, steps: list, last_result: str, already_called: set = None,
                  phase: str = PHASE_EXPLORE, max_tool_calls: int = MAX_AGENT_TOOL_CALLS,
                  max_explore: int = MAX_AGENT_EXPLORE_CALLS) -> str:
    recent = steps[-5:]
    recent_text = "\n".join(f"{i + 1}. tool={step.get('tool')} success={step.get('success')} result={step.get('result', '')[:200]}" for i, step in enumerate(recent))
    already_text = ""
    if already_called:
        already_text = "\nAlready executed this turn — do NOT repeat:\n" + "\n".join(f"  - {t}({a})" for t, a in sorted(already_called))
    return (
        "You are Jarvis running an autonomous agent loop. "
        "Respond ONLY with a single JSON object — no explanation, no backticks.\n"
        f"Goal: {goal}\n\n"
        "Recent steps:\n"
        f"{recent_text if recent_text else '(none yet)'}\n\n"
        f"Last result: {last_result or '(none yet)'}\n"
        f"{already_text}\n\n"
        "Required JSON format:\n"
        '{{"thought": "brief reasoning", "tool": "exact_tool_name", '
        '"args": {{}}, "phase": "explore|plan|implement (optional advance)", "done": false, '
        '"final_answer": ""}}\n\n'
        f"Current phase: {phase}\n"
        "Phase contract (advance only, never regress):\n"
        "- explore: gather information with read-only tools only "
        "(read_file, web_search, search_*, docs_get, get_agent_status, ...)\n"
        "- plan: think through the approach (at most 2 steps)\n"
        "- implement: execute the changes that complete the goal\n"
        "- Set \"phase\" in your JSON to advance; the loop enforces it "
        "monotonically\n"
        f"- Hard budgets: at most {max_tool_calls} total tool calls, "
        f"at most {max_explore} read-only calls\n\n"
        "Rules:\n"
        "- done=true only when the goal is fully complete or cannot continue\n"
        "- If done=true, put the full response in final_answer and leave tool empty\n"
        "- Pick exactly one tool per step\n"
        "- args must be a valid JSON object\n"
        "- Never call the same tool with the same arguments twice\n"
        "- Return NOTHING except the JSON object"
    )


def _synthesize_from_steps(goal, steps, note: str = None) -> str:
    lines = [f"Agent summary for: {goal}"]
    successful = [step for step in steps if step.get("success")]
    if not successful:
        lines.append("- No successful tool steps completed.")
    for step in successful:
        tool = step.get("tool", "unknown")
        result = str(step.get("result", "")).strip()
        # Phase 4: Filter out verbose narration from results
        result = _filter_narration(result)
        if len(result) > 180:
            result = result[:177] + "..."
        lines.append(f"- {tool}: {result or 'completed'}")
    if note:
        lines.append(f"- Stopped: {note}")
    elif len(steps) >= MAX_STEPS:
        lines.append("- Stopped after reaching the maximum step limit.")
    return "\n".join(lines)


def get_agent_stats(steps) -> dict:
    tools = [step.get("tool") for step in steps if step.get("tool")]
    return {
        "total_steps": len(steps),
        "tools_used": sorted(set(tools)),
        "successful_steps": sum(1 for step in steps if step.get("success")),
    }


def _canonical_args_key(args: dict) -> str:
    return str(sorted((k, str(v)) for k, v in (args or {}).items()))


def run_agent_loop(goal: str, execute_tool_fn, ask_llm_fn, speak_fn=None, max_iterations: int = 30, task: "ActiveTask | None" = None) -> str:
    # Record start time for duration metrics
    start_time = time.time()
    # Clamp the iteration cap so a derailed loop can't spin on empty/no-tool steps.
    max_tool_calls = int(_env_budget("JARVIS_AGENT_MAX_TOOL_CALLS", MAX_AGENT_TOOL_CALLS))
    max_explore = int(_env_budget("JARVIS_AGENT_MAX_EXPLORE", MAX_AGENT_EXPLORE_CALLS))
    wall_clock_s = _env_budget("JARVIS_AGENT_WALL_CLOCK_S", MAX_AGENT_LOOP_WALL_CLOCK_S)
    max_iterations = max(1, min(max_iterations or MAX_STEPS, max_tool_calls + 4))
    deadline = time.time() + wall_clock_s
    agent = Agent(goal=goal, max_iterations=max_iterations)
    
    # Phase 2: Initialize agent with task context if provided
    if task:
        # Pre-populate already_called with task's tool history to prevent cross-turn duplicates
        for history_key, step_idx in task.tool_history.items():
            if ":" in history_key:
                tool_name, arg_key = history_key.split(":", 1)
                agent.already_called.add((tool_name, arg_key))
        # Inject task context into goal
        if task.get_context_for_prompt():
            goal = f"{task.get_context_for_prompt()}\n\nCurrent Goal: {goal}"
    
    _register_agent(agent)

    if speak_fn:
        try:
            speak_fn("On it, let me work through this step by step.")
        except Exception:
            pass

    tool_calls = 0
    explore_calls = 0
    empty_decisions = 0
    plan_steps = 0
    phase = PHASE_EXPLORE
    abort_reason = None

    for step_num in range(1, agent.max_iterations + 1):
        if agent.status == "stopped":
            agent.final_answer = _synthesize_from_steps(goal, agent.steps)
            return agent.final_answer

        if time.time() > deadline:
            abort_reason = f"wall clock limit ({wall_clock_s:g}s) reached"
            break
        if tool_calls >= max_tool_calls:
            abort_reason = f"tool budget ({max_tool_calls} calls) exhausted"
            break

        # Phase 7A: Budget enforcement - check all budgets at start of each iteration
        if task:
            budget = task.execution_budget
            # Check time budget
            if budget["time_spent_seconds"] >= budget["time_budget_seconds"]:
                abort_reason = f"time budget ({budget['time_budget_seconds']}s) exhausted"
                task.enter_blocked(f"Time budget exhausted ({budget['time_budget_seconds']}s)")
                break
            # Check LLM budget
            if budget["llm_calls"] >= budget["llm_budget"]:
                abort_reason = f"LLM budget ({budget['llm_calls']}/{budget['llm_budget']} calls) exhausted"
                task.enter_blocked(f"LLM budget exhausted ({budget['llm_budget']} calls)")
                break
            # Check tool budget
            if budget["tool_calls"] >= budget["tool_budget"]:
                abort_reason = f"tool budget ({budget['tool_budget']} calls) exhausted"
                task.enter_blocked(f"Tool budget exhausted ({budget['tool_budget']} calls)")
                break
            # Check replan budget
            if budget["replans"] >= budget["replan_budget"]:
                abort_reason = f"replan budget ({budget['replan_budget']}) exhausted"
                task.enter_blocked(f"Replan budget exhausted ({budget['replan_budget']} replans)")
                break
            # Check recovery budget
            if budget["recoveries"] >= budget["recovery_budget"]:
                abort_reason = f"recovery budget ({budget['recovery_budget']}) exhausted"
                task.enter_blocked(f"Recovery budget exhausted ({budget['recovery_budget']} recoveries)")
                break

        prompt = _build_prompt(goal, agent.steps, agent.last_result,
                               agent.already_called, phase, max_tool_calls, max_explore)
        raw_decision = ask_llm_fn(prompt)
        # Phase 7A: Increment LLM call budget
        if task:
            task.execution_budget["llm_calls"] += 1
            task.execution_budget["time_spent_seconds"] = int(time.time() - start_time)
        decision = _parse_decision(raw_decision)

        if decision.get("done"):
            # Phase 6: Verify goal before completing
            if task:
                goal_passed, verification_results = task.verify_goal()
                if not goal_passed:
                    agent.last_result = f"[GOAL VERIFICATION FAILED] {goal}: criteria not met.\n"
                    for r in verification_results:
                        if r.get("required", True) and not r.get("passed"):
                            agent.last_result += f"  - FAIL: {r['type']}: {r['target']} - {r['message'][:200]}\n"
                    agent.last_result += "\n[RECOVERY] Task incomplete. Entering recovery mode."
                    task.enter_blocked("Goal verification failed")
                    # Don't return - let the agent continue to try to fix
                    decision["done"] = False
                    continue
            
            agent.final_answer = (decision.get("final_answer") or "").strip()
            # Phase 4: Filter narration from final answer
            agent.final_answer = _filter_narration(agent.final_answer)
            agent.final_answer = agent.final_answer or _synthesize_from_steps(goal, agent.steps)
            agent.status = "completed"
            agent.checkpoint()
            # Phase 5F: Log completion event
            _log_agent_event(AgentEvent(
                event_type=TaskEventType.VERIFIED_REALITY,
                step_num=step_num,
                timestamp=datetime.datetime.now().isoformat(),
                agent_id=agent.id,
                goal=goal,
                phase=phase,
                verified=True,
                success=True,
            ))
            return agent.final_answer

        # Phase 5F: Log model thought
        _log_agent_event(AgentEvent(
            event_type=TaskEventType.MODEL_THOUGHT,
            step_num=step_num,
            timestamp=datetime.datetime.now().isoformat(),
            agent_id=agent.id,
            goal=goal,
            phase=phase,
            model_thought=decision.get("thought", ""),
            model_tool=decision.get("tool", ""),
            model_args=decision.get("args", {}),
        ))

        # Monotonic phase advance requested by the model
        requested_phase = decision.get("phase")
        if isinstance(requested_phase, str):
            new_phase = _advance_phase(phase, requested_phase.strip().lower())
            if new_phase != phase:
                phase = new_phase
                plan_steps = 0
                transition_note = f"[Phase transition] Now in {phase} phase."
                agent.last_result = f"{agent.last_result}\n{transition_note}" if agent.last_result else transition_note

                # Phase 6B: Automatic verification at phase boundaries
                if task and task.goal_criteria:
                    goal_passed, verification_results = task.verify_goal()
                    if not goal_passed:
                        agent.last_result += f"\n[PHASE VERIFICATION] Goal criteria not met for {phase} phase:"
                        for r in verification_results:
                            if r.get("required", True) and not r.get("passed"):
                                agent.last_result += f"\n  - FAIL: {r['type']}: {r['target']} - {r['message'][:200]}"
                        agent.last_result += "\n[RECOVERY] Cannot advance phase without meeting criteria."

        # Phase 2: Stagnation detection — if task has 3+ non-progress steps, force re-evaluation
        if task and task.check_stagnation():
            agent.last_result = f"{agent.last_result}\n[STAGNATION DETECTED] No meaningful progress in last 3+ steps. Re-evaluating approach."
            if phase == PHASE_EXPLORE:
                phase = PHASE_PLAN
            elif phase == PHASE_PLAN:
                phase = PHASE_IMPLEMENT
            task.reset_stagnation()

        # Phase 6: Auto-transition recovery states
        if task:
            new_status = task.check_recovery_transitions()
            if new_status != task.status:
                agent.last_result = f"{agent.last_result}\n[RECOVERY] Task state transitioned to {task.status.upper()}"
                _log_agent_event(AgentEvent(
                    event_type=TaskEventType.STATE_CHANGE,
                    step_num=step_num,
                    timestamp=datetime.datetime.now().isoformat(),
                    agent_id=agent.id,
                    goal=goal,
                    phase=phase,
                    old_status=task.status,
                    new_status=new_status,
                ))

        if decision.get("_no_response"):
            empty_decisions += 1
            agent.last_result = "No response from model (empty reply)."
            agent.steps.append(
                {
                    "tool": "",
                    "args": {},
                    "success": False,
                    "result": agent.last_result,
                    "thought": decision.get("thought", ""),
                }
            )
            if empty_decisions >= 2:
                abort_reason = "model stopped responding"
                break
            continue

        tool_name = (decision.get("tool") or "").strip()
        args = decision.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        if not tool_name:
            agent.steps.append(
                {
                    "tool": "",
                    "args": args,
                    "success": False,
                    "result": "No tool selected.",
                    "thought": decision.get("thought", ""),
                }
            )
            agent.last_result = "No tool selected."
            continue

        arg_key = _canonical_args_key(args)

        # Exploration budget — block further read-only calls once exhausted
        if tool_name in EXPLORE_TOOLS and explore_calls >= max_explore:
            agent.last_result = (f"Skipped {tool_name}: exploration budget "
                                 f"({max_explore} read-only calls) reached — move to implementation.")
            agent.steps.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "success": False,
                    "result": agent.last_result,
                    "thought": decision.get("thought", ""),
                }
            )
            if phase == PHASE_EXPLORE:
                phase = PHASE_PLAN
                plan_steps = 0
            continue

        if (tool_name, arg_key) in agent.already_called:
            agent.last_result = f"Skipped {tool_name}: already called with these args."
            agent.steps.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "success": False,
                    "result": agent.last_result,
                    "thought": decision.get("thought", ""),
                }
            )
            continue

        if agent.fail_counts.get(tool_name, 0) >= 2:
            agent.last_result = f"Skipped {tool_name}: failed too many times."
            agent.steps.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "success": False,
                    "result": agent.last_result,
                    "thought": decision.get("thought", ""),
                }
            )
            continue

        # Rapid-repeat prevention — skip if same tool used ≥2 times in last 3 steps
        HIGH_FREQ_TOOLS = {"browser_navigate", "open_app", "quit_app", "web_search", "spotify_play", "spotify_skip"}
        if tool_name in HIGH_FREQ_TOOLS and len(agent.steps) >= 2:
            last_tools = [s.get("tool") for s in agent.steps[-3:] if s.get("tool")]
            if last_tools.count(tool_name) >= 2:
                agent.last_result = f"Skipped {tool_name}: used {last_tools.count(tool_name)}x in last 3 steps."
                agent.steps.append(
                    {
                        "tool": tool_name,
                        "args": args,
                        "success": False,
                        "result": agent.last_result,
                        "thought": decision.get("thought", ""),
                    }
                )
                continue

        print(f"[AGENT] Step {step_num}: {tool_name}({args})")

        try:
            result = execute_tool_fn(tool_name, args)
            result_str = str(result)
            success = _is_success_result(tool_name, result_str)
            if not success:
                agent.fail_counts[tool_name] = agent.fail_counts.get(tool_name, 0) + 1
            agent.last_result = result_str
        except Exception as e:
            success = False
            agent.fail_counts[tool_name] = agent.fail_counts.get(tool_name, 0) + 1
            agent.last_result = f"Tool execution error: {e}"

        # Phase 7A: Increment tool call budget
        if task:
            task.execution_budget["tool_calls"] += 1

        # Phase 4: Verify tool execution actually accomplished the goal
        verified, verify_msg = _verify_tool_execution(tool_name, args, agent.last_result, execute_tool_fn)
        if not verified and success:
            # Tool appeared to succeed but verification failed
            success = False
            agent.last_result = f"{agent.last_result}\n[Verification Failed] {verify_msg}"
        elif verified:
            # Verification passed - append verification info
            agent.last_result = f"{agent.last_result}\n[Verified] {verify_msg}"

        # Phase 5F: Log tool action and verified reality
        _log_agent_event(AgentEvent(
            event_type=TaskEventType.TOOL_ACTION,
            step_num=step_num,
            timestamp=datetime.datetime.now().isoformat(),
            agent_id=agent.id,
            goal=goal,
            phase=phase,
            tool_name=tool_name,
            tool_args=args,
            success=success,
            tool_result=agent.last_result[:500],
        ))
        
        if verified:
            _log_agent_event(AgentEvent(
                event_type=TaskEventType.VERIFIED_REALITY,
                step_num=step_num,
                timestamp=datetime.datetime.now().isoformat(),
                agent_id=agent.id,
                goal=goal,
                phase=phase,
                tool_name=tool_name,
                tool_args=args,
                verified=True,
                verification_msg=verify_msg,
                tool_result=agent.last_result[:500],
                success=success,
                verified_progress_count=task.progress_metrics.get("verified_progress_count", 0) if task else 0,
                false_progress_count=task.progress_metrics.get("false_progress_count", 0) if task else 0,
            ))

        tool_calls += 1
        if tool_name in EXPLORE_TOOLS:
            explore_calls += 1

        agent.already_called.add((tool_name, arg_key))
        step_data = {
            "tool": tool_name,
            "args": args,
            "success": success,
            "result": agent.last_result,
            "thought": decision.get("thought", ""),
        }
        agent.steps.append(step_data)

        # Phase 2: Record step to active task for cross-turn persistence
        if task:
            task.record_step(step_data, tool_name, args, verified=verified)
            # Sync task phase with agent phase via validated transition
            task.phase = _advance_phase(task.phase, phase)
            task.next_action = decision.get("thought", "")[:200]
            task.save()

        # Plan phase lasts at most 2 executed steps, then auto-advance
        if phase == PHASE_PLAN:
            plan_steps += 1
            if plan_steps >= 2:
                phase = PHASE_IMPLEMENT
                plan_steps = 0

        if step_num % 5 == 0:
            # Emit progress event for sub‑agent
            publish("subagent_progress", {
                "agent_id": agent.id,
                "step": step_num,
                "tool": tool_name,
                "status": "running",
            })
            agent.checkpoint()

    # Graceful degradation: report partial progress instead of "No response from model."
    if abort_reason:
        agent.status = "completed"
        agent.final_answer = _synthesize_from_steps(goal, agent.steps, note=abort_reason)
        agent.checkpoint()
        publish("subagent_completed", {
            "agent_id": agent.id,
            "goal": goal,
            "final_answer": agent.final_answer,
            "steps": [
                {"tool": s.get("tool"), "status": "success" if s.get("success") else "failed"}
                for s in agent.steps
            ],
            "duration": time.time() - start_time,
            "timestamp": time.time(),
            "abort_reason": abort_reason,
        })
        return agent.final_answer

    agent.status = "completed"
    agent.final_answer = _synthesize_from_steps(goal, agent.steps)
    agent.checkpoint()
    # Emit sub‑agent completed event with summary
    publish("subagent_completed", {
        "agent_id": agent.id,
        "goal": goal,
        "final_answer": agent.final_answer,
        "steps": [
            {"tool": s.get("tool"), "status": "success" if s.get("success") else "failed"}
            for s in agent.steps
        ],
        "duration": time.time() - start_time,
        "timestamp": time.time(),
    })
    return agent.final_answer


def spawn_agent(goal: str, tools: dict = None) -> str:
    goal = _sanitize_goal(goal)
    sub = Agent(goal=goal, tools=tools)
    _register_agent(sub)
    # Emit sub‑agent started event
    publish("subagent_started", {"agent_id": sub.id, "goal": goal, "timestamp": time.time()})
    return sub.id


# ── Tool registration ──


def _agent_spawn_tool(goal: str) -> str:
    aid = spawn_agent(goal)
    return f"Spawned sub-agent [{aid}] for: {goal[:100]}"


def _list_agents_tool() -> str:
    agents = list_agents()
    if not agents:
        return "No sub-agents currently running."
    lines = [f"{len(agents)} sub-agent(s):"]
    for a in agents:
        lines.append(f"  [{a['id']}] {a['goal'][:80]} — {a['status']} ({a['successful_steps']}/{a['step_count']} steps ok)")
    return "\n".join(lines)


def _get_agent_status_tool(agent_id: str) -> str:
    agent = get_agent(agent_id)
    if not agent:
        return f"No agent found with id '{agent_id}'."
    d = agent.to_dict()
    lines = [
        f"Agent [{d['id']}]",
        f"  Goal: {d['goal']}",
        f"  Status: {d['status']}",
        f"  Steps: {d['successful_steps']}/{d['step_count']} successful",
        f"  Tools used: {', '.join(d['tools_used']) if d['tools_used'] else 'none'}",
    ]
    if d.get("error"):
        lines.append(f"  Error: {d['error']}")
    if d.get("final_answer"):
        lines.append(f"  Final: {d['final_answer']}")
    if d.get("parent_id"):
        lines.append(f"  Parent: {d['parent_id']}")
    return "\n".join(lines)


AGENT_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "agent_spawn",
            "description": "Launch a sub-agent for a multi-step goal. The agent runs autonomously and returns a summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The goal or task for the sub-agent to complete"},
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_agents",
            "description": "List all sub-agents with IDs, goals, and status. Use this before calling get_agent_status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agent_status",
            "description": "Get detailed status of a sub-agent by ID. Returns goal, steps, tools, errors, final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "The 8-character hex agent ID (e.g. 'a1b2c3d4')"},
                },
                "required": ["agent_id"],
            },
        },
    },
]

AGENT_TOOLS = {
    "agent_spawn": _agent_spawn_tool,
    "list_agents": _list_agents_tool,
    "get_agent_status": _get_agent_status_tool,
}
