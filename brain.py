import datetime
import inspect
import json
import os
import re
import threading
import time

import requests
from dotenv import load_dotenv

import action_sandbox
import file_sandbox
import learner
from agent import needs_agent_loop, needs_planner, run_agent_loop, run_planner_loop
from config import (
    GEMINI_DAILY_LIMIT,
    MODEL_CONTEXT_LIMITS,
    SAFETY_PENDING_TTL,
)
from event_bus import subscribe
from memory import (
    build_memory_block,
    build_semantic_memory_block,
    forget_memory,
    get_all_memories,
    get_recent_summaries,
    save_memory,
)
from decision_log import new_request_id, log_decision
from tool_parser import (
    detect_and_parse,
    strip_json_tool_calls,
    strip_markdown_json_tool_calls,
    strip_tool_call_tags,
    strip_xml_tool_calls,
)
from tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from vector_memory import add_to_vector_memory

load_dotenv()

_rate_limited_providers: set = set()


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


GEMINI_API_KEY = _env("GOOGLE_GENAI_API_KEY")
NVIDIA_API_KEY = _env("NVIDIA_API_KEY")
NVIDIA_NEMOTRON_API_KEY = _env("NVIDIA_NEMOTRON_API_KEY")
GROQ_API_KEY = _env("GROQ_API_KEY")
if os.getenv("JARVIS_DISABLE_GROQ", "0") == "1":
    GROQ_API_KEY = ""
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
HUGGINGFACE_TOKEN = _env("HUGGINGFACE_TOKEN")

MOCK_PROVIDER_URL = os.getenv("MOCK_PROVIDER_URL", "http://localhost:8888").rstrip("/")
JARVIS_MOCK_PROVIDERS = os.getenv("JARVIS_MOCK_PROVIDERS", "0") == "1"
JARVIS_LLM_FIRST = os.getenv("JARVIS_LLM_FIRST", "1") == "1"
# Phase 2A routing policy: deterministic capability/health/latency/cost
# provider selection. Default OFF during dev — flip only after BEFORE/AFTER
# benchmark shows it winning. JARVIS_ROUTER_CHEAP gates the cheap intent
# classifier (Groq) used when the local NN is not confident.
JARVIS_ROUTER_POLICY = os.getenv("JARVIS_ROUTER_POLICY", "0") == "1"
JARVIS_ROUTER_CHEAP = os.getenv("JARVIS_ROUTER_CHEAP", "0") == "1"
# Latency policy (vNext): effort-tier routing via latency_policy.py. Pure
# routing-layer change — the E3/2B classifier and its gates are untouched.
JARVIS_LATENCY_POLICY = os.getenv("JARVIS_LATENCY_POLICY", "0") == "1"
JARVIS_LOCAL_ENABLED = os.getenv("JARVIS_LOCAL_ENABLED", "1") == "1"
JARVIS_EVAL_MODE = os.getenv("JARVIS_EVAL_MODE", "0") == "1"
JARVIS_LOCAL_INTENT_ENABLED = os.getenv("JARVIS_LOCAL_INTENT_ENABLED", "1") == "1"
JARVIS_LOCAL_INTENT_CONFIDENCE = float(os.getenv("JARVIS_LOCAL_INTENT_CONFIDENCE", "0.85"))
JARVIS_LOCAL_AGREEMENT_GATE = os.getenv("JARVIS_LOCAL_AGREEMENT_GATE", "0") == "1"
JARVIS_DAILY_BUDGET_USD = float(os.getenv("JARVIS_DAILY_BUDGET_USD", "0"))

_LOCAL_INTENT_THRESHOLDS = {
    intent: float(os.getenv(f"JARVIS_LOCAL_INTENT_CONFIDENCE_{intent.upper()}", str(JARVIS_LOCAL_INTENT_CONFIDENCE)))
    for intent in ("chat", "coding", "tool_use", "reasoning", "self_mod", "automation")
}


def _local_intent_threshold(intent: str) -> float:
    """Per-intent confidence gate; falls back to the global JARVIS_LOCAL_INTENT_CONFIDENCE."""
    return _LOCAL_INTENT_THRESHOLDS.get(intent, JARVIS_LOCAL_INTENT_CONFIDENCE)


_budget_warned = False


def _daily_budget_exceeded() -> tuple[bool, float, float]:
    """Check the daily cloud-LLM budget (JARVIS_DAILY_BUDGET_USD; 0 = unlimited)."""
    global _budget_warned
    if JARVIS_DAILY_BUDGET_USD <= 0:
        return False, 0.0, 0.0
    try:
        from jarvis_logger import get_daily_cost

        spent = get_daily_cost()
    except Exception:
        return False, 0.0, 0.0
    exceeded = spent >= JARVIS_DAILY_BUDGET_USD
    if exceeded and not _budget_warned:
        _budget_warned = True
        try:
            from event_bus import publish

            publish(
                "self_test",
                {
                    "timestamp": time.time(),
                    "message": (
                        f"Daily AI budget exhausted (${spent:.2f} of ${JARVIS_DAILY_BUDGET_USD:.2f}). "
                        "Cloud calls suspended until tomorrow."
                    ),
                    "phase": "budget",
                },
            )
        except Exception:
            pass
    return exceeded, spent, JARVIS_DAILY_BUDGET_USD

GEMINI_TOOL_MODEL = "gemini-2.5-flash"
GEMINI_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash"]
# NIM slot ladder (2C): functional slots replace the legacy numeric tiers.
# All models verified live on build.nvidia.com free endpoints (2026-08-13).
NIM_MODEL_FAST = [
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nemotron-3-nano-30b-a3b",
    "stepfun-ai/step-3.7-flash",
]
NIM_MODEL_CODING = [
    "minimaxai/minimax-m3",
    "openai/gpt-oss-20b",
    "deepseek-ai/deepseek-v4-flash-0731",
]
NIM_MODEL_FRONTIER = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "minimaxai/minimax-m3",
]
# Defined but NOT registered as an active routing slot: the E4 probe must
# prove reliable (<10s) latency first (2026-08-13 probe: 21.7s for 1 token).
NIM_MODEL_REASONING = ["nvidia/nemotron-3-super-120b-a12b"]
GROQ_MODEL = "llama-3.3-70b-versatile"
# OpenRouter: the ':free' slug for llama-3.1-8b was pulled (404 "unavailable
# for free" — provider suggests the plain, paid-but-cheap slug). Override in
# .env via OPENROUTER_MODEL.
OPENROUTER_MODEL = _env("OPENROUTER_MODEL") or "meta-llama/llama-3.1-8b-instruct"
POLLINATIONS_MODEL = "openai"
NEMOTRON_ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


USER_NAME = "Debasish"
USER_CITY = "Buffalo Grove"
DEBUG = os.getenv("JARVIS_DEBUG", "0").lower() in ("1", "true", "yes", "on")
GEMINI_MAX_TOOL_ROUNDS = 8
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_TIMEOUT = 600
# Hard time budget per provider slot in a tool-calling loop. Without it a
# single stalled model can burn minutes of API calls across the fallback
# chain (observed: NIM Fast 113s + NIM Coding 52s on one request).
PROVIDER_SLOT_TIME_BUDGET = 60

# ----- LLM‑first helper utilities -----

def _llm_choose(prompt: str, valid_options: set[str] | None = None, max_retries: int = 2) -> str | None:
    """Ask Nemotron Ultra (LLM‑first) for a short decision.
    Returns the stripped, lower‑cased response if it matches one of ``valid_options``
    (or any non‑empty string if ``valid_options`` is ``None``). Returns ``None`` on any error.
    """
    if not (JARVIS_LLM_FIRST and NVIDIA_NEMOTRON_API_KEY and _provider_available("Nemotron Ultra")):
        return None
    if JARVIS_MOCK_PROVIDERS:
        return None
    try:
        resp = _retry(lambda: ask_nemotron_ultra(prompt, [], timeout=45), max_retries=max_retries).strip().lower()
        if not resp:
            return None
        if valid_options is not None and resp not in valid_options:
            return None
        return resp
    except Exception as e:
        _debug(f"[LLM‑choose] error: {e}")
        return None

def _select_primary_provider(user_message: str, intent: str) -> str | None:
    """Deterministic primary selection (Phase 2A).

    Policy-on (JARVIS_ROUTER_POLICY=1): top eligible candidate per
    routing_policy scoring (capability fit x health x latency x cost).
    Policy-off: coding -> ``"nemotron_ultra"`` (DeepSeek v4 Flash is EOL —
    no LLM round-trip for arbitration). Returns a profile key or None.
    """
    if not JARVIS_LLM_FIRST:
        return None
    if JARVIS_ROUTER_POLICY:
        try:
            ordered = _policy_candidate_order(intent)
            top = ordered[0] if ordered else None
            if top and top.get("status") == "eligible":
                return top["provider"]
            _debug("[Router] policy found no eligible primary")
            return None
        except Exception as e:
            _debug(f"[Router] policy selection failed, defaulting: {e}")
    return "nemotron_ultra" if intent == "coding" else None


# ── Phase 2A routing policy wiring (routing_policy.py holds the pure logic) ──
# Profile key -> (display name, availability check, ask fn, model label).
_POLICY_PRIMARY_CALLS: dict[str, tuple[str, object, object, str]] = {
    "nemotron_ultra": (
        "Nemotron Ultra",
        lambda: bool(NVIDIA_NEMOTRON_API_KEY),
        lambda u: ask_nemotron_ultra(u, [], timeout=90),
        NEMOTRON_ULTRA_MODEL,
    ),
    "gemini": (
        "Gemini",
        lambda: _gemini_available(),
        lambda u: ask_gemini(u),
        GEMINI_TOOL_MODEL,
    ),
    "groq": (
        "Groq",
        lambda: bool(GROQ_API_KEY),
        lambda u: ask_groq(u, []),
        GROQ_MODEL,
    ),
    "nim_fast": (
        "NIM Fast",
        lambda: bool(NVIDIA_NEMOTRON_API_KEY),
        lambda u: ask_nim_fast(u, []),
        NIM_MODEL_FAST[0],
    ),
    "nim_coding": (
        "NIM Coding",
        lambda: bool(NVIDIA_NEMOTRON_API_KEY),
        lambda u: ask_nim_coding(u, []),
        NIM_MODEL_CODING[0],
    ),
    "nim_frontier": (
        "NIM Frontier",
        lambda: bool(NVIDIA_NEMOTRON_API_KEY),
        lambda u: ask_nemotron_ultra(u, [], timeout=90),
        NIM_MODEL_FRONTIER[0],
    ),
    "openrouter": (
        "OpenRouter",
        lambda: bool(OPENROUTER_API_KEY),
        lambda u: ask_openrouter(u, []),
        OPENROUTER_MODEL,
    ),
    "pollinations": (
        "Pollinations",
        lambda: True,
        lambda u: ask_pollinations(u, []),
        POLLINATIONS_MODEL,
    ),
}

_PROVIDER_KEY_BY_DISPLAY = {
    display: key for key, (display, _, _, _) in _POLICY_PRIMARY_CALLS.items()
}


def _observed_latency_ms(display_name: str) -> float | None:
    """Rolling avg latency (ms) from provider health, when observed."""
    try:
        h = get_provider_health().get(display_name, {})
        avg = h.get("avg_latency")
        return round(avg * 1000, 1) if avg else None
    except Exception:
        return None


def _policy_candidate_order(intent: str) -> list[dict]:
    """Score live providers for an intent via routing_policy (deterministic).

    Hard eligibility (unavailable / health < 30 -> INVALID) happens inside
    score_candidates; the result is decision-logged for telemetry (2A.4).
    """
    return _candidate_order_scored(intent)


def _effort_candidate_order(intent: str, weights: dict, prof: dict) -> list[dict]:
    """Latency-policy chain: score the fleet with effort-adjusted weights,
    filtered to providers that fit the task profile's constraints."""
    return _candidate_order_scored(
        intent,
        weights=weights,
        min_capability=prof.get("capability_floor"),
        max_latency_ms=prof.get("latency_cap_ms"),
    )


def _candidate_order_scored(
    intent: str,
    weights: dict | None = None,
    min_capability: float | None = None,
    max_latency_ms: float | None = None,
) -> list[dict]:
    """Assemble live candidate facts and score them (deterministic).

    Optional effort filters (latency-policy only):
      min_capability   drop providers whose capability fit is below this
      max_latency_ms   drop providers whose base latency exceeds this
    """
    from routing_policy import PROVIDER_PROFILES, capability_fit, score_candidates

    health_view = get_provider_health()
    cands = []
    for key, (display, available_fn, _fn, _model) in _POLICY_PRIMARY_CALLS.items():
        profile = PROVIDER_PROFILES.get(key, {})
        base_lat = profile.get("base_latency_ms")
        if max_latency_ms is not None and base_lat is not None and base_lat > max_latency_ms:
            continue
        if min_capability is not None and capability_fit(intent, profile) < min_capability:
            continue
        h = health_view.get(display, {})
        health = h.get("health_score", 100)
        latency_ms = _observed_latency_ms(display)
        if latency_ms is None and base_lat:
            latency_ms = base_lat
        try:
            avail = bool(available_fn()) and not _backoff_active(display)
        except Exception:
            avail = False
        cands.append(
            {"provider": key, "health": health, "latency_ms": latency_ms, "available": avail}
        )
    try:
        exceeded, spent, budget = _daily_budget_exceeded()
        budget_warning = bool(budget and budget > 0 and not exceeded and spent / budget > 0.75)
    except Exception:
        budget_warning = False
    ordered = score_candidates(intent, cands, budget_warning=budget_warning, weights=weights)
    _log_policy_decision(intent, ordered)
    return ordered


def _backoff_active(display_name: str) -> bool:
    """Backoff/circuit gate only — health threshold is scored by the policy.
    Returns True when the provider is currently cooling down (skip it)."""
    if display_name not in _provider_backoff_until:
        return False
    return _provider_backoff_until[display_name] > time.time()


def _log_policy_decision(intent: str, ordered: list[dict]) -> None:
    """2A.4: explainable routing telemetry — every candidate + components."""
    try:
        log_decision(
            phase="plan",
            decision="routing_policy_ordered",
            decision_source="routing_policy.score_candidates",
            measurable_inputs={
                "intent": intent,
                "policy_on": JARVIS_ROUTER_POLICY,
                "candidates": [
                    {
                        "provider": e["provider"],
                        "score": e.get("score"),
                        "status": e["status"],
                        "reason": e.get("reason"),
                        **e.get("components", {}),
                        "observed": e.get("observed", {}),
                    }
                    for e in ordered
                ],
            },
        )
    except Exception as e:
        _debug(f"[Router] telemetry log failed: {e}")


_local_override: str | None = None  # "local" | "cloud" | None (auto)


def set_local_override(mode: str | None) -> None:
    """Set a session-wide routing override: 'local', 'cloud', or None (auto).
    Switching to cloud frees the local model's ~2.5GB of RAM."""
    global _local_override
    _local_override = mode if mode in ("local", "cloud") else None
    if _local_override != "local":
        try:
            from local_provider import unload

            unload()
            _debug("[Router] Local MLX model unloaded (cloud mode)")
        except Exception:
            pass


def _try_local_routing(user_message: str, intent: str | None = None) -> str | None:
    """Handle 'use local'/'use cloud' overrides and simple-chat local replies.
    Returns a reply string when handled locally, None to continue to cloud."""
    global _last_provider_used, _last_model_used
    msg = (user_message or "").strip().lower()
    if msg in ("use local", "use the local model", "switch to local"):
        set_local_override("local")
        return "Switched to the local model for this session."
    if msg in ("use cloud", "use the cloud", "switch to cloud", "use online"):
        set_local_override("cloud")
        return "Switched to cloud providers for this session."
    if _local_override is not None and _local_override != "local":
        return None
    if not JARVIS_LOCAL_ENABLED:
        return None
    try:
        from local_provider import is_available

        if not is_available():
            return None
    except Exception:
        return None
    if intent is None:
        intent = classify_intent(user_message)
    if intent != "chat" or _estimate_task_complexity(user_message) > 4:
        return None
    try:
        from local_provider import JARVIS_LOCAL_MODEL, ask_local

        reply = ask_local(user_message)
    except Exception as e:
        _debug(f"[Router] Local provider error: {e}")
        reply = ""
    if not reply:
        return None
    _debug("[Router] Local MLX reply (offline-capable)")
    _last_provider_used = "local_mlx"
    _last_model_used = JARVIS_LOCAL_MODEL
    return reply




def _classify_error(error_text: str) -> str | None:
    """Deterministic error classification from error-text patterns.

    Labels: rate_limit, auth_error, timeout, quota, internal, unknown.
    Replaces the former LLM-first classification call (a full Nemotron
    round-trip per provider failure) with pattern matching already used
    elsewhere (see ``_retry`` and ``_ask_nim_loop``).
    """
    err = (error_text or "").lower()
    if not err:
        return "unknown"
    if "429" in err or "rate limit" in err or "rate_limit" in err or "too many requests" in err:
        return "rate_limit"
    if "401" in err or "403" in err or "unauthorized" in err or "invalid api key" in err:
        return "auth_error"
    if "permission denied" in err or "forbidden" in err:
        return "auth_error"
    if "timeout" in err or "timed out" in err or "read timeout" in err or "connection" in err:
        return "timeout"
    if "connect error" in err or "econnreset" in err or "econnrefused" in err or "socketerror" in err:
        return "timeout"
    if "quota" in err or "insufficient_quota" in err or "exceeded your current quota" in err:
        return "quota"
    if "500" in err or "502" in err or "503" in err or "504" in err or "unavailable" in err:
        return "internal"
    if "overloaded" in err or "internal server error" in err or "server error" in err:
        return "internal"
    if "did not resolve" in err or "malformed tool call" in err:
        # Quality failure, not an outage: the model spent its tool rounds and
        # never produced a final answer (or emitted garbage tool names).
        # Must NOT trip the circuit breaker — one bad request shouldn't take
        # down the whole provider chain.
        return "resolution"
    return "unknown"

def _handle_provider_failure(provider_name: str, exc: Exception) -> None:
    """Centralized failure handling that uses LLM error classification to decide backoff.
    Calls ``_record_provider_failure`` and ``_backoff_provider`` with a duration based on the error type.
    """
    label = _classify_error(str(exc))
    if label == "resolution":
        # Quality failure (tool loop never resolved) — record the health dip
        # but do NOT touch the circuit breaker: _backoff_provider(name, 0)
        # would still clamp the circuit open for 300s, and one bad request
        # cascading the whole fallback chain is exactly the incident this
        # classification exists to prevent.
        _record_provider_failure(provider_name)
        return
    if label == "rate_limit":
        backoff_secs = 1800
    elif label == "auth_error":
        backoff_secs = 0
    elif label == "timeout":
        backoff_secs = 300
    elif label == "quota":
        backoff_secs = 1200
    else:
        backoff_secs = CIRCUIT_BREAKER_TIMEOUT
    _record_provider_failure(provider_name)
    _backoff_provider(provider_name, seconds=backoff_secs)


def _summarize_conversation(recent_turns: int = 10) -> str:
    """LLM‑first summarization of the most recent conversation turns.
    Returns an empty string if disabled, unavailable, or the LLM call fails.
    """
    if not JARVIS_LLM_FIRST:
        return ""
    if len(conversation) < 2:
        return ""
    recent = conversation[-(recent_turns * 2):]
    prompt = (
        "Summarize the following conversation in 2-3 sentences. Preserve any tasks, "
        "decisions, and key facts. Respond with only the summary.\n\n"
    )
    for m in recent:
        role = m.get("role", "user")
        content = (m.get("content") or "")[:800]
        prompt += f"{role.title()}: {content}\n"
    return _llm_choose(prompt) or ""


conversation = []
_process_lock = threading.RLock()
pending_action = {"fn": None, "description": None, "expires_at": 0}
_learned_tools = {}
_loading_learned_tools = False
_pending_safe = {"tool": None, "fn": None, "args": None, "level": None, "expires_at": 0}
_pending_lock = threading.Lock()
_genai_client = None
_gemini_backoff_until = 0.0
_last_model_used = "unknown"
_last_provider_used = "unknown"
_GEMINI_USAGE_FILE = os.path.expanduser(os.getenv("JARVIS_GEMINI_USAGE_FILE", "~/.jarvis_gemini_usage.json"))
_PROVIDER_HEALTH_FILE = os.path.expanduser(os.getenv("JARVIS_PROVIDER_HEALTH_FILE", "~/.jarvis/provider_health.json"))
_provider_backoff_until = {}
_provider_consecutive_failures = {}
_provider_usage_count = {}
_provider_health_scores = {}
_provider_health: dict[str, dict] = {}  # {name: {total, successes, failures, avg_latency, last_success, last_failure, circuit_open, health_score}}
_nemotron_usage_count = 0
_tool_call_names: list[str] = []
_turn_memo_cache: dict[str, str] = {}
_current_request_id = ""
_last_user_message: str = ""
_resolved_project_root: str | None = None


# ─────────────────────────────────────────────
# CONVERSATION CONTEXT SYSTEM
# ─────────────────────────────────────────────
class ConversationState:
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_CONFIRM = "awaiting_confirm"
    AWAITING_RESPONSE = "awaiting_response"
    ERROR = "error"


class ConversationContext:
    def __init__(self):
        self.state = ConversationState.IDLE
        self.last_problem: str | None = None
        self.last_solution: str | None = None
        self.last_intent: str | None = None
        self.last_provider: str | None = None
        self.last_tools: list[str] = []
        self.last_error: str | None = None
        self.fragment_awaiting_context: bool = False
        self.intent_cache: dict[str, str] = {}

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "last_problem": self.last_problem[:200] if self.last_problem else None,
            "last_solution": self.last_solution[:200] if self.last_solution else None,
            "last_intent": self.last_intent,
            "last_provider": self.last_provider,
            "last_tools": self.last_tools,
            "fragment_awaiting_context": self.fragment_awaiting_context,
        }

    def detect_fragment(self, text: str) -> bool:
        t = text.lower().strip()
        fragments = {"solution", "answer", "explain", "soln", "ans"}
        if t in fragments:
            return True
        for f in fragments:
            if t.startswith(f + " ") or t.startswith(f + ","):
                return True
        return False

    def update(self, text: str, reply: str, intent: str, provider: str, tools: list[str], elapsed: float):
        self.state = ConversationState.IDLE
        self.last_provider = provider
        self.last_tools = tools
        problem_patterns = [
            r"\bproblem\s+\d+",
            r"\bquestion\s+\d+",
            r"\bchallenge\b",
            r"^(?:solve|answer|explain|find|fix|debug|implement|write)\b",
            r"\b(solve|answer|explain|find|fix|debug|implement|write)\s",
        ]
        if any(re.search(p, text.lower()) for p in problem_patterns):
            self.last_problem = text
            self.last_solution = reply
        self.last_intent = intent

    def add_message(self, role: str, content: str):
        """Inject a system-level note directly into the live conversation history
        so it actually reaches the model on the next call (not just stored on
        this context object, which build_system_prompt/get_working_memory never read).
        """
        conversation.append({"role": role, "content": content})
        if role == "system":
            self.last_solution = content[:200]


conversation_context = ConversationContext()


def get_conversation_context() -> dict:
    with _process_lock:
        return conversation_context.snapshot()


def _load_session_history(session_id: str) -> None:
    """Reload the persistent session's messages into the in-memory buffer."""
    from session_store import ensure_default_session, load_messages

    ensure_default_session()
    conversation.clear()
    conversation.extend(load_messages(session_id))


def _session_append(session_id: str, role: str, content: str) -> None:
    """Append a turn to the in-memory buffer and the persistent store."""
    from session_store import append_message

    conversation.append({"role": role, "content": content})
    append_message(session_id, role, content)


def reset_conversation(session_id: str = "default", timeout: float = 60.0) -> None:
    """Clear pending confirmations, in-memory context, and persisted history.

    Waits up to `timeout` seconds for the process lock (a request may be
    mid-turn); on timeout the persisted history is cleared anyway so a hung
    provider can't wedge reset, and the in-memory buffer is left for the
    in-flight turn to finish.
    """
    from session_store import clear_session

    clear_pending_safe()
    acquired = _process_lock.acquire(timeout=timeout)
    try:
        if acquired:
            conversation.clear()
            conversation_context.__init__()
    finally:
        if acquired:
            _process_lock.release()
    clear_session(session_id)


def _check_fragment(text: str, ctx: ConversationContext) -> str | None:
    if ctx.detect_fragment(text):
        ctx.fragment_awaiting_context = True
        if ctx.last_problem:
            return "Are you asking about the last problem I solved? Type 'yes' to continue, or paste the problem you want me to work on."
        else:
            return "It looks like you're asking for a solution or explanation, but I don't have context. Could you paste the problem or question you'd like me to work on?"
    return None


from jarvis_logger import generate_request_id, log_request

# ─────────────────────────────────────────────
# INTENT CLASSIFIER
# ─────────────────────────────────────────────
CODING_KEYWORDS = {
    "code",
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "rust",
    "golang",
    "bug",
    "debug",
    "fix",
    "compile",
    "syntax",
    "function",
    "class",
    "method",
    "repo",
    "github",
    "program",
    "script",
    "error",
    "traceback",
    "exception",
    "refactor",
    "implement",
    "write code",
    "api",
    "endpoint",
    "database",
    "sql",
    "query",
    "docker",
    "kubernetes",
    "terraform",
    "yaml",
    "json",
    "xml",
    "html",
    "css",
    "read_file",
    "write_file",
    "run_python",
    "scan_project",
    "search_in_files",
}

TOOL_USE_KEYWORDS = {
    "open",
    "launch",
    "start",
    "close",
    "quit",
    "weather",
    "forecast",
    "search",
    "find",
    "play",
    "pause",
    "send",
    "message",
    "email",
    "calendar",
    "event",
    "timer",
    "remind",
    "schedule",
    "browse",
    "navigate",
    "click",
    "screenshot",
    "screen",
    "spotify",
    "discord",
    "safari",
    "chrome",
    "volume",
    "download",
    "organize",
    "usage",
    "read",
    "show",
    "list",
    "recap",
    "create",
    "make",
    "disk",
    "cpu",
    "ram",
    "memory",
    "battery",
    "storage",
    "system",
    "ip",
}

# Only high-precision CAD tokens live here. Generic engineering words
# (feature, depth, dimension, part, body, sketch, hole, ...) were removed
# because they crashed into everyday language ("visual feature extraction",
# "flatten a nested list of arbitrary depth") and misrouted chat/coding
# requests into the Onshape handler. Strong tokens trigger alone; the rest
# (extrude/fillet/chamfer) only count alongside a second signal, see
# _cad_triggered().
CAD_KEYWORDS = {
    "onshape",
    "cad",
    "part studio",
    "partstudio",
    "extrude",
    "fillet",
    "chamfer",
}

CAD_STRONG_WORDS = ("onshape", "part studio", "partstudio", "cad")


def _cad_triggered(t: str) -> bool:
    """True only for unambiguous CAD/Onshape signals.

    Any Onshape URL counts. Otherwise a single strong token (onshape, cad,
    part studio) is enough — they have no everyday meaning. The weaker CAD
    verbs (extrude/fillet/chamfer) need a second distinct keyword so a lone
    "salmon fillet" or "fillet of cod" doesn't land in the Onshape handler.
    """
    if "cad.onshape.com" in t or "/documents/" in t:
        return True
    import re
    hits = re.findall(r"\b(" + "|".join(re.escape(k) for k in CAD_KEYWORDS) + r")\b", t)
    distinct = set(hits)
    if any(w in distinct for w in CAD_STRONG_WORDS):
        return True
    return len(distinct) >= 2


def _local_intent_predict(text: str) -> tuple[str | None, float]:
    """Predict intent with the on-device NN router. Returns (None, 0.0) when disabled/unavailable."""
    if not JARVIS_LOCAL_INTENT_ENABLED:
        return None, 0.0
    try:
        from jarvis_local_nn.integration.local_router import predict_intent

        return predict_intent(text)
    except Exception:
        return None, 0.0


_last_fine_intent: str | None = None
_last_fine_confidence: float = 0.0
_last_classifier_path: dict | None = None
_cheap_call_count: int = 0
_cheap_error_count: int = 0


def get_last_classifier_path() -> dict | None:
    """Classifier path (local_nn/cheap_groq/nemotron/keyword/cad/...) from the most
    recent classify_intent call, with confidence and raw output where available."""
    return _last_classifier_path


def get_last_fine_intent() -> tuple[str | None, float]:
    """Fine-grained intent + confidence from the most recent classify_intent call."""
    return _last_fine_intent, _last_fine_confidence


def _local_fine_predict(text: str, coarse: str) -> tuple[str | None, float]:
    """Two-stage: coarse bucket → specialist MLP. Returns (fine_intent, confidence).

    Per-class confidence gates are applied inside predict_fine_gated — below the
    gate the label is withheld so callers fall through to LLM routing.
    """
    global _last_fine_intent, _last_fine_confidence
    _last_fine_intent, _last_fine_confidence = None, 0.0
    try:
        from jarvis_local_nn.integration.specialist_router import predict_fine_gated

        fine, conf = predict_fine_gated(coarse, text)
        if fine is not None:
            _last_fine_intent, _last_fine_confidence = fine, conf
        return fine, conf
    except Exception:
        return None, 0.0


def _clear_fine_intent() -> None:
    """Reset fine-intent globals so stale values from a prior request never leak."""
    global _last_fine_intent, _last_fine_confidence, _last_classifier_path
    _last_fine_intent, _last_fine_confidence = None, 0.0
    _last_classifier_path = None
    _last_fine_intent, _last_fine_confidence = None, 0.0


def _log_classifier_path(
    path: str,
    intent: str,
    confidence: float | None = None,
    raw: dict | None = None,
) -> None:
    """2B.0 telemetry: record which classifier produced an intent decision."""
    measurable = {"path": path, "intent": intent}
    if confidence is not None:
        measurable["confidence"] = round(float(confidence), 3)
    if raw:
        measurable["raw"] = raw
    global _last_classifier_path
    _last_classifier_path = measurable
    log_decision("understand", "classifier_path", measurable_inputs=measurable)


def classify_intent(text: str) -> str:
    """Route requests into categories: coding, tool_use, reasoning, self_mod, chat.
    Primary classification uses keyword-based heuristics for precise patterns (self_mod),
    then LLM for ambiguous cases.
    """
    _clear_fine_intent()
    t = text.lower()
    # Self-modification — touching Jarvis's own source code (keyword-first for precision)
    self_mod_files = ("brain.py", "tts.py", "safety.py", "agent.py", "terminal.py", "server.py")
    self_mod_files_with_path = ("tools/", "config.py", "tools/__init__.py")
    self_mod_verbs = (
        "fix",
        "change",
        "modify",
        "update",
        "edit",
        "rewrite",
        "improve",
        "refactor",
        "debug",
        "read and fix",
        "modify yourself",
        "improve yourself",
        "your code",
        "your source",
        "yourself",
        "your own",
    )
    if any(f in t for f in self_mod_files):
        if any(v in t for v in self_mod_verbs):
            _debug(f"[Intent] classify_intent('{text}'): 'self_mod' — file + verb")
            _log_classifier_path("keyword", "self_mod")
            return "self_mod"
    if any(f in t for f in self_mod_files_with_path):
        _debug(f"[Intent] classify_intent('{text}'): 'self_mod' — file path")
        _log_classifier_path("keyword", "self_mod")
        return "self_mod"
    if any(k in t for k in ("my code", "jarvis code")):
        if any(v in t for v in self_mod_verbs):
            _debug(f"[Intent] classify_intent('{text}'): 'self_mod' — keyword + verb")
            _log_classifier_path("keyword", "self_mod")
            return "self_mod"

    # Memory triggers — handled directly in process(), not via tools
    memory_triggers = (
        "remember that",
        "remember my",
        "remember i",
        "keep in mind",
        "don't forget",
        "note that",
        "forget ",
        "erase ",
        "delete memory",
    )
    if any(k in t for k in memory_triggers):
        _debug(f"[Intent] classify_intent('{text}'): 'chat' — memory trigger")
        _log_classifier_path("memory", "chat")
        return "chat"

    # Knowledge/memory queries — handled directly in process()
    knowledge_triggers = (
        "what do you know",
        "what do you remember",
        "what have you learned",
        "what have you told",
    )
    if any(k in t for k in knowledge_triggers):
        _debug(f"[Intent] classify_intent('{text}'): 'chat' — knowledge query")
        _log_classifier_path("knowledge", "chat")
        return "chat"

    # CAD / Onshape — keyword-first detection before LLM (like self_mod).
    # Only unambiguous signals (see _cad_triggered): strong tokens/URLs alone,
    # weaker CAD verbs only with a second CAD keyword.
    if _cad_triggered(t):
        _debug(f"[Intent] classify_intent('{text}'): 'cad' — CAD keyword/URL")
        _log_classifier_path("keyword", "cad")
        return "cad"

    # Check intent cache (per-turn)
    if text in conversation_context.intent_cache:
        cached = conversation_context.intent_cache[text]
        _debug(f"[Intent] classify_intent('{text}'): '{cached}' — cached")
        _log_classifier_path("cache", cached)
        return cached

    # Local NN fast-path — skip cloud LLM when the on-device router is confident.
    # Enabled in eval mode too so evaluation measures production routing behavior.
    if not JARVIS_MOCK_PROVIDERS:
        _local_intent, _local_conf = _local_intent_predict(text)
        if _local_intent is not None and _local_conf >= _local_intent_threshold(_local_intent):
            conversation_context.intent_cache[text] = _local_intent
            _debug(f"[Intent] classify_intent('{text}'): '{_local_intent}' — local NN ({_local_conf:.2f})")
            # Two-stage: fine-grained intent from the bucket's specialist (best-effort)
            _fine, _fine_conf = _local_fine_predict(text, _local_intent)
            if _fine is not None:
                _debug(f"[Intent] fine: '{_fine}' ({_fine_conf:.2f})")
            agreement_suspicious = JARVIS_LOCAL_AGREEMENT_GATE and _fine is None
            if agreement_suspicious:
                _debug(
                    f"[Intent] agreement gate: coarse '{_local_intent}' ({_local_conf:.2f}) "
                    "but specialist withheld — escalating to LLM classify"
                )
            else:
                _log_classifier_path(
                    "local_nn", _local_intent, confidence=_local_conf,
                    raw={"fine": _fine, "fine_conf": _fine_conf},
                )
                return _local_intent

    # LLM‑first classification for remaining cases (≈90% of cases)
    if JARVIS_LLM_FIRST and not JARVIS_MOCK_PROVIDERS:
        if JARVIS_ROUTER_CHEAP and GROQ_API_KEY:
            # 2A.2: cheap Groq classifier — replaces the Nemotron classify
            # round-trip. It can only emit intent fields (never a provider).
            global _cheap_call_count, _cheap_error_count
            _cheap_call_count += 1
            try:
                parsed = _cheap_classify(text)
                classification = parsed["intent"] if parsed else None
            except Exception as e:
                _cheap_error_count += 1
                _debug(f"[Intent] cheap classifier error: {e}")
                classification = None
            if classification in {"coding", "tool_use", "reasoning", "self_mod", "chat"}:
                _debug(f"[Intent] cheap classifier gave '{classification}' (fine={parsed.get('fine_intent')})")
                escalation = classification == "tool_use" and parsed.get("tool_required") is False
                # 2B.3: apply the frozen per-intent confidence gate to the cheap path too
                cheap_conf = parsed.get("confidence")
                if not escalation and cheap_conf is None:
                    _debug("[Intent] cheap classifier gave no confidence — escalating to LLM classify")
                    escalation = True
                elif not escalation and cheap_conf < _local_intent_threshold(classification):
                    _debug(
                        f"[Intent] cheap confidence {cheap_conf:.2f} below gate "
                        f"{_local_intent_threshold(classification):.2f} — escalating to LLM classify"
                    )
                    escalation = True
                if not escalation and JARVIS_LOCAL_AGREEMENT_GATE and not parsed.get("fine_intent"):
                    _debug("[Intent] agreement gate: cheap intent without fine intent — escalating to LLM classify")
                    escalation = True
                if escalation:
                    _debug("[Intent] cheap said tool_use but tool_required=false — escalating to LLM classify")
                else:
                    try:
                        if parsed.get("fine_intent"):
                            _last_fine_intent, _last_fine_confidence = parsed["fine_intent"], 0.6
                    except Exception:
                        pass
                    _log_classifier_path(
                        "cheap_groq", classification, confidence=parsed.get("confidence"), raw=parsed
                    )
                    conversation_context.intent_cache[text] = classification
                    return classification
        if NVIDIA_NEMOTRON_API_KEY and _provider_available("Nemotron Ultra"):
            try:
                llm_prompt = (
                    "Classify the following user request into one of these categories: "
                    "coding, tool_use, reasoning, self_mod, chat. Respond with only the category name.\n\n"
                    f"User request: {text}\n"
                )
                # Use the low‑level Nemotron Ultra call to avoid routing loops
                classification = (
                    _retry(lambda: ask_nemotron_ultra(llm_prompt, [], timeout=45), max_retries=2).strip().lower()
                )
                if classification in {"coding", "tool_use", "reasoning", "self_mod", "chat"}:
                    _debug(f"[Intent] LLM classification gave '{classification}'")
                    _log_classifier_path("nemotron", classification)
                    # If LLM says 'chat' but text matches reasoning patterns, defer to keyword fallback
                    reasoning_patterns = (
                        "why does", "why is", "why are", "why would", "why did", "why do",
                        "how do", "how does", "how can i", "how would", "how is it",
                        "explain", "analyze", "design a", "design an", "strategy",
                        "compare", "evaluate", "pros and cons", "think about",
                        "what is the", "describe", "tell me about",
                    )
                    if classification == "chat" and any(k in t for k in reasoning_patterns):
                        _debug("[Intent] LLM said 'chat' but reasoning pattern matched — deferring to keyword fallback")
                    else:
                        conversation_context.intent_cache[text] = classification
                        return classification
            except Exception as e:
                _debug(f"[Intent] LLM classification error: {e}")

    # Keyword fallback for remaining cases

    # Coding — high-confidence check first (tool keywords may overlap)
    coding_hit = any(k in t for k in CODING_KEYWORDS)
    tool_hit = any(k in t for k in TOOL_USE_KEYWORDS)

    if coding_hit and not tool_hit:
        _debug(f"[Intent] classify_intent('{text}'): 'coding' — coding keyword")
        _log_classifier_path("keyword", "coding")
        return "coding"

    # Both coding and tool keywords — disambiguate
    if coding_hit and tool_hit:
        tool_verbs = (
            "open",
            "launch",
            "go to",
            "visit",
            "browse",
            "search",
            "find",
            "play",
            "pause",
            "send",
            "message",
            "weather",
            "timer",
            "navigate",
            "click",
            "screenshot",
        )
        if any(k in t for k in tool_verbs):
            _debug(f"[Intent] classify_intent('{text}'): 'tool_use' — both keywords, disambig→tool")
            _log_classifier_path("keyword", "tool_use")
            return "tool_use"
        _debug(f"[Intent] classify_intent('{text}'): 'coding' — both keywords, disambig→coding")
        _log_classifier_path("keyword", "coding")
        return "coding"

    # Tool use
    if tool_hit:
        _debug(f"[Intent] classify_intent('{text}'): 'tool_use' — tool keyword")
        _log_classifier_path("keyword", "tool_use")
        return "tool_use"

    # Reasoning — use phrases to avoid matching casual greetings like "hello how are you"
    reasoning_patterns = (
        "why does",
        "why is",
        "why are",
        "why would",
        "why did",
        "why do",
        "how do",
        "how does",
        "how can i",
        "how would",
        "how is it",
        "explain",
        "analyze",
        "design a",
        "design an",
        "strategy",
        "compare",
        "evaluate",
        "pros and cons",
        "think about",
        "what is the",
        "describe",
        "tell me about",
    )
    if any(k in t for k in reasoning_patterns):
        _debug(f"[Intent] classify_intent('{text}'): 'reasoning' — reasoning pattern")
        _log_classifier_path("keyword", "reasoning")
        return "reasoning"

    # Capability/gap analysis — questions about what Jarvis can do or is missing
    capability_gap_patterns = [
        r"\bwhat (should|could|can|do) (i |we |you )?(add|build|create|implement|make)\b",
        r"\bwhat'?s (missing|lacking|needed)\b",
        r"\b(what|single) (highest-leverage|most important|biggest|key)\b.*(missing|gap|lack|need)\b",
        r"\bgap\b.*(feature|capability|functionality)\b",
        r"\b(feature|capability|functionality).*(missing|lack|need)\b",
        r"\bwhat should we build next\b",
        r"\bwhat can you( not)? do\b",
        r"\bwhat are you (missing|lacking)\b",
        r"\b(how )?can i (improve|enhance|upgrade)\b.*(you|jarvis)\b",
    ]
    if any(re.search(p, t) for p in capability_gap_patterns):
        _debug(f"[Intent] classify_intent('{text}'): 'tool_use' — capability/gap analysis")
        _log_classifier_path("keyword", "tool_use")
        return "tool_use"

    # File extension hint — .py files imply coding
    if re.search(r"\b\w+\.(py|js|ts|java|cpp|c|h|go|rs|rb|sh)\b", t):
        _debug(f"[Intent] classify_intent('{text}'): 'coding' — file extension")
        _log_classifier_path("keyword", "coding")
        return "coding"

    _debug(f"[Intent] classify_intent('{text}'): 'chat' — default fallback")
    _log_classifier_path("keyword", "chat")
    conversation_context.intent_cache[text] = "chat"
    return "chat"

# ----- Event handling -----
def _event_to_message(payload):
    # Determine display text based on payload keys
    if 'alert_type' in payload:
        return f"[PROACTIVE] {payload.get('alert_type').upper()}: {payload.get('message', '')}"
    elif payload.get('agent_id'):
        # Sub‑agent events (started, progress, completed)
        if payload.get('final_answer'):
            # Completed
            return f"[SUB-AGENT] Goal: {payload.get('goal')}\nResult: {payload.get('final_answer')}"
        elif payload.get('step'):
            return f"[SUB-AGENT] Step {payload.get('step')} progress…"
        else:
            return f"[SUB-AGENT] Started: {payload.get('goal')}"
    else:
        return f"[EVENT] {payload}"

def _handle_event(payload):
    try:
        content = _event_to_message(payload)
        conversation.append({"role": "assistant", "content": content})
    except Exception as e:
        _debug(f"[Event] Failed to handle event: {e}")

# Register subscriptions for relevant event types (run once on import)
subscribe('proactive_alert', _handle_event)
subscribe('subagent_started', _handle_event)
subscribe('subagent_progress', _handle_event)
subscribe('subagent_completed', _handle_event)


def _resolve_project_root(path_hint: str = "~/Jarvis") -> str:
    """Resolve and cache the project root path once per session."""
    global _resolved_project_root
    if _resolved_project_root is None:
        resolved = os.path.realpath(os.path.expanduser(path_hint))
        if os.path.isdir(resolved):
            _resolved_project_root = resolved
    return _resolved_project_root or os.path.realpath(os.path.expanduser("~/Jarvis"))


def _get_project_root() -> str:
    """Get the cached project root or resolve it."""
    if _resolved_project_root:
        return _resolved_project_root
    return _resolve_project_root()


# ─────────────────────────────────────────────
# TASK COMPLEXITY ESTIMATOR
# ─────────────────────────────────────────────
def _estimate_task_complexity(text: str) -> int:
    """
    Estimate task complexity and return appropriate max_iterations (4-12 range).

    Considers:
    - Number of implied steps
    - Whether coding/research is needed
    - Whether multiple tools/files involved
    - User's explicit complexity hints
    """
    t = (text or "").lower()

    # Very complex: full projects, migrations, end-to-end builds
    very_complex_patterns = [
        r"\b(full[- ]stack|end[- ]to[- ]end|complete app|full app)\b",
        r"\b(migrate|refactor|rewrite).*(app|system|codebase)\b",
        r"\b(build|create).*(from scratch|from ground up)\b",
        r"\bentire\s+(app|system|website|platform)\b",
    ]
    if any(re.search(p, t) for p in very_complex_patterns):
        return 12

    # Complex: coding, debugging, multi-file work
    complex_patterns = [
        r"\b(fix|debug|troubleshoot).*(bug|error|issue|problem)\b|\b(bug|error|issue|problem).*(fix|debug|troubleshoot)\b",
        r"\b(build|create|make).*(app|tool|script|program)\b",
        r"\bimplement\s+(feature|functionality|system)\b",
        r"\b(multiple files|several components|modular)\b",
    ]
    if any(re.search(p, t) for p in complex_patterns):
        return 8

    # Medium: research, comparison, multi-step tasks
    medium_patterns = [
        r"\b(compare|analyze|research|investigate)\b",
        r"\b(find|gather).*(and|then)\b",
        r"\b(comprehensive|thorough|detailed)\b",
        r"\bstep by step|systematic|methodical\b",
    ]
    if any(re.search(p, t) for p in medium_patterns):
        return 6

    # Simple: single tool, factual queries
    simple_patterns = [
        r"\b(weather|time|date|open|quit|search|find)\b",
        r"\b(system status|cpu|ram|disk)\b",
        r"\b(spotify|calendar|message)\b",
        r"\b(what'?s|tell me|show me|list)\b",
    ]
    if any(re.search(p, t) for p in simple_patterns):
        return 4

    return 6


# ─────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────


def _default_health_entry() -> dict:
    return {
        "total": 0,
        "successes": 0,
        "failures": 0,
        "avg_latency": 0.0,
        "last_success": None,
        "last_failure": None,
        "circuit_open": False,
        "health_score": 100,
    }


def _record_provider_failure(provider_name: str):
    _provider_consecutive_failures[provider_name] = _provider_consecutive_failures.get(provider_name, 0) + 1
    count = _provider_consecutive_failures[provider_name]
    _provider_health_scores[provider_name] = max(0, _provider_health_scores.get(provider_name, 100) - 10)
    h = _provider_health.setdefault(provider_name, _default_health_entry())
    h["health_score"] = _provider_health_scores[provider_name]
    h["failures"] += 1
    h["total"] += 1
    h["last_failure"] = datetime.datetime.now().isoformat()
    if count >= CIRCUIT_BREAKER_THRESHOLD:
        _backoff_provider(provider_name, CIRCUIT_BREAKER_TIMEOUT)
        h["circuit_open"] = True
        _debug(f"Circuit breaker tripped for {provider_name} ({count} consecutive failures)")
        _provider_consecutive_failures[provider_name] = 0
    _save_provider_health()


def _record_provider_success(provider_name: str, latency: float = 0.0):
    _provider_consecutive_failures.pop(provider_name, None)
    _provider_backoff_until.pop(provider_name, None)
    _provider_health_scores[provider_name] = min(100, _provider_health_scores.get(provider_name, 100) + 5)
    _provider_usage_count[provider_name] = _provider_usage_count.get(provider_name, 0) + 1
    h = _provider_health.setdefault(provider_name, _default_health_entry())
    h["health_score"] = _provider_health_scores[provider_name]
    h["successes"] += 1
    h["total"] += 1
    h["last_success"] = datetime.datetime.now().isoformat()
    h["circuit_open"] = False
    h.pop("circuit_open_until", None)
    # Rolling average latency
    prev_avg = h["avg_latency"]
    n = h["successes"]
    h["avg_latency"] = ((prev_avg * (n - 1)) + latency) / n if n > 1 else latency
    _save_provider_health()


def get_provider_health() -> dict[str, dict]:
    """Return current provider health data for the dashboard."""
    now_ts = datetime.datetime.now().timestamp()
    result = {}
    for name, h in sorted(_provider_health.items()):
        entry = dict(h)
        entry["backoff_until"] = _provider_backoff_until.get(name)
        if entry["backoff_until"]:
            entry["backoff_remaining_s"] = max(0, entry["backoff_until"] - now_ts)
        entry["consecutive_failures"] = _provider_consecutive_failures.get(name, 0)
        result[name] = entry
    # Include any providers in usage_count but not yet in _provider_health
    for name in _provider_usage_count:
        if name not in result:
            result[name] = _default_health_entry()
            result[name]["usage_count"] = _provider_usage_count.get(name, 0)
    return result


def _save_provider_health():
    """Persist provider health state to disk (Phase 1.6)."""
    try:
        os.makedirs(os.path.dirname(_PROVIDER_HEALTH_FILE), exist_ok=True)
        with open(_PROVIDER_HEALTH_FILE, "w") as fh:
            json.dump(_provider_health, fh, indent=2)
    except Exception as e:
        _debug(f"Failed to save provider health: {e}")


def _load_provider_health():
    """Restore provider health state from disk (Phase 1.6)."""
    try:
        if os.path.exists(_PROVIDER_HEALTH_FILE):
            with open(_PROVIDER_HEALTH_FILE, "r") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                _provider_health.clear()
                for name, h in data.items():
                    _provider_health[name] = h
                    _provider_health_scores[name] = h.get("health_score", 100)
    except Exception as e:
        _debug(f"Failed to load provider health: {e}")


def _load_gemini_usage():
    try:
        if not os.path.exists(_GEMINI_USAGE_FILE):
            return {"date": datetime.date.today().isoformat(), "count": 0}
        with open(_GEMINI_USAGE_FILE, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {"date": datetime.date.today().isoformat(), "count": 0}
        return data
    except Exception:
        return {"date": datetime.date.today().isoformat(), "count": 0}


def _save_gemini_usage(data: dict):
    try:
        with open(_GEMINI_USAGE_FILE, "w") as fh:
            json.dump(data, fh)
    except Exception:
        pass


def _reset_gemini_usage_if_needed():
    data = _load_gemini_usage()
    today = datetime.date.today().isoformat()
    if data.get("date") != today:
        data = {"date": today, "count": 0}
        _save_gemini_usage(data)
    return data


def _get_gemini_usage_count() -> int:
    data = _reset_gemini_usage_if_needed()
    return int(data.get("count", 0))


def _increment_gemini_usage() -> int:
    data = _load_gemini_usage()
    today = datetime.date.today().isoformat()
    if data.get("date") != today:
        data = {"date": today, "count": 0}
    try:
        data["count"] = int(data.get("count", 0)) + 1
    except Exception:
        data["count"] = 1
    _save_gemini_usage(data)
    return int(data["count"])


def _notify_gemini_disabled(reason: str):
    """Speak a short notification explaining why Gemini has been disabled."""
    try:
        from tts import speak

        if reason == "daily_limit":
            msg = "Gemini has reached its daily request limit and will be disabled until tomorrow."
        else:
            msg = "Gemini API is currently rate-limited and will be disabled until rate limits clear."

        # Best-effort: use interrupt to notify immediately.
        speak(msg, interrupt=True)
    except Exception as e:
        _debug(f"Failed to notify user about Gemini disable: {e}")


def _debug_payload(payload):
    try:
        return json.dumps(payload, default=str, indent=None)[:2000]
    except Exception:
        try:
            if hasattr(payload, "to_dict"):
                return json.dumps(payload.to_dict(), default=str, indent=None)[:2000]
        except Exception:
            pass
        return repr(payload)[:2000]


def _debug_provider_response(provider_name: str, label: str, response):
    if not DEBUG:
        return
    _debug(f"{provider_name} {label}: {_debug_payload(response)}")


def _backoff_provider(provider_name: str, seconds: int = 600, exponential: bool = True):
    failures = _provider_consecutive_failures.get(provider_name, 0)
    if exponential and failures > 0:
        seconds = min(seconds * (2**failures), 3600)
    now = datetime.datetime.now().timestamp()
    _provider_backoff_until[provider_name] = now + seconds
    # circuit_open_until: separate immutable timeout — doesn't reset until cleared by success
    circuit_seconds = max(seconds, 300)
    h = _provider_health.setdefault(provider_name, _default_health_entry())
    h["circuit_open_until"] = now + circuit_seconds
    h["circuit_open"] = True
    _provider_health_scores[provider_name] = max(0, _provider_health_scores.get(provider_name, 100) - 20)
    health_score = _provider_health_scores.get(provider_name, 100)
    _debug(f"Backing off {provider_name} for {seconds}s (circuit until +{circuit_seconds}s, health={health_score})")


def _provider_available(provider_name: str) -> bool:
    now = datetime.datetime.now().timestamp()
    if now < _provider_backoff_until.get(provider_name, 0):
        return False
    h = _provider_health.get(provider_name, {})
    circuit_until = h.get("circuit_open_until", 0)
    if now < circuit_until:
        return False
    return True


_DEBUG_LOG_PATH = os.path.join(os.path.expanduser("~/.jarvis/logs"), "debug.log")


def _debug(message: str):
    if DEBUG:
        print(f"  [DEBUG] {message}")
        try:
            os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
            with open(_DEBUG_LOG_PATH, "a") as f:
                f.write(f"{datetime.datetime.now().isoformat()} {message}\n")
        except OSError:
            pass


def get_runtime_status() -> dict:
    return {
        "provider": _last_provider_used,
        "model_preferred": NEMOTRON_ULTRA_MODEL,
        "model_fallback": GROQ_MODEL,
        "model_last_used": _last_model_used,
        "api_key_configured": any([NVIDIA_NEMOTRON_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY]),
        "nemotron_usage": _nemotron_usage_count,
        "gemini_usage": _get_gemini_usage_count(),
        "providers": {
            "nemotron_ultra": bool(NVIDIA_NEMOTRON_API_KEY),
            "gemini": bool(GEMINI_API_KEY),
            "groq": bool(GROQ_API_KEY),
            "openrouter": bool(OPENROUTER_API_KEY),
            "pollinations": True,
            "huggingface": bool(HUGGINGFACE_TOKEN),
            "nvidia_nim": bool(NVIDIA_NEMOTRON_API_KEY),
        },
    }


def check_providers() -> dict:
    providers = {
        "nemotron_ultra": bool(NVIDIA_NEMOTRON_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
        "groq": bool(GROQ_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY),
        "nvidia_nim": bool(NVIDIA_NEMOTRON_API_KEY),
        "pollinations": True,
        "huggingface": bool(HUGGINGFACE_TOKEN),
    }
    labels = {
        "nemotron_ultra": "NVIDIA Nemotron Ultra (primary, tool-first + final)",
        "gemini": "Gemini 2.5 Flash (fallback, tool calling)",
        "groq": "Groq llama-3.3-70b (fallback)",
        "openrouter": "OpenRouter llama-3.1-8b (last-resort)",
        "nvidia_nim": "NVIDIA NIM (fallback tiers)",
        "pollinations": "Pollinations.ai (no-key emergency)",
        "huggingface": "HuggingFace (embeddings/downloads)",
    }
    print("  Checking AI providers...")
    for key, configured in providers.items():
        status = "OK" if configured else "MISSING"
        print(f"    {status}: {labels[key]}")
    return providers


def _get_client():
    global _genai_client
    if _genai_client:
        return _genai_client
    if not GEMINI_API_KEY:
        print("  No Gemini API key configured.")
        return None
    try:
        from google import genai

        if JARVIS_MOCK_PROVIDERS:
            from google.genai import types as _gtypes

            _genai_client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=_gtypes.HttpOptions(
                    baseUrl=MOCK_PROVIDER_URL,
                    timeout=45_000,
                ),
            )
        else:
            from google.genai import types as _gtypes

            _genai_client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=_gtypes.HttpOptions(timeout=45_000),
            )
        print(f"  Gemini client ready. Preferred: {', '.join(GEMINI_MODELS)}")
        return _genai_client
    except Exception as e:
        print(f"  Gemini init failed: {e}")
        return None


def _gemini_available() -> bool:
    try:
        from google import genai  # noqa: F401

        return bool(GEMINI_API_KEY)
    except ImportError:
        return False


def _internet_available() -> bool:
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False


def _on_tool_learned(name, fn, description):
    _learned_tools[name] = fn
    TOOL_REGISTRY[name] = fn
    TOOL_DEFINITIONS.append(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    if not _loading_learned_tools:
        try:
            from jarvis_local_nn.training.auto_retrain import schedule_retrain

            schedule_retrain()
        except Exception:
            pass


def _iter_unique_tool_definitions():
    seen = set()
    for tool_def in TOOL_DEFINITIONS:
        fn = tool_def.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        if name in seen:
            _debug(f"[Tools] Skipping duplicate tool definition: {name}")
            continue
        seen.add(name)
        yield tool_def


SYSTEM_INFO = {}


def init():
    from system_info import fetch_system_info

    global SYSTEM_INFO
    SYSTEM_INFO = fetch_system_info()
    global _loading_learned_tools
    _loading_learned_tools = True
    try:
        learned = learner.load_learned_tools()
    finally:
        _loading_learned_tools = False
    for name, fn in learned.items():
        if name not in _learned_tools:
            _on_tool_learned(name, fn, f"learned tool: {name}")
    from tts import speak

    learner.init(speak, _on_tool_learned)
    _get_client()
    _load_provider_health()
    check_providers()
    warm_up_providers()
    # Phase 1.2: Pre-warm MiniLM embedding model in background
    import threading as _t

    _t.Thread(target=lambda: __import__("vector_memory").prewarm_minilm(), daemon=True).start()
    _load_plugins()
    print(f"  Brain initialized. Nemotron requests: {_nemotron_usage_count}")


def warm_up_providers():
    """Non-blocking provider warm-up on startup. Daemon threads, max 3s total wait."""
    import threading
    import time

    def _warm_nim():
        try:
            _debug("[Warm-up] Warming NIM endpoint...")
            requests.head(
                "https://ai.api.nvidia.com/v1/gr/meta/llama-3.3-70b-instruct/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_NEMOTRON_API_KEY}"},
                timeout=5,
            )
            _debug("[Warm-up] NIM endpoint warm")
        except Exception:
            pass

    def _warm_groq():
        try:
            _debug("[Warm-up] Warming Groq...")
            requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=5)
            _debug("[Warm-up] Groq warm")
        except Exception:
            pass

    def _warm_gemini():
        try:
            _debug("[Warm-up] Warming Gemini...")
            if _genai_client:
                _genai_client.models.get("gemini-2.5-flash")
                _debug("[Warm-up] Gemini warm")
        except Exception:
            pass

    threads = []
    if NVIDIA_NEMOTRON_API_KEY:
        t = threading.Thread(target=_warm_nim, daemon=True)
        t.start()
        threads.append(t)
    if GROQ_API_KEY:
        t = threading.Thread(target=_warm_groq, daemon=True)
        t.start()
        threads.append(t)
    if GEMINI_API_KEY:
        t = threading.Thread(target=_warm_gemini, daemon=True)
        t.start()
        threads.append(t)

    deadline = time.time() + 3.0
    for t in threads:
        remaining = deadline - time.time()
        if remaining > 0:
            t.join(timeout=min(remaining, 2.0))


def _load_plugins():
    try:
        from plugin_manager import load_all_plugins, print_plugin_status

        results = load_all_plugins()
        loaded = [r for r in results if r.get("ok")]
        if loaded:
            for r in loaded:
                print(f"  Plugin loaded: {r['name']} ({r['tools']} tools, {r['providers']} providers)")
        print_plugin_status()
    except Exception as e:
        _debug(f"Plugin load error: {e}")


def build_system_prompt(query: str = ""):
    now = datetime.datetime.now().strftime("%A, %B %d %Y, %I:%M %p")
    device_block = (
        f"\n- Model: {SYSTEM_INFO.get('model', 'Unknown')}\n- Chip: {SYSTEM_INFO.get('chip', 'Unknown')}\n- RAM: {SYSTEM_INFO.get('ram', 'Unknown')}\n- OS: {SYSTEM_INFO.get('os', 'Unknown')}"
    )
    memory_block = build_memory_block()
    learned_str = ", ".join(_learned_tools.keys()) if _learned_tools else "none yet"

    semantic_block = ""
    if query:
        semantic = build_semantic_memory_block(query)
        if semantic:
            semantic_block = f"\n\nSemantically relevant context:\n{semantic}"
        try:
            from associative_memory import build_association_context

            associations = build_association_context(query)
            if associations:
                semantic_block += f"\n\nAssociative context:\n{associations}"
        except Exception:
            pass

    return f"""You are Jarvis, a smart proactive AI assistant running locally on {USER_NAME}'s Mac.
You have access to tools — use them when needed for real data.

CRITICAL RULES:
- When a tool returns a result, USE THAT RESULT.
- NEVER pretend to run commands or hallucinate outputs.
- NEVER narrate "saved", "created", "wrote", or "added" without calling the corresponding tool.
  Example: If user gives an API key, call save_api_key() — do NOT just say "key saved".
  Example: If user asks to write a file, call write_file() — do NOT just say "file created".
- For weather: call get_weather_detailed with NO arguments.
- For searches: call web_search with query only.
- For get_recent_events: pass only hours.
- You have access to file tools: read_file, write_file, create_file, list_directory
- You have access to code tools: run_python, scan_project_structure, search_in_files
- You can read, write, and execute code — you are NOT limited to conversation
- Keep responses short (1-3 sentences), no markdown.
- Say what you're doing briefly.
- When asked about capabilities, missing features, or setup review: call inspect_capabilities(). Do NOT scan source code or list directories to discover capabilities.
- Knowledge graph extraction runs automatically after each response. Do NOT call graph extraction tools manually.

Browser and UI automation:
- Screenshots show the frontmost app. Before read_screen or find_on_screen, call focus_app("Safari") or pass app_name="Safari".
- find_on_screen does not click. For Discord in Safari: browser_navigate(discord.com), browser_quick_search(query=channel name), then type_text + press_key enter — or use discord_open_and_send.
- Prefer browser_quick_search / discord_open_channel over repeated find_on_screen.
- If macOS shows an osascript prompt, tell the user to allow Terminal in System Settings → Privacy & Security → Accessibility.

Facts:
- User: {USER_NAME}, Location: {USER_CITY}, Illinois
- Date/Time: {now}
- System:{device_block}
- Project root: {_get_project_root()} (use this for file operations)

Memory:{memory_block}{semantic_block}

Learned capabilities: {learned_str}"""


def _tool_loop_exhausted(tool_results: list[str] | None = None) -> str:
    if tool_results:
        tail = "; ".join(tool_results[-4:])
        return f"I hit my step limit but completed several actions: {tail[:450]}. Say continue to resume. For Discord, use discord_open_and_send after you're logged in."
    return "I ran into an issue — likely too many tool steps. For Discord in Safari, try: open Safari, log into discord.com, then ask me to open a channel and send a message."


def _is_tool_use_failed(exc: Exception) -> bool:
    err = str(exc).lower()
    return "tool_use_failed" in err or "tool call validation" in err


def _parse_failed_tool_use(exc: Exception) -> tuple[str, dict] | None:
    """Extract tool name/args from Groq's failed_generation error text."""
    text = str(exc)
    patterns = [
        r"function=(\w+)\s*(\{[^}]+\})",
        r"<function=(\w+)\s*(\{[^}]+\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        name = match.group(1)
        raw_args = match.group(2)
        try:
            args = json.loads(raw_args)
            if isinstance(args, dict):
                return name, args
        except Exception:
            pass
        query_match = re.search(r'"query"\s*:\s*"([^"]+)"', raw_args)
        if query_match:
            return name, {"query": query_match.group(1)}
    return None


def _build_search_query(user_message: str) -> str:
    text = (user_message or "").strip()
    if re.search(r"what i mentioned|that i mentioned|my last (?:question|message)", text, re.I):
        for msg in reversed(conversation):
            if msg.get("role") == "user" and msg.get("content", "").strip() != text:
                prior = msg["content"].strip()
                if re.search(r"\binstagram\b|\binsta\b", prior, re.I) and re.search(r"\bvisit", prior, re.I):
                    return "can you see whose Instagram profiles someone visits"
                if re.search(r"\bhow\b", prior, re.I):
                    return prior[:250]
                return f"how to {prior[:200]}"
    if re.search(r"\binstagram\b|\binsta\b", text, re.I) and re.search(r"\bvisit", text, re.I):
        return "can you see whose Instagram profiles someone visits"
    cleaned = re.sub(
        r"^(?:(?:ok|hey|jarvis)[,.\s]+)*(?:(?:could|can) you\s+)?(?:please\s+)?(?:first\s+)?",
        "",
        text,
        flags=re.I,
    )
    cleaned = re.sub(
        r"^(?:search(?:\s+up)?|look\s+up)\s*(?:how\s+to\s+(?:do\s+)?)?",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    return (cleaned or text)[:250]


# Fine-grained intent → safe no-arg data tools for local prefetch (two-stage router)
_FINE_PREFETCH = {
    "weather_current": ("get_weather_detailed", {}),
    "sys_info": ("get_system_info", {}),
    "disk_usage": ("disk_usage", {}),
}


def _prefetch_tools_for_message(user_message: str) -> list[str]:
    """Run obvious tools locally so fallback LLMs (Groq) don't need to call them."""
    results = []
    t = (user_message or "").lower()

    # Smart weather detection — only call get_weather_detailed for current local weather
    weather_only_patterns = [
        r"\b(?:what'?s|what is|current|today'?s)\s+(?:the\s+)?weather\b",
        r"\bweather\s+(?:now|outside|like|today|currently)\b",
        r"\b(?:temperature|temp|rain|snow|forecast|humidity)\b",
    ]
    is_weather_only = any(re.search(p, t) for p in weather_only_patterns)

    # Check if user wants search/research (not just current weather)
    search_patterns = [
        r"\bsearch(?:\s+up)?\b",
        r"\blook\s+up\b",
        r"\bgoogle\b",
        r"\bfind out how\b",
        r"\bfigure out (?:a way|how)\b",
        r"\bcould you (?:first )?search\b",
        r"\bweather\s+(?:trends|data|history|patterns|research)\b",
    ]
    wants_search = any(re.search(p, t) for p in search_patterns)
    if not wants_search and re.search(r"\bhow (?:to|do i|can i)\b", t):
        wants_search = True
    if not wants_search and re.search(r"\bfigure out\b", t) and re.search(r"\b(?:instagram|insta|discord|tiktok|snapchat)\b", t):
        wants_search = True

    # Only call get_weather_detailed for current local weather (not research)
    if is_weather_only and not wants_search:
        results.append(f"get_weather_detailed: {_execute_tool('get_weather_detailed', {})}")

    # IP address / system info queries — call get_system_info to prevent hallucination
    ip_patterns = [
        r"\bwhat'?s\s+my\s+ip\b",
        r"\bwhat\s+is\s+my\s+ip\b",
        r"\bmy\s+ip\s+(?:address|)\b",
        r"\bip\s+address\b",
    ]
    if any(re.search(p, t) for p in ip_patterns):
        results.append(f"get_system_info: {_execute_tool('get_system_info', {})}")

    if wants_search:
        query = _build_search_query(user_message)
        if query:
            results.append(f"web_search: {_execute_tool('web_search', {'query': query})}")

    # Fine-grained intent prefetch (two-stage router) — safe no-arg data tools only
    fine, _fine_conf = get_last_fine_intent()
    if fine and not results:
        entry = _FINE_PREFETCH.get(fine)
        if entry:
            tool_name, args = entry
            try:
                results.append(f"{tool_name}: {_execute_tool(tool_name, args)}")
            except Exception:
                pass

    return results


def _format_short_circuit(prefetched: list[str]) -> str | None:
    """T1: turn a locally prefetched tool result into a natural reply with
    zero LLM calls. Returns None when the data isn't template-able."""
    for r in prefetched:
        if r.startswith("get_weather_detailed: "):
            data = r[len("get_weather_detailed: ") :]
            try:
                temp = re.search(r"Temperature: ([^,]+)", data)
                cond = re.search(r"°F,\s*([^.]*)\.", data)
                hum = re.search(r"Humidity: (\d+)%", data)
                wind = re.search(r"Wind: ([\d.]+) mph", data)
                rain = re.search(r"Rain chance next 3 hours: (\d+)%", data)
                parts = []
                if temp:
                    parts.append(f"Currently {temp.group(1).strip()}")
                if cond:
                    parts.append(cond.group(1).strip())
                if hum:
                    parts.append(f"humidity {hum.group(1)}%")
                if wind:
                    parts.append(f"wind at {wind.group(1)} mph")
                if rain:
                    parts.append(f"{rain.group(1)}% chance of rain in the next 3 hours")
                if parts:
                    return ", ".join(parts) + "."
            except Exception:
                pass
            return data
    return None


def _sanitize_assistant_text(text: str) -> str:
    """Strip malformed structured blobs and reasoning artifacts some models return as plain text."""
    if not text:
        return ""
    stripped = text.strip()

    # Strip DeepSeek/NIM <think> reasoning blocks
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL).strip()

    # Strip residual XML tool call markup
    stripped = strip_xml_tool_calls(stripped)

    # Strip ```json ... ``` markdown fence tool calls (Nemotron regression)
    stripped = strip_markdown_json_tool_calls(stripped)

    # Strip bare JSON tool calls like {"tool_calls": [...]}
    stripped = strip_json_tool_calls(stripped)

    # Strip <tool_call>function(name)</tool_call> tags
    stripped = strip_tool_call_tags(stripped)

    # Models returning JSON tool call as plain object: {"tool": "...", "arguments": {...}}
    if stripped.startswith("{") and ('"tool"' in stripped or '"name"' in stripped):
        # Only strip if it actually parses as a tool call JSON
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and (parsed.get("tool") or parsed.get("name")):
                return ""
        except Exception:
            pass  # not valid JSON, keep the text

    # Nemotron dot-notation tool call: .tool_call function_name(args)
    if stripped.startswith(".tool_call"):
        return ""

    # NIM returning JSON tool call arrays as text: [{"type": "tool", "name": "...", ...}]
    if stripped.startswith("[{") and '"name"' in stripped:
        return ""

    # NIM returning Python-repr style content blobs: [{'type': 'text', 'text': '...'}]
    if stripped.startswith("[{") and "'type'" in stripped and "'text'" in stripped:
        match = re.search(r"'text':\s*'((?:\\'|[^'])*)'", stripped)
        if match:
            return match.group(1).replace("\\'", "'")
        return "I looked into that but need a clearer question — try asking me to search for something specific."

    return stripped


def _nemotron_extra_body(reasoning_budget: int | None) -> dict:
    """Build the NIM extra_body for a reasoning budget (None -> legacy 2048)."""
    budget = reasoning_budget if reasoning_budget is not None else 2048
    body: dict = {"chat_template_kwargs": {"enable_thinking": budget > 0}}
    if budget > 0:
        body["reasoning_budget"] = budget
    return body


def ask_nemotron_ultra(
    user_message: str,
    tool_results: list[str],
    timeout: float = 90,
    reasoning_budget: int | None = None,
    models: list[str] | None = None,
) -> str:
    """Nemotron Ultra with in-slot understudy (NIM_MODEL_FRONTIER).

    Tries each model in order under the same bounded timeout; only when the
    whole slot is exhausted is the last exception re-raised, so the provider
    chain records a real failure instead of a silent half-answer.
    """
    models_to_try = models or NIM_MODEL_FRONTIER
    last_exc: Exception | None = None
    for model in models_to_try:
        try:
            return _ask_nemotron_ultra_model(
                user_message,
                tool_results,
                model=model,
                timeout=timeout,
                reasoning_budget=reasoning_budget,
            )
        except Exception as e:
            _debug(f"Nemotron model {model} failed: {e}")
            last_exc = e
    raise last_exc if last_exc else Exception("NIM frontier slot exhausted")


def _ask_nemotron_ultra_model(
    user_message: str,
    tool_results: list[str],
    model: str = NEMOTRON_ULTRA_MODEL,
    timeout: float = 90,
    reasoning_budget: int | None = None,
) -> str:
    from openai import OpenAI

    api_key = NVIDIA_NEMOTRON_API_KEY
    base_url = "https://integrate.api.nvidia.com/v1"
    mock_b = _mock_base_url()
    if mock_b:
        base_url = mock_b
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_nim_messages(user_message, tool_results)

    def _append_tool_msg(fn_name, fn_args, result_str):
        tc_id = f"call_{len([m for m in messages if m.get('role') == 'tool'])}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": fn_name, "arguments": json.dumps(fn_args)}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_str})

    _nemotron_loop_count = 0
    slot_deadline = time.monotonic() + PROVIDER_SLOT_TIME_BUDGET
    for _ in range(4):
        _nemotron_loop_count += 1
        if time.monotonic() > slot_deadline:
            raise Exception(
                "Nemotron Ultra tool execution did not resolve (time budget exceeded)"
            )
        _debug(f"[Nemotron Ultra] Loop iter {_nemotron_loop_count}/4")
        # Phase 0.5: Force synthesis on last iteration to prevent tool-call-only loop
        is_last_iter = _nemotron_loop_count >= 4
        tc = "none" if is_last_iter else "auto"
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            top_p=0.95,
            max_tokens=4096,
            extra_body=_nemotron_extra_body(reasoning_budget) if "nemotron" in model else None,
            tools=_build_openai_tools() if not is_last_iter else [],
            tool_choice=tc,
            timeout=timeout,
        )

        if not response.choices:
            raise Exception("Nemotron Ultra returned no choices")

        message = response.choices[0].message
        # Try modern tool_calls first, fall back to legacy function_call
        tc_list = getattr(message, "tool_calls", None) or []
        fc = getattr(message, "function_call", None)
        content = (getattr(message, "content", None) or "").strip()

        # Structured tool_calls from modern format
        if tc_list:
            for tc in tc_list:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
                raw_args = getattr(fn, "arguments", "{}") if fn else "{}"
                try:
                    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
                except Exception:
                    fn_args = {}
                if not name:
                    continue
                _debug(f"[Nemotron Ultra] Tool: {name}({fn_args})")
                result = _execute_tool(name, fn_args)
                if has_pending_safe():
                    return result
                _debug(f"[Nemotron Ultra] Result: {str(result)[:100]}")
                _append_tool_msg(name, fn_args, str(result))
            continue

        # Legacy function_call fallback
        if fc:
            name = getattr(fc, "name", None)
            raw_args = getattr(fc, "arguments", "{}")
            try:
                fn_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
            except Exception:
                fn_args = {}
            _debug(f"[Nemotron Ultra] Legacy Tool: {name}({fn_args})")
            result = _execute_tool(name, fn_args)
            if has_pending_safe():
                return result
            _debug(f"[Nemotron Ultra] Result: {str(result)[:100]}")
            _append_tool_msg(name, fn_args, str(result))
            continue

        # Multiple raw JSON tool calls, one per line
        if content.startswith("{") and "\n{" in content:
            try:
                lines = [l.strip() for l in content.strip().splitlines() if l.strip().startswith("{")]
                executed = False
                for line in lines:
                    parsed = json.loads(line)
                    fn_name = parsed.get("name") or parsed.get("tool", "")
                    fn_args = parsed.get("arguments") or parsed.get("args") or {}
                    if not isinstance(fn_args, dict):
                        fn_args = {}
                    if fn_name:
                        _debug(f"[Nemotron Ultra] Multi-line tool: {fn_name}({fn_args})")
                        result = _execute_tool(fn_name, fn_args)
                        if has_pending_safe():
                            return result
                        _debug(f"[Nemotron Ultra] Result: {str(result)[:100]}")
                        _append_tool_msg(fn_name, fn_args, str(result))
                        executed = True
                if executed:
                    continue
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Single raw JSON tool call: {"tool": "name", ...} or {"name": "name", ...}
        if content.startswith("{") and ('"tool"' in content or '"name"' in content):
            try:
                parsed = json.loads(content)
                fn_name = parsed.get("name") or parsed.get("tool", "")
                fn_args = parsed.get("arguments") or parsed.get("args") or {}
                if not isinstance(fn_args, dict):
                    fn_args = {}
                if fn_name:
                    _debug(f"[Nemotron Ultra] Raw JSON tool: {fn_name}({fn_args})")
                    result = _execute_tool(fn_name, fn_args)
                    if has_pending_safe():
                        return result
                    _debug(f"[Nemotron Ultra] Result: {str(result)[:100]}")
                    _append_tool_msg(fn_name, fn_args, str(result))
                    continue
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Naked argument dict — no tool name, infer and auto-execute
        if content.startswith("{") and not ('"tool"' in content or '"name"' in content):
            try:
                parsed = json.loads(content)
                inferred = infer_tool_from_args(parsed, user_message)
                if inferred:
                    fn_name, fn_args = inferred
                    _debug(f"[Nemotron Ultra] Inferred tool from naked args: {fn_name}({fn_args})")
                    validation = validate_tool_args(fn_name, fn_args)
                    if isinstance(validation, str):
                        _debug(f"[Nemotron Ultra] Validation failed: {validation}")
                        continue
                    fn_name, fn_args = validation
                    result = _execute_tool(fn_name, fn_args)
                    if has_pending_safe():
                        return result
                    _debug(f"[Nemotron Ultra] Result: {str(result)[:100]}")
                    _append_tool_msg(fn_name, fn_args, str(result))
                    continue
                _debug(f"[Nemotron Ultra] Unrecognized naked args dict: keys={list(parsed.keys())}, raw={content[:80]}")
            except Exception:
                pass
            _debug("[Nemotron Ultra] Skipping naked args dict, retrying...")
            continue

        # Plain text response — return it
        if not content:
            _debug(f"[Nemotron Ultra] Empty content after tool execution (iter {_nemotron_loop_count})")
            for msg in reversed(messages):
                if msg.get("role") == "tool":
                    return msg.get("content", "Done.")
            raise Exception("Nemotron Ultra returned empty response")

        # Check all known tool call formats via FormatDetector
        tc_calls = detect_and_parse(content, provider="nemotron_ultra")
        if tc_calls:
            for fn_name, fn_args in tc_calls:
                _debug(f"[TOOL PARSED] nemotron: {fn_name}({fn_args})")
                result = _execute_tool(fn_name, fn_args)
                if has_pending_safe():
                    return result
                _debug(f"[TOOL EXECUTED] {fn_name} -> {str(result)[:100]}")
                _append_tool_msg(fn_name, fn_args, str(result))
            continue

        # Detect "narrated but not executed" tool calls — model describes an action
        # instead of emitting the actual tool_call block. Force one more iteration.
        _narration_pattern = r"\b(let me|i'll|i will|going to|need to)\s+(call|use|check|run)\b"
        if content and re.search(_narration_pattern, content, re.IGNORECASE) and _nemotron_loop_count < 4:
            _debug(f"[Nemotron Ultra] Narrated tool call detected, forcing continuation: {content[:80]}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "You said you would call a tool but didn't actually call it. Call it now."})
            continue

        _debug(f"[Nemotron Ultra] Text response ({_nemotron_loop_count} iters): {content[:80]}...")
        return _sanitize_assistant_text(content)

    # ── Diagnostic: dump the last model response to understand why tool_choice="none" was ignored ──
    last_assistant_msg = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    _debug("[Nemotron Ultra] === LOOP EXHAUSTED DIAGNOSTIC ===")
    _debug(f"[Nemotron Ultra] Total iterations: {_nemotron_loop_count}")
    _debug("[Nemotron Ultra] tool_choice was 'none' on last iter, tools=[]")
    if last_assistant_msg:
        _debug(f"[Nemotron Ultra] Last assistant content (raw): {repr(last_assistant_msg.get('content', ''))}")
        tc = last_assistant_msg.get("tool_calls")
        if tc:
            _debug(f"[Nemotron Ultra] Last assistant tool_calls count: {len(tc)}")
            for i, t in enumerate(tc):
                fn = getattr(t, "function", None) or {}
                args = getattr(fn, "arguments", "?")
                _debug(f"[Nemotron Ultra]   tool_call[{i}]: {getattr(fn, 'name', '?')}({args})")
        else:
            _debug("[Nemotron Ultra] No structured tool_calls in last assistant message")
        _debug(f"[Nemotron Ultra] Full message keys: {list(last_assistant_msg.keys())}")
    else:
        _debug("[Nemotron Ultra] No assistant message found in message history")
    tool_result_count = sum(1 for m in messages if m.get("role") == "tool")
    _debug(f"[Nemotron Ultra] {tool_result_count} tool results in messages before exit")
    # Also check if content contained tool-like JSON that our parsers may have missed
    if last_assistant_msg:
        content = last_assistant_msg.get("content", "") or ""
        if '"tool"' in content or '"name"' in content or '"args"' in content:
            _debug("[Nemotron Ultra] WARNING: content has tool-call JSON despite tool_choice='none'")
            _debug(f"[Nemotron Ultra] content (full): {content}")
    _debug("[Nemotron Ultra] === END DIAGNOSTIC ===")
    raise Exception("Nemotron Ultra tool execution did not resolve")


def infer_tool_from_args(parsed: dict, user_message: str = "") -> tuple | None:
    """Infer tool name+args from a naked argument dict. Returns (tool_name, args) or None."""
    if not isinstance(parsed, dict):
        return None

    # LLM‑first inference — ask the model to map the raw args to a known tool
    if JARVIS_LLM_FIRST:
        try:
            tool_names = [d["function"]["name"] for d in _iter_unique_tool_definitions()]
            prompt = (
                "You are mapping raw JSON arguments to a tool call. The model returned a "
                "naked argument dict (no tool name). Choose the best tool from the available "
                "list and return COMPLETE arguments for it.\n\n"
                f"User request: {user_message}\n"
                f"Raw args: {json.dumps(parsed)[:500]}\n\n"
                f"Available tools: {', '.join(tool_names)}\n\n"
                'Respond with ONLY JSON in this exact format: {"tool": "tool_name", "args": {...}}'
            )
            raw = _llm_choose(prompt)
            if raw:
                match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    fn_name = data.get("tool")
                    fn_args = data.get("args") or {}
                    if isinstance(fn_name, str) and isinstance(fn_args, dict):
                        validation = validate_tool_args(fn_name, fn_args)
                        if not isinstance(validation, str):
                            return validation
        except Exception:
            pass

    keys = list(parsed.keys())
    if len(keys) == 1:
        key = keys[0]
        if key == "query":
            return ("web_search", {"query": parsed["query"]})
        elif key == "url":
            return ("browser_navigate", {"url": parsed["url"]})
        elif key == "app_name":
            if any(kw in user_message.lower() for kw in ("quit", "close", "kill")):
                return ("quit_app", {"app_name": parsed["app_name"]})
            return ("open_app", {"app_name": parsed["app_name"]})
    if len(keys) == 2 and "app_name" in keys:
        if any(kw in user_message.lower() for kw in ("quit", "close", "kill")):
            return ("quit_app", {"app_name": parsed["app_name"]})
        return ("open_app", {"app_name": parsed["app_name"]})
    return None


def validate_tool_args(fn_name: str, fn_args: dict) -> tuple | str:
    """Validate tool arguments against required parameters. Returns (name, args) or error string."""
    fn = TOOL_REGISTRY.get(fn_name) or _learned_tools.get(fn_name)
    if not fn:
        return f"ERROR: Tool '{fn_name}' not found"
    try:
        sig = inspect.signature(fn)
    except Exception:
        return (fn_name, fn_args)
    required_params = []
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                required_params.append(name)
    missing = [p for p in required_params if p not in fn_args or fn_args.get(p) in (None, "", [])]
    if missing:
        return f"ERROR: Missing required argument(s) for {fn_name}: {', '.join(missing)}"
    safe_args = _sanitize_tool_args(fn, fn_args)
    return (fn_name, safe_args)


def _sanitize_tool_args(fn, raw_args):
    if not isinstance(raw_args, dict):
        raw_args = {}
    try:
        sig = inspect.signature(fn)
    except Exception:
        return raw_args
    params = sig.parameters
    if not params:
        return {}
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return raw_args
    allowed = {name for name, p in params.items() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    return {k: v for k, v in raw_args.items() if k in allowed}


def _print_file_change(path: str, diff: str, ins: int, dels: int):
    print(f"\n========== [Jarvis Sandbox] Pending change: {path} (+{ins} -{dels}) ==========")
    print(diff)
    print("========== Say YES to apply, or NO to discard ==========\n")


def _is_malformed_tool_name(fn_name: str) -> bool:
    """NIM reasoning models occasionally leak CoT channel markers into tool
    names (observed: 'read_file<|channel|>commentary'). Such calls can never
    resolve — fail fast so the provider slot hands off quickly instead of
    burning another tool round (~20-30s) on a garbage name."""
    return bool(fn_name) and any(ch in fn_name for ch in ("<", ">", "|"))


def _execute_tool(fn_name: str, fn_args: dict) -> str:
    tool_start = time.time()
    if _is_malformed_tool_name(fn_name):
        # Classifies as 'resolution' upstream: health dip, no circuit trip.
        raise Exception(
            f"tool execution did not resolve (malformed tool call '{fn_name}')"
        )
    # Track tool names for eval harness / audit
    _tool_call_names.append(fn_name)
    # Per-turn memoization — same tool+args within one user message
    memo_key = f"{fn_name}:{str(sorted((k, str(v)) for k, v in (fn_args or {}).items()))}"
    cached = _turn_memo_cache.get(memo_key)
    if cached is not None:
        log_decision(
            phase="act",
            decision="tool_memoized",
            decision_source="memo_cache",
            measurable_inputs={"tool": fn_name},
            outcome="success",
            latency_ms=int((time.time() - tool_start) * 1000),
        )
        _debug(f"[Memo] Reusing cached result for {fn_name}")
        return cached

    fn = TOOL_REGISTRY.get(fn_name) or _learned_tools.get(fn_name)
    if not fn:
        log_decision(
            phase="act",
            decision="tool_not_found",
            decision_source="TOOL_REGISTRY_lookup",
            measurable_inputs={"tool": fn_name},
            outcome="failure",
            latency_ms=int((time.time() - tool_start) * 1000),
        )
        learner.learn_capability(f"implement {fn_name}")
        return f"Tool '{fn_name}' not available yet — learning it now."

    validation = validate_tool_args(fn_name, fn_args)
    if isinstance(validation, str):
        log_decision(
            phase="act",
            decision="tool_validation_failed",
            decision_source="validate_tool_args",
            measurable_inputs={"tool": fn_name, "error": validation},
            outcome="failure",
            latency_ms=int((time.time() - tool_start) * 1000),
        )
        return validation
    fn_name, fn_args = validation

    from safety import (
        TOOL_PERMISSIONS,
        NeedsConfirmation,
        PermissionDenied,
        analyze_command,
        check_permission,
        is_session_confirmed,
        log_audit,
    )
    from tts import speak as _speak_direct

    try:
        safe_args = _sanitize_tool_args(fn, fn_args)
        _sb_pending = None
        _sb_gated = False
        # ── SELF-MODIFICATION → typed review (never auto-proceeds) ──
        if (
            action_sandbox.enabled()
            and not action_sandbox.eval_mode()
            and action_sandbox.is_selfmod(fn_name, safe_args)
        ):
            log_decision(
                phase="safety",
                decision="selfmod_detected",
                decision_source="action_sandbox.is_selfmod",
                measurable_inputs={"tool": fn_name, "target": safe_args.get("target", "")},
            )
            _sm = action_sandbox.resolve_selfmod(fn_name, safe_args, fn)
            if _sm["new_code"].strip() != _sm["old_code"].strip():
                _sm_analysis = action_sandbox.analyze_selfmod(_sm["target"], _sm["old_code"], _sm["new_code"])
                _sm_dry = action_sandbox.dry_run_code(_sm["new_code"])
                _sm_diff = file_sandbox.make_diff(_sm["target"], _sm["old_code"], _sm["new_code"])
                _sm_preview = action_sandbox.format_selfmod_preview(_sm["target"], _sm_analysis, _sm_dry, _sm_diff)
                print("\n" + _sm_preview)
                if _sm_analysis["blocked"] or not _sm_dry["ok"]:
                    _sm_blocked = _sm_analysis["blocked"] or [_sm_dry.get("error", "dry run failed")]
                    log_audit(fn_name, safe_args, "DANGEROUS", "BLOCKED", "; ".join(_sm_blocked))
                    log_decision(
                        phase="safety",
                        decision="selfmod_blocked",
                        decision_source="action_sandbox.analyze_selfmod",
                        measurable_inputs={"tool": fn_name, "blocked_reasons": _sm_blocked},
                        outcome="blocked",
                        latency_ms=int((time.time() - tool_start) * 1000),
                    )
                    return "Self-modification refused by sandbox review: " + "; ".join(_sm_blocked)
                action_sandbox.stage_typed(
                    fn_name, fn, safe_args, action_sandbox.confirm_text_for(_sm["target"]), _sm_preview, _sm
                )
                log_decision(
                    phase="safety",
                    decision="selfmod_staged_for_confirmation",
                    decision_source="action_sandbox.stage_typed",
                    measurable_inputs={"tool": fn_name, "target": _sm["target"]},
                    outcome="pending",
                    latency_ms=int((time.time() - tool_start) * 1000),
                )
                raise NeedsConfirmation(
                    fn_name,
                    safe_args,
                    "DANGEROUS",
                    f"Self-modification of {_sm['target']} is ready. "
                    f"Type: {action_sandbox.confirm_text_for(_sm['target'])} — or say no to cancel.",
                )
            return f"No change needed — {_sm['target']} already matches the requested state."
        # ── FILE WRITES → unified diff preview ──
        if fn_name in file_sandbox.WRITE_TOOLS and file_sandbox.enabled() and not file_sandbox.eval_mode():
            log_decision(
                phase="safety",
                decision="file_write_detected",
                decision_source="file_sandbox.WRITE_TOOLS",
                measurable_inputs={"tool": fn_name, "path": safe_args.get("path", "")},
            )
            _sb_sim = file_sandbox.simulate(fn_name, safe_args)
            if _sb_sim:
                _sb_path, _sb_old, _sb_new = _sb_sim
                _sb_diff = file_sandbox.make_diff(_sb_path, _sb_old, _sb_new)
                if _sb_diff:
                    log_decision(
                        phase="safety",
                        decision="file_write_staged",
                        decision_source="file_sandbox.stage",
                        measurable_inputs={"tool": fn_name, "path": _sb_path, "diff_lines": len(_sb_diff.splitlines())},
                        outcome="pending",
                        latency_ms=int((time.time() - tool_start) * 1000),
                    )
                    file_sandbox.stage(fn_name, safe_args, _sb_path, _sb_old, _sb_new, _sb_diff)
                    _sb_ins, _sb_dels = file_sandbox.counts(_sb_diff)
                    _print_file_change(_sb_path, _sb_diff, _sb_ins, _sb_dels)
                    _sb_pending = (_sb_path, _sb_ins, _sb_dels)
        if action_sandbox.enabled() and not action_sandbox.eval_mode():
            # ── TERMINAL / PYTHON → real sandboxed run preview ──
            if fn_name in action_sandbox.EXEC_TOOLS:
                log_decision(
                    phase="safety",
                    decision="exec_tool_detected",
                    decision_source="action_sandbox.EXEC_TOOLS",
                    measurable_inputs={"tool": fn_name, "command": safe_args.get("command", "")[:200]},
                )
                if fn_name in ("run_terminal_command", "run_command_sandboxed"):
                    _cmd_ok, _cmd_level, _cmd_reason = analyze_command(safe_args.get("command", ""))
                    if not _cmd_ok:
                        log_audit(fn_name, safe_args, "CRITICAL", "BLOCKED", _cmd_reason)
                        log_decision(
                            phase="safety",
                            decision="command_blocked",
                            decision_source="analyze_command",
                            measurable_inputs={"tool": fn_name, "reason": _cmd_reason},
                            outcome="blocked",
                            latency_ms=int((time.time() - tool_start) * 1000),
                        )
                        raise PermissionDenied(f"Command blocked: {_cmd_reason}")
                log_decision(
                    phase="safety",
                    decision="exec_preview_generated",
                    decision_source="action_sandbox.run_exec_preview",
                    measurable_inputs={"tool": fn_name},
                )
                _sb_exec = action_sandbox.run_exec_preview(fn_name, safe_args)
                print(
                    f"\n========== [Jarvis Sandbox] Preview (sandboxed run) ==========\n"
                    f"{_sb_exec}\n============================================================"
                )
                _sb_gated = True
                if is_session_confirmed(fn_name):
                    if action_sandbox.auto_proceed(3):
                        pass
                    else:
                        raise NeedsConfirmation(
                            fn_name, safe_args, "WARNING", "Cancelled — say yes if you change your mind."
                        )
                else:
                    raise NeedsConfirmation(
                        fn_name, safe_args, "WARNING", action_sandbox.confirm_message(fn_name, safe_args)
                    )
            elif fn_name in action_sandbox.INTENT_PREVIEW_TOOLS:
                log_decision(
                    phase="safety",
                    decision="intent_preview_generated",
                    decision_source="action_sandbox.format_intent_preview",
                    measurable_inputs={"tool": fn_name},
                )
                _sb_intent = action_sandbox.format_intent_preview(fn_name, safe_args)
                print(
                    f"\n========== [Jarvis Sandbox] Intent preview ==========\n"
                    f"{_sb_intent}\n============================================================"
                )
                _sb_gated = True
                if is_session_confirmed(fn_name):
                    if action_sandbox.auto_proceed(3):
                        pass
                    else:
                        raise NeedsConfirmation(
                            fn_name, safe_args, "WARNING", "Cancelled — say yes if you change your mind."
                        )
                else:
                    raise NeedsConfirmation(
                        fn_name, safe_args, "WARNING", action_sandbox.confirm_message(fn_name, safe_args)
                    )
        if _sb_pending:
            _sb_path, _sb_ins, _sb_dels = _sb_pending
            _sb_message = (
                f"Write to {_sb_path} (+{_sb_ins}/-{_sb_dels} lines) awaits your approval "
                "— say yes to apply, no to cancel."
            )
            log_decision(
                phase="safety",
                decision="confirmation_required",
                decision_source="file_sandbox_or_action_sandbox",
                measurable_inputs={"tool": fn_name, "path": _sb_path, "ins": _sb_ins, "dels": _sb_dels},
                outcome="pending",
                latency_ms=int((time.time() - tool_start) * 1000),
            )
            raise NeedsConfirmation(fn_name, safe_args, "WARNING", _sb_message)
        if not _sb_gated:
            log_decision(
                phase="safety",
                decision="permission_check",
                decision_source="check_permission",
                measurable_inputs={"tool": fn_name, "level": TOOL_PERMISSIONS.get(fn_name, "WARNING")},
            )
            check_permission(fn_name, safe_args)
        log_decision(
            phase="act",
            decision="tool_execute_start",
            decision_source="direct_execution",
            measurable_inputs={"tool": fn_name, "args_keys": list(safe_args.keys())},
            latency_ms=int((time.time() - tool_start) * 1000),
        )
        _debug(f"[TOOL EXECUTE] {fn_name}({safe_args})")
        result = fn(**safe_args)
        log_audit(fn_name, safe_args, TOOL_PERMISSIONS.get(fn_name, "WARNING"), "EXECUTED")
        result_str = str(result)
        try:
            from event_bus import publish

            _args_preview = ", ".join(f"{k}={str(v)[:80]}" for k, v in (safe_args or {}).items())
            publish("tool_call", {"tool": fn_name, "args": _args_preview[:200]})
        except Exception:
            pass
        log_decision(
            phase="act",
            decision="tool_execute_complete",
            decision_source="direct_execution",
            measurable_inputs={"tool": fn_name, "result_len": len(result_str)},
            outcome="success",
            latency_ms=int((time.time() - tool_start) * 1000),
        )
        _debug(f"[TOOL RESULT] {fn_name}: {result_str[:200]}")
        _turn_memo_cache[memo_key] = result_str
        return result_str
    except NeedsConfirmation as e:
        log_decision(
            phase="safety",
            decision="confirmation_needed",
            decision_source="NeedsConfirmation",
            measurable_inputs={"tool": fn_name, "level": e.level, "message": e.message[:200]},
            outcome="pending",
            latency_ms=int((time.time() - tool_start) * 1000),
        )
        set_pending_safe(fn_name, fn, safe_args, e.level)
        return e.message
    except PermissionDenied as e:
        log_decision(
            phase="safety",
            decision="permission_denied",
            decision_source="PermissionDenied",
            measurable_inputs={"tool": fn_name, "error": str(e)},
            outcome="blocked",
            latency_ms=int((time.time() - tool_start) * 1000),
        )
        _speak_direct("Blocked for safety.")
        return str(e)
    except Exception as e:
        log_decision(
            phase="act",
            decision="tool_error",
            decision_source="unhandled_exception",
            measurable_inputs={"tool": fn_name, "error": str(e)},
            outcome="failure",
            latency_ms=int((time.time() - tool_start) * 1000),
            error=str(e),
        )
        return f"Tool error: {e}"


def _execute_tool_by_name(step: str) -> str:
    parts = step.split()
    if not parts:
        return ""
    fn_name = parts[0]
    args = {}
    if len(parts) > 1:
        remainder = " ".join(parts[1:])
        fn = TOOL_REGISTRY.get(fn_name) or _learned_tools.get(fn_name)
        if fn:
            try:
                sig = inspect.signature(fn)
                required = [name for name, p in sig.parameters.items() if p.default is inspect._empty and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
                if len(required) == 1:
                    args[required[0]] = remainder
            except Exception:
                pass
    return _execute_tool(fn_name, args)


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def estimate_conversation_tokens(messages: list[dict]) -> int:
    """Estimate total token usage across all conversation messages."""
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


def get_model_context_limit(model_id: str) -> int:
    """Get the context window size for a given model ID."""
    return MODEL_CONTEXT_LIMITS.get(model_id, 8192)


def get_working_memory(current_message: str) -> list:
    max_tokens = int(get_model_context_limit(_last_model_used) * 0.3)
    output = []
    used = 0

    older = conversation[:-10]
    middle = conversation[-10:-3]
    recent = conversation[-3:]

    if older:
        try:
            from vector_memory import search_vector_memory

            matches = search_vector_memory(current_message, n_results=3, category="conversation")
            if matches:
                context = "Earlier context: " + " | ".join(content for content, _, _, _ in matches)
                tokens = estimate_tokens(context)
                if used + tokens <= max_tokens:
                    output.append({"role": "system", "content": context})
                    used += tokens
        except Exception:
            pass

    if middle:
        middle_text = " | ".join(f"{m['role']}: {str(m.get('content', ''))[:200]}" for m in middle)
        summary_text = f"Recent earlier conversation summary: {middle_text}"
        tokens = estimate_tokens(summary_text)
        if used + tokens <= max_tokens:
            output.append({"role": "system", "content": summary_text[:1200]})
            used += tokens

    for msg in recent:
        tokens = estimate_tokens(msg.get("content", ""))
        if used + tokens <= max_tokens:
            output.append(msg)
            used += tokens

    return output


def _build_gemini_tools():
    from google.genai import types

    declarations = []
    for tool_def in _iter_unique_tool_definitions():
        fn = tool_def["function"]
        params = fn.get("parameters", {})
        properties = {}
        for prop_name, prop_info in params.get("properties", {}).items():
            prop_type = prop_info.get("type", "string").upper()
            type_map = {
                "STRING": "STRING",
                "INTEGER": "INTEGER",
                "NUMBER": "NUMBER",
                "BOOLEAN": "BOOLEAN",
            }
            properties[prop_name] = types.Schema(
                type=type_map.get(prop_type, "STRING"),
                description=prop_info.get("description", ""),
            )

        declarations.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties=properties,
                    required=params.get("required", []),
                ),
            )
        )

    return [types.Tool(function_declarations=declarations)]


def _ask_gemini_with_model(user_message: str, model_name: str) -> str:
    from google.genai import types

    client = _get_client()
    if not client:
        raise Exception("Gemini client not available")

    tools = _build_gemini_tools()
    contents = []
    for msg in get_working_memory(user_message):
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(user_message),
        tools=tools,
        temperature=0.7,
    )

    tool_results = []
    for _ in range(GEMINI_MAX_TOOL_ROUNDS):
        # Track daily Gemini usage and enforce limit (hard stop at limit - 2).
        usage = _increment_gemini_usage()
        if usage >= GEMINI_DAILY_LIMIT - 1:
            tomorrow = datetime.date.today() + datetime.timedelta(days=1)
            midnight_ts = datetime.datetime.combine(tomorrow, datetime.time.min).timestamp()
            global _gemini_backoff_until
            _gemini_backoff_until = midnight_ts
            try:
                _notify_gemini_disabled("daily_limit")
            except Exception:
                pass
            raise Exception("Gemini daily request limit exceeded")
        response = client.models.generate_content(model=model_name, contents=contents, config=config)
        candidate = response.candidates[0]

        tool_calls = []
        text_parts = []
        for part in candidate.content.parts or []:
            if hasattr(part, "function_call") and part.function_call:
                tool_calls.append(part.function_call)
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if not tool_calls:
            text = " ".join(text_parts).strip()
            # Check all known tool call formats via FormatDetector
            tc_calls = detect_and_parse(text, provider="gemini")
            if tc_calls:
                tool_result_parts = []
                for fn_name, fn_args in tc_calls:
                    _debug(f"[TOOL PARSED] gemini: {fn_name}({fn_args})")
                    result = _execute_tool(fn_name, fn_args)
                    _debug(f"[TOOL EXECUTED] {fn_name} -> {str(result)[:100]}")
                    if has_pending_safe():
                        return result
                    tool_results.append(f"{fn_name}: {result}")
                    tool_result_parts.append(types.Part(function_response=types.FunctionResponse(name=fn_name, response={"result": result})))
                contents.append(types.Content(role="tool", parts=tool_result_parts))
                continue
            return text

        contents.append(candidate.content)
        tool_result_parts = []
        for call in tool_calls:
            fn_name = call.name
            fn_args = dict(call.args) if call.args else {}
            _debug(f"[Gemini] Tool: {fn_name}({fn_args})")
            result = _execute_tool(fn_name, fn_args)
            tool_results.append(f"{fn_name}: {result}")
            _debug(f"[Gemini] Result: {str(result)[:100]}")
            if has_pending_safe():
                return result
            tool_result_parts.append(types.Part(function_response=types.FunctionResponse(name=fn_name, response={"result": result})))

        contents.append(types.Content(role="tool", parts=tool_result_parts))

    return _tool_loop_exhausted(tool_results)


def ask_gemini_tools_only(user_message: str) -> list[str]:
    from google.genai import types

    client = _get_client()
    if not client:
        raise Exception("Gemini client not available")

    contents = []
    for msg in get_working_memory(user_message):
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(user_message),
        tools=_build_gemini_tools(),
        temperature=0.1,
    )

    results = []
    for _ in range(3):
        usage = _increment_gemini_usage()
        if usage > 20:
            tomorrow = datetime.date.today() + datetime.timedelta(days=1)
            midnight_ts = datetime.datetime.combine(tomorrow, datetime.time.min).timestamp()
            global _gemini_backoff_until
            _gemini_backoff_until = midnight_ts
            try:
                _notify_gemini_disabled("daily_limit")
            except Exception:
                pass
            raise Exception("Gemini daily request limit exceeded")
        response = client.models.generate_content(
            model=GEMINI_TOOL_MODEL,
            contents=contents,
            config=config,
        )
        candidate = response.candidates[0]
        tool_calls = [part.function_call for part in (candidate.content.parts or []) if hasattr(part, "function_call") and part.function_call]
        if not tool_calls:
            break

        contents.append(candidate.content)
        tool_result_parts = []
        for call in tool_calls:
            fn_name = call.name
            fn_args = dict(call.args) if call.args else {}
            _debug(f"[Gemini tools] Tool: {fn_name}({fn_args})")
            result = _execute_tool(fn_name, fn_args)
            _debug(f"[Gemini tools] Result: {str(result)[:100]}")
            results.append(f"{fn_name}: {result}")
            tool_result_parts.append(types.Part(function_response=types.FunctionResponse(name=fn_name, response={"result": result})))
        contents.append(types.Content(role="tool", parts=tool_result_parts))

    return results


def ask_gemini(user_message: str) -> str:
    global _last_model_used
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            reply = _ask_gemini_with_model(user_message, model_name)
            _last_model_used = model_name
            return reply
        except Exception as e:
            last_error = e
            _debug(f"Gemini model {model_name} failed: {e}")
            continue
    raise last_error if last_error else Exception("No Gemini model available")


def _retry(fn, max_retries: int = 3):
    waits = [2, 3, 5]
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            transient = "503" in err or "429" in err or "unavailable" in err or "overloaded" in err or "quota" in err
            if transient and attempt < max_retries - 1:
                wait_s = waits[min(attempt, len(waits) - 1)]
                _debug(f"Provider busy, retrying in {wait_s}s...")
                time.sleep(wait_s)
                continue
            raise


def sanitize_messages_for_fallback(messages: list[dict]) -> list[dict]:
    """Strip internal tool error traces from assistant messages for fallback providers."""
    sanitized = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role in ("function", "tool"):
            continue
        if isinstance(content, str):
            cl = content.lower()
            if any(
                marker in cl
                for marker in [
                    "tool error:",
                    "tool execution failed",
                    "stack trace",
                    "traceback",
                    "missing required argument",
                    "function_response",
                    "tool_call",
                    "function_call",
                ]
            ):
                continue
        if role in ("user", "assistant", "system"):
            sanitized.append(msg)
    return sanitized


def _provider_messages(user_message: str, tool_results: list[str], max_context_chars: int = 0):
    system_prompt = build_system_prompt(user_message)
    if tool_results:
        results_block = "\n".join(f"- {r}" for r in tool_results)
        if max_context_chars > 0 and len(results_block) > max_context_chars:
            results_block = results_block[:max_context_chars] + "..."
        system_prompt += "\n\nTool results already gathered:\n" + results_block
        system_prompt += "\nUse these tool results directly. Do not call tools again."
    else:
        system_prompt += "\n\nAnswer the user directly from your knowledge. Do NOT mention calling tools, running commands, or checking anything. Just respond naturally and helpfully."

    messages = [{"role": "system", "content": system_prompt}]
    clean_memory = [m for m in get_working_memory(user_message) if m.get("role") in ("user", "assistant", "system") and m.get("content")]
    clean_memory = sanitize_messages_for_fallback(clean_memory)
    if max_context_chars > 0:
        # Budget the recent history so the total request stays under provider limits.
        budget = max_context_chars
        trimmed = []
        for m in reversed(clean_memory):
            content = m.get("content") or ""
            if len(content) <= budget:
                trimmed.append(m)
                budget -= len(content)
            else:
                m = dict(m)
                m["content"] = content[:budget]
                trimmed.append(m)
                break
        clean_memory = list(reversed(trimmed))
    messages.extend(clean_memory)
    messages.append({"role": "user", "content": user_message})
    return messages


def _build_openai_functions():
    """Legacy functions format — for providers that still use it (NIM)."""
    functions = []
    for tool_def in _iter_unique_tool_definitions():
        fn = tool_def["function"]
        functions.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
            }
        )
    return functions


def _build_openai_tools(max_tools: int = 0):
    """Modern tools format — required by Groq and OpenRouter.
    Pass max_tools > 0 to limit the number of tools (Groq limit is 128)."""
    tools = []
    for tool_def in _iter_unique_tool_definitions():
        if max_tools > 0 and len(tools) >= max_tools:
            break
        fn = tool_def["function"]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
                },
            }
        )
    return tools


def _openai_message_get_tool_call(message):
    """Extract tool call from either legacy function_call or modern tool_calls format."""
    # Modern format: tool_calls list
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    if tool_calls:
        return ("tool_calls", tool_calls)

    # Legacy format: function_call
    fc = getattr(message, "function_call", None)
    if not fc and isinstance(message, dict):
        fc = message.get("function_call")
    if fc:
        return ("function_call", fc)

    return (None, None)


def _openai_message_get_text(message):
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


# Providers that require the modern tools/tool_choice format
_TOOLS_FORMAT_PROVIDERS = {"Groq", "OpenRouter"}


def _mock_base_url() -> str | None:
    if JARVIS_MOCK_PROVIDERS:
        return f"{MOCK_PROVIDER_URL}/v1"
    return None


def _ask_openai_compatible(
    provider_name: str,
    api_key: str,
    base_url: str,
    model: str,
    user_message: str,
    tool_results: list[str],
    use_tools: bool = False,
    functions: list[dict] | None = None,
    max_context_chars: int = 0,
) -> str:
    if not api_key:
        raise Exception(f"{provider_name} API key not configured")

    mock_b = _mock_base_url()
    if mock_b:
        base_url = mock_b
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _provider_messages(user_message, tool_results, max_context_chars=max_context_chars)

    # Decide which calling format to use
    use_modern_format = provider_name in _TOOLS_FORMAT_PROVIDERS
    enable_tools = not tool_results

    # Hard deadline for the whole slot: a stalled model must hand off to the
    # fallback chain instead of eating minutes of API calls (observed 113s on
    # NIM Fast for a single request).
    slot_deadline = time.monotonic() + PROVIDER_SLOT_TIME_BUDGET
    for _ in range(4):  # up from 2 — allows tool call + result + follow-up
        if time.monotonic() > slot_deadline:
            raise Exception(f"{provider_name} tool execution did not resolve (time budget exceeded)")
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 30,
        }

        if use_modern_format and enable_tools:
            tool_limit = 128 if provider_name == "Groq" else 0
            kwargs["tools"] = _build_openai_tools(max_tools=tool_limit)
            kwargs["tool_choice"] = "auto"
        elif not use_modern_format and enable_tools and functions is not None:
            kwargs["functions"] = functions
            kwargs["function_call"] = "auto"

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            if enable_tools and _is_tool_use_failed(e):
                parsed = _parse_failed_tool_use(e)
                if parsed:
                    name, fn_args = parsed
                    _debug(f"[{provider_name}] Recovering malformed tool call: {name}({fn_args})")
                    result = _execute_tool(name, fn_args)
                    tool_results = list(tool_results) + [f"{name}: {result}"]
                    messages = _provider_messages(user_message, tool_results)
                    enable_tools = False
                    continue
                # Try format detection on the failed generation
                failed_gen = getattr(e, "failed_generation", "") or getattr(e, "body", "") or str(e)
                tc_calls = detect_and_parse(failed_gen, provider=provider_name)
                if tc_calls:
                    for fn_name, fn_args in tc_calls:
                        _debug(f"[{provider_name}] Format recovery: {fn_name}({fn_args})")
                        result = _execute_tool(fn_name, fn_args)
                        tool_results = list(tool_results) + [f"{fn_name}: {result}"]
                    messages = _provider_messages(user_message, tool_results)
                    enable_tools = False
                    continue
            err_text = str(e).upper()
            if "429" in err_text or "RATE LIMIT" in err_text or "QUOTA" in err_text:
                _backoff_provider(provider_name, 600)
                if provider_name not in _rate_limited_providers:
                    _rate_limited_providers.add(provider_name)
                    _debug(f"[{provider_name}] Rate limited — backing off 600s")
                raise
            elif "TIMEOUT" in err_text or "READ TIMEOUT" in err_text or "CONNECTION" in err_text:
                _backoff_provider(provider_name, 120)
            _debug_provider_response(provider_name, "request exception", e)
            raise

        _debug_provider_response(provider_name, "raw response", response)
        if not hasattr(response, "choices") or len(response.choices) == 0:
            _backoff_provider(provider_name, 120)
            raise Exception(f"{provider_name} returned no choices")

        choice = response.choices[0]
        message = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
        if message is None:
            _backoff_provider(provider_name, 120)
            raise Exception(f"{provider_name} returned an invalid choice")

        call_type, call_data = _openai_message_get_tool_call(message)
        text = _openai_message_get_text(message) or ""

        # No tool call — check for bare JSON / XML / markdown tool calls, or return text response
        if call_type is None:
            if not text.strip():
                _debug_provider_response(provider_name, "raw message", message)
                _backoff_provider(provider_name, 120)
                raise Exception(f"{provider_name} returned an empty response")
            # Check all known tool call formats via FormatDetector
            tc_calls = detect_and_parse(text, provider=provider_name)
            if tc_calls:
                for fn_name, fn_args in tc_calls:
                    _debug(f"[TOOL PARSED] {provider_name}: {fn_name}({fn_args})")
                    result = _execute_tool(fn_name, fn_args)
                    _debug(f"[TOOL EXECUTED] {fn_name} -> {str(result)[:100]}")
                    if has_pending_safe():
                        return result
                    if use_modern_format:
                        messages.append({"role": "assistant", "content": text, "tool_calls": []})
                        messages.append({"role": "tool", "tool_call_id": "detect_0", "content": str(result)})
                    else:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "function_call": {"name": fn_name, "arguments": json.dumps(fn_args)},
                            }
                        )
                        messages.append({"role": "function", "name": fn_name, "content": str(result)})
                continue
            return _sanitize_assistant_text(text.strip())

        # Modern tool_calls format (Groq, OpenRouter)
        if call_type == "tool_calls":
            tool_calls_list = call_data if isinstance(call_data, list) else list(call_data)
            # Append assistant message with tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": getattr(tc, "id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": getattr(tc.function, "name", ""),
                                "arguments": getattr(tc.function, "arguments", "{}"),
                            },
                        }
                        for i, tc in enumerate(tool_calls_list)
                    ],
                }
            )
            # Execute each tool and append tool result messages
            for tc in tool_calls_list:
                tc_id = getattr(tc, "id", "call_0")
                fn_name = getattr(tc.function, "name", "")
                raw_args = getattr(tc.function, "arguments", "{}")
                try:
                    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    fn_args = {}
                _debug(f"[{provider_name}] Tool: {fn_name}({fn_args})")
                result = _execute_tool(fn_name, fn_args)
                _debug(f"[{provider_name}] Result: {str(result)[:100]}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": str(result),
                    }
                )
            continue  # loop to get final text response

        # Legacy function_call format (NIM)
        if call_type == "function_call":
            fc = call_data
            name = fc.get("name") if isinstance(fc, dict) else getattr(fc, "name", None)
            arguments = fc.get("arguments") if isinstance(fc, dict) else getattr(fc, "arguments", None)
            try:
                fn_args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
            except Exception:
                fn_args = {}
            _debug(f"[{provider_name}] Tool: {name}({fn_args})")
            result = _execute_tool(name, fn_args)
            _debug(f"[{provider_name}] Result: {str(result)[:100]}")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": name, "arguments": json.dumps(fn_args)},
                }
            )
            messages.append({"role": "function", "name": name, "content": str(result)})
            continue

    raise Exception(f"{provider_name} tool execution did not resolve")


def _build_nim_messages(user_message: str, tool_results: list[str]) -> list[dict]:
    """
    Build messages for NIM. The nemotron reasoning model ignores the functions
    parameter during its chain-of-thought and falls back on training knowledge,
    so we must make tool availability unmistakably explicit in the prompt itself.
    """
    base = build_system_prompt(user_message)

    # Hard-list every tool by name so the reasoning trace can see them
    tool_names = [d["function"]["name"] for d in _iter_unique_tool_definitions()]
    tool_list_str = "\n".join(f"  - {n}" for n in tool_names)

    nim_rules = f"""
TOOL CALLING RULES — READ BEFORE REASONING:
You have a function-calling interface. The following tools are FULLY AVAILABLE and WILL WORK:
{tool_list_str}

Before you decide you "cannot" do something, check the list above.
Examples:
   - "search web for python 3.13 release date" → call web_search(query="python 3.13 release date")
  - "open safari" or "open discord" → call open_app(app_name=...)
  - "go to a website" → call browser_navigate(url=...)
  - "open discord on safari" → open_app("Safari"), browser_navigate(discord.com), discord_open_and_send(channel_name, message)
  - "open a discord channel" → discord_open_channel — never loop find_on_screen
  - "what's the weather" → call get_weather_detailed()
  - "what's the news" or "war news" → call warwatch_news()
  - "nuclear news" → call warwatch_news(query="nuclear")
  - "add a goal" → call add_goal(title=..., priority=...)

NEVER say you lack permission or capability for actions that have a matching tool above.
ALWAYS call the tool. If it needs confirmation, the safety layer will handle it.
web_search REQUIRES a "query" argument — never call it empty."""

    if tool_results:
        nim_rules += "\n\nTool results already gathered:\n" + "\n".join(f"- {r}" for r in tool_results)
        nim_rules += "\nUse these results to answer. Do not call tools again."
    else:
        nim_rules += "\n\nCall the relevant tool(s) now. Do not explain — just call them."

    system_prompt = base + nim_rules

    messages = [{"role": "system", "content": system_prompt}]
    clean_memory = [m for m in get_working_memory(user_message) if m.get("role") in ("user", "assistant", "system") and m.get("content")]
    messages.extend(clean_memory)
    messages.append({"role": "user", "content": user_message})
    return messages


def ask_nim_with_context(user_message: str, tool_results: list[str], models: list[str] | None = None, provider_name: str = "NVIDIA NIM") -> str:
    if not NVIDIA_NEMOTRON_API_KEY:
        raise Exception("NVIDIA NIM API key not configured")

    from openai import OpenAI

    api_key = NVIDIA_NEMOTRON_API_KEY
    base_url = "https://integrate.api.nvidia.com/v1"
    mock_b = _mock_base_url()
    if mock_b:
        base_url = mock_b
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_nim_messages(user_message, tool_results)

    models_to_try = models or NIM_MODEL_FAST
    last_exc = Exception("No NIM model attempted")
    for model in models_to_try:
        try:
            return _ask_nim_loop(client, messages, model, provider_name)
        except Exception as e:
            _debug(f"NIM model {model} failed: {e}")
            last_exc = e
    raise last_exc


def _ask_nim_loop(client, messages: list, model: str, provider_name: str = "NVIDIA NIM") -> str:
    """Inner loop for a single NIM model — tries up to 4 iterations."""
    for _ in range(4):
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "timeout": 30,
        }
        # Always use modern tools/tool_choice format (NVIDIA deprecated functions/function_call)
        tools = _build_openai_tools()
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            err_text = str(e).upper()
            if "429" in err_text or "RATE LIMIT" in err_text or "QUOTA" in err_text:
                _backoff_provider(provider_name, 600)
            elif "TIMEOUT" in err_text or "CONNECTION" in err_text:
                _backoff_provider(provider_name, 120)
            _debug_provider_response(provider_name, "request exception", e)
            raise

        _debug_provider_response(provider_name, "raw response", response)
        if not hasattr(response, "choices") or not response.choices:
            _backoff_provider(provider_name, 120)
            raise Exception(f"{provider_name} returned no choices")

        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            _backoff_provider(provider_name, 120)
            raise Exception(f"{provider_name} returned invalid choice")

        tc_list = getattr(message, "tool_calls", None) or []
        fc = getattr(message, "function_call", None)
        text = (getattr(message, "content", None) or "").strip()

        if not tc_list and not fc:
            if not text:
                _backoff_provider(provider_name, 120)
                raise Exception(f"{provider_name} returned empty response")

            tc_calls = detect_and_parse(text, provider="nim")
            if tc_calls:
                for fn_name, fn_args in tc_calls:
                    _debug(f"[{provider_name}] Tool: {fn_name}({fn_args})")
                    result = _execute_tool(fn_name, fn_args)
                    _debug(f"[{provider_name}] Result: {str(result)[:100]}")
                    if has_pending_safe():
                        return result
                    tc_id = f"call_{len([m for m in messages if m.get('role') == 'tool'])}"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tc_id,
                                    "type": "function",
                                    "function": {"name": fn_name, "arguments": json.dumps(fn_args)},
                                }
                            ],
                        }
                    )
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(result)})
                continue

            return _sanitize_assistant_text(text)

        # Modern tool_calls format
        if tc_list:
            for i, tc in enumerate(tc_list):
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") if fn else ""
                args = getattr(fn, "arguments", "{}") if fn else "{}"
                try:
                    fn_args = json.loads(args) if isinstance(args, str) else (args or {})
                except Exception:
                    fn_args = {}
                _debug(f"[{provider_name}] Tool: {name}({fn_args})")
                result = _execute_tool(name, fn_args)
                _debug(f"[{provider_name}] Result: {str(result)[:100]}")
                if has_pending_safe():
                    return result
                tc_id = getattr(tc, "id", f"call_{i}")
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(fn_args)},
                            }
                        ],
                    }
                )
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(result)})
            continue

        # Legacy function_call fallback
        name = getattr(fc, "name", None)
        raw_args = getattr(fc, "arguments", "{}")
        try:
            fn_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except Exception:
            fn_args = {}

        _debug(f"[{provider_name}] Legacy Tool: {name}({fn_args})")
        result = _execute_tool(name, fn_args)
        _debug(f"[{provider_name}] Result: {str(result)[:100]}")
        if has_pending_safe():
            return result
        tc_id = f"call_{len([m for m in messages if m.get('role') == 'tool'])}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": name, "arguments": json.dumps(fn_args)}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(result)})

    raise Exception(f"{provider_name} tool execution did not resolve")


def ask_nim_fast(user_message: str, tool_results: list[str]) -> str:
    """NIM Fast slot: super-49b-v1.5 → nano-30b → step-3.7 (verified 0.3-0.4s)."""
    return ask_nim_with_context(user_message, tool_results, models=NIM_MODEL_FAST, provider_name="NIM Fast")


def ask_nim_coding(user_message: str, tool_results: list[str]) -> str:
    """NIM Coding slot: minimax-m3 → gpt-oss-20b → deepseek-v4-flash (verified 1.1-1.8s)."""
    return ask_nim_with_context(user_message, tool_results, models=NIM_MODEL_CODING, provider_name="NIM Coding")


def _cheap_classify(user_message: str) -> dict | None:
    """2A.2 cheap intent classifier (Groq, JSON completion).

    Contract (routing_policy.CLASSIFIER_KEYS): may only return intent fields
    — it can NEVER select a provider. Falls back to the legacy path on any
    error by raising, so callers degrade gracefully.
    """
    from openai import OpenAI
    from routing_policy import parse_classifier_output

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    system = (
        "You classify user requests into exactly one of five intents. Each request gets "
        "exactly ONE intent, plus an optional fine-grained intent and a confidence score.\n\n"
        "INTENT DEFINITIONS:\n"
        "- coding: the user wants CODE WORK - writing, fixing, refactoring, reviewing, "
        "porting, optimizing, debugging, or documenting scripts/functions/projects. "
        "Includes 'write a script that...', 'write a function...', 'add feature X to a CLI "
        "tool', 'design the endpoints/API for...', 'review this function', 'refactor this', "
        "'port X to Y', 'translate this script'. The words script, function, endpoint, API, "
        "code, CLI, class, README, bug are strong coding signals.\n"
        "- tool_use: the user wants Jarvis to DO something on the machine or web: open/quit "
        "apps, search the web, weather, timers, spotify, calendar, files/folders, "
        "screen/computer actions, terminal commands, iMessage, discord. 'Open', 'search "
        "for', 'play', 'set a timer', 'what's my CPU', 'click', 'navigate', 'upload a file "
        "to X' are tool signals.\n"
        "- reasoning: conceptual explanation or analysis of HOW/WHY something works, general "
        "knowledge ('why does the sky look blue', 'explain the GIL', 'compare X and Y', "
        "'pros and cons', 'what is the difference between'). NOT code tasks: 'explain a "
        "context manager' is reasoning, but 'fix my context manager' is coding.\n"
        "- self_mod: the user asks Jarvis to change Jarvis's own code or source files "
        "(brain.py, terminal.py, server.py, tools/, config.py) - 'fix brain.py', 'modify "
        "yourself', 'improve your code'.\n"
        "- chat: everything else - casual conversation, memory ('remember that'), questions "
        "about Jarvis itself, greetings.\n\n"
        "RULES:\n"
        "- A request to WRITE or FIX CODE is always coding, never tool_use, even when the "
        "code does something (watching a directory, uploading files, parsing CSV).\n"
        "- 'Design the endpoints/API/schema for X' is coding (implement_feature or "
        "design_api), never reasoning and never tool_use.\n"
        "- 'Review/Refactor/Optimize/Port' followed by code context is coding.\n"
        "- tool_required: true if fulfilling the request needs a tool (web search, file "
        "ops, app control, running code). 'Write me a function' is false; 'find files in "
        "~/Downloads' is true.\n"
        "- confidence: your confidence in the intent choice, 0.0-1.0.\n\n"
        "EXAMPLES:\n"
        "User: 'Write a Python function that computes the Levenshtein distance'\n"
        " -> {\"intent\": \"coding\", \"fine_intent\": \"write_code\", \"complexity\": 2, "
        "\"tool_required\": false, \"confidence\": 0.98}\n"
        "User: 'Write a script that finds duplicate files by content hash in a given directory'\n"
        " -> {\"intent\": \"coding\", \"fine_intent\": \"write_script\", \"complexity\": 2, "
        "\"tool_required\": false, \"confidence\": 0.95}\n"
        "User: 'Design the endpoints for a file upload service with resumable uploads'\n"
        " -> {\"intent\": \"coding\", \"fine_intent\": \"design_api\", \"complexity\": 3, "
        "\"tool_required\": false, \"confidence\": 0.93}\n"
        "User: 'Add progress bar support to a CLI tool using tqdm'\n"
        " -> {\"intent\": \"coding\", \"fine_intent\": \"implement_feature\", "
        "\"complexity\": 2, \"tool_required\": false, \"confidence\": 0.9}\n"
        "User: 'Review this function for bugs and edge cases: it slices a list without any bounds checks'\n"
        " -> {\"intent\": \"coding\", \"fine_intent\": \"review_code\", \"complexity\": 2, "
        "\"tool_required\": false, \"confidence\": 0.92}\n"
        "User: 'Open safari and search for weather in Tokyo'\n"
        " -> {\"intent\": \"tool_use\", \"fine_intent\": \"browser\", \"complexity\": 1, "
        "\"tool_required\": true, \"confidence\": 0.99}\n"
        "User: 'Set a timer called tea for 10 seconds'\n"
        " -> {\"intent\": \"tool_use\", \"fine_intent\": \"timer\", \"complexity\": 1, "
        "\"tool_required\": true, \"confidence\": 0.99}\n"
        "User: 'Why does Python's GIL affect threading performance'\n"
        " -> {\"intent\": \"reasoning\", \"fine_intent\": \"explain_concept\", "
        "\"complexity\": 2, \"tool_required\": false, \"confidence\": 0.97}\n"
        "User: 'Explain what a context manager is and why with open works'\n"
        " -> {\"intent\": \"reasoning\", \"fine_intent\": \"explain_concept\", "
        "\"complexity\": 1, \"tool_required\": false, \"confidence\": 0.95}\n"
        "User: 'Fix brain.py, the wake word handler crashes'\n"
        " -> {\"intent\": \"self_mod\", \"fine_intent\": \"modify_self\", \"complexity\": 3, "
        "\"tool_required\": false, \"confidence\": 0.97}\n"
        "User: 'hello how are you today'\n"
        " -> {\"intent\": \"chat\", \"fine_intent\": \"conversation\", \"complexity\": 0, "
        "\"tool_required\": false, \"confidence\": 0.99}\n\n"
        "Respond ONLY with a JSON object with keys: intent (one of the five), fine_intent "
        "(optional string), complexity (0-4), tool_required (bool), confidence (0-1). "
        "No other keys or text."
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": (user_message or "")[:1200]},
        ],
        temperature=0,
        max_tokens=140,
        timeout=30,
        response_format={"type": "json_object"},
    )
    content = (resp.choices[0].message.content or "") if resp.choices else ""
    return parse_classifier_output(content)


def ask_groq(user_message: str, tool_results: list[str]) -> str:
    # Groq requires modern tools format — use_tools handled via _TOOLS_FORMAT_PROVIDERS
    return _ask_openai_compatible(
        "Groq",
        GROQ_API_KEY,
        "https://api.groq.com/openai/v1",
        GROQ_MODEL,
        user_message,
        tool_results,
        # llama-3.3-70b on the on_demand tier caps at 12k tokens/request;
        # trim tool results + history so we never 413.
        max_context_chars=int(os.getenv("JARVIS_GROQ_CONTEXT_CHARS", "24000")),
    )


def ask_openrouter(user_message: str, tool_results: list[str]) -> str:
    # OpenRouter requires modern tools format
    return _ask_openai_compatible(
        "OpenRouter",
        OPENROUTER_API_KEY,
        "https://openrouter.ai/api/v1",
        OPENROUTER_MODEL,
        user_message,
        tool_results,
    )


def ask_pollinations(user_message: str, tool_results: list[str]) -> str:
    url = "https://text.pollinations.ai/openai/chat/completions"
    mock_b = _mock_base_url()
    if mock_b:
        url = f"{mock_b}/chat/completions"
    response = requests.post(
        url,
        json={
            "model": POLLINATIONS_MODEL,
            "messages": _provider_messages(user_message, tool_results),
            "temperature": 0.7,
            "max_tokens": 500,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    _debug_provider_response("Pollinations", "raw response", data)
    msg = data["choices"][0].get("message", {})
    text = msg.get("content") or ""
    if not text.strip():
        # Check for tool_calls in response
        tc_list = msg.get("tool_calls")
        if tc_list:
            for tc in tc_list:
                fn = tc.get("function", {})
                text += f"{fn.get('name', '')}({fn.get('arguments', '{}')}) "
            text = text.strip()
        if not text:
            _debug_provider_response("Pollinations", "empty response data", data)
            _backoff_provider("Pollinations", 120)
            raise Exception("Pollinations returned an empty response")
    return text.strip()


def ask_local_offline(user_message: str, tool_results: list[str]) -> str:
    """Last-resort provider: deterministic reply from the on-device intent router.

    Never raises. Uses locally prefetched tool results when available (weather,
    system info, web search) so the user still gets real data offline.
    """
    try:
        from jarvis_local_nn.integration.local_router import predict_intent

        intent, _confidence = predict_intent(user_message)
    except Exception:
        intent = None

    results = [str(r) for r in (tool_results or []) if str(r).strip()]

    if intent == "tool_use" and results:
        body = "\n".join(f"- {r[:400]}" for r in results[:5])
        return (
            "I'm currently offline (all cloud providers unreachable), but I found "
            f"this locally for you:\n\n{body}\n\n"
            "(I can't run follow-up tools or hold a full conversation until the "
            "network comes back.)"
        )

    canned = {
        "tool_use": (
            "I'm currently offline — every cloud provider is unreachable. I "
            "recognize this as a task I'd normally do with a tool, but I can't "
            "reach the services needed. Please try again when the network is back."
        ),
        "chat": (
            "I'm currently offline, so I can only give you this canned reply. "
            "When the network comes back I'll be able to chat normally."
        ),
        "coding": (
            "I'm currently offline, so I can't generate or run code right now. "
            "Please try again once the network is back."
        ),
        "reasoning": (
            "I'm currently offline, so I can't reason about that properly. "
            "Please try again once the network is back."
        ),
        "self_mod": (
            "I'm currently offline. Self-modification requires a working model, "
            "and it's protected behind a typed confirmation anyway — please try "
            "again once the network is back."
        ),
        "automation": (
            "I'm currently offline, so I can't run multi-step automations. "
            "Please try again once the network is back."
        ),
    }
    if intent and intent in canned:
        return canned[intent]
    return (
        "I'm currently offline — every cloud provider is unreachable. "
        "Please try again shortly."
    )


def ask_gemini_tool_first(user_message: str) -> tuple[list[str], str]:
    from google.genai import types

    client = _get_client()
    if not client:
        raise Exception("Gemini client not available")

    tool_names = [d["function"]["name"] for d in _iter_unique_tool_definitions()]
    tool_summary = ", ".join(tool_names)

    discovery_instruction = f"""You are a tool-first AI agent.

ALWAYS start by considering ALL available tools: {tool_summary}

For EVERY message:
1. Scan the full tool list
2. Decide which tools (if any) would give better or real-time information
3. Call relevant tools BEFORE generating a response
4. If no tools are useful, respond directly

Never assume you know current data (weather, time, system state, files).
Always prefer real tool results over assumptions."""

    contents = []
    for msg in get_working_memory(user_message):
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    config = types.GenerateContentConfig(
        system_instruction=discovery_instruction,
        tools=_build_gemini_tools(),
        temperature=0.1,
    )

    results = []
    direct_answer = ""

    for _ in range(GEMINI_MAX_TOOL_ROUNDS):
        usage = _increment_gemini_usage()
        if usage >= GEMINI_DAILY_LIMIT - 2:
            tomorrow = datetime.date.today() + datetime.timedelta(days=1)
            midnight_ts = datetime.datetime.combine(tomorrow, datetime.time.min).timestamp()
            global _gemini_backoff_until
            _gemini_backoff_until = midnight_ts
            try:
                _notify_gemini_disabled("daily_limit")
            except Exception:
                pass
            raise Exception("Gemini daily request limit exceeded")
        response = client.models.generate_content(
            model=GEMINI_TOOL_MODEL,
            contents=contents,
            config=config,
        )
        candidate = response.candidates[0]

        tool_calls = []
        text_parts = []
        for part in candidate.content.parts or []:
            if hasattr(part, "function_call") and part.function_call:
                tool_calls.append(part.function_call)
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if not tool_calls:
            direct_answer = " ".join(text_parts).strip()
            break

        contents.append(candidate.content)
        tool_result_parts = []
        for call in tool_calls:
            fn_name = call.name
            fn_args = dict(call.args) if call.args else {}
            _debug(f"[Tool-first] Tool: {fn_name}({fn_args})")
            result = _execute_tool(fn_name, fn_args)
            _debug(f"[Tool-first] Result: {str(result)[:100]}")
            results.append(f"{fn_name}: {result}")
            if has_pending_safe():
                return results, result
            tool_result_parts.append(types.Part(function_response=types.FunctionResponse(name=fn_name, response={"result": result})))
        contents.append(types.Content(role="tool", parts=tool_result_parts))
    else:
        if results and not direct_answer:
            direct_answer = _tool_loop_exhausted(results)

    return results, direct_answer


def _handle_cad_request(user_message: str) -> str:
    """Handle CAD/Onshape read requests (Phase 1: read-only)."""
    _debug(f"[CAD] Handling request: {user_message}")
    try:
        from tools.onshape_tools import (
            list_documents,
            get_document,
            list_elements,
            get_features,
            get_feature_params,
            get_part_studio_info,
            parse_onshape_url,
            _extract_ids_from_message,
            format_document_list,
            format_feature_list,
            format_feature_params,
            format_part_studio_info,
        )
    except Exception as e:
        _debug(f"[CAD] Import error: {e}")
        return "Onshape tools not available."

    # Try to extract document/workspace/element IDs from message
    did = wid = eid = None
    extracted = _extract_ids_from_message(user_message)
    if extracted:
        did, wid, eid = extracted

    t = user_message.lower()

    # 1. List documents
    if any(k in t for k in ("list", "show", "what", "my documents", "documents")):
        if "document" in t and ("list" in t or "show" in t or "what" in t):
            result = list_documents(limit=20)
            if not result.get("ok"):
                return f"Failed to list documents: {result.get('error')}"
            from tools.onshape_tools import format_document_list
            return f"Your Onshape documents:\n{format_document_list(result['data'])}"

    # List elements in workspace (only needs did + wid)
    if did and wid and not eid:
        if "element" in t or "part studio" in t or "assembly" in t or "drawing" in t:
            if "list" in t or "show" in t or "what" in t:
                result = list_elements(did, wid)
                if not result.get("ok"):
                    return f"Failed to list elements: {result.get('error')}"
                elems = result["data"]
                if not elems:
                    return "No elements found in that workspace."
                lines = ["Elements in workspace:"]
                for e in elems:
                    lines.append(f"  • {e.get('name', 'Unnamed')}  [{e.get('type', 'unknown')}]  id:{e.get('elementId', '')[:8]}...")
                return "\n".join(lines)

    # If we have a document URL or IDs, handle element-specific queries
    if did and wid and eid:
        # List elements in a document/workspace
        if "element" in t or "part studio" in t or "assembly" in t or "drawing" in t:
            if "list" in t or "show" in t or "what" in t:
                result = list_elements(did, wid)
                if not result.get("ok"):
                    return f"Failed to list elements: {result.get('error')}"
                elems = result["data"]
                if not elems:
                    return "No elements found in that workspace."
                lines = ["Elements in workspace:"]
                for e in elems:
                    lines.append(f"  • {e.get('name', 'Unnamed')}  [{e.get('type', 'unknown')}]  id:{e.get('elementId', '')[:8]}...")
                return "\n".join(lines)

        # Get features for a Part Studio
        if "feature" in t or "extrude" in t or "fillet" in t or "hole" in t or "sketch" in t or "plane" in t:
            # Specific feature parameter query takes precedence over list
            # e.g., "what's the diameter of Hole 1" or "depth of Extrude 1"
            param_keywords = ("diameter", "radius", "depth", "dimension", "parameter", "value")
            if any(k in t for k in param_keywords):
                # Try to find feature by name in message
                import re
                feature_name_match = re.search(r'["\']([^"\']+)["\']', user_message)
                if not feature_name_match:
                    feature_name_match = re.search(r'\b(Hole|Fillet|Extrude|Sketch|Plane|Hole|Chamfer)\s+\d+', user_message, re.IGNORECASE)
                feature_name = feature_name_match.group(0) if feature_name_match else None

                if feature_name:
                    feat_result = get_features(did, wid, eid)
                    if feat_result.get("ok"):
                        features = feat_result["data"]
                        for f in features:
                            if feature_name.lower() in f.get("name", "").lower():
                                param_result = get_feature_params(did, wid, eid, f["featureId"])
                                if param_result.get("ok"):
                                    from tools.onshape_tools import format_feature_params
                                    return f"{f['name']} parameters:\n{format_feature_params(param_result['data']['parameters'])}"
                                break
                        else:
                            return f"Feature '{feature_name}' not found in Part Studio."
                else:
                    return "Please specify which feature (e.g., 'Hole 1', 'Fillet 1')."

            # List all features
            if "list" in t or "show" in t or "what" in t or "all" in t:
                result = get_features(did, wid, eid)
                if not result.get("ok"):
                    return f"Failed to get features: {result.get('error')}"
                from tools.onshape_tools import format_feature_list
                return f"Features in Part Studio:\n{format_feature_list(result['data'])}"

        # Part Studio info (parts, bodies)
        if "part" in t and ("list" in t or "show" in t or "what" in t):
            result = get_part_studio_info(did, wid, eid)
            if not result.get("ok"):
                return f"Failed to get Part Studio info: {result.get('error')}"
            from tools.onshape_tools import format_part_studio_info
            return format_part_studio_info(result["data"])

    # If we have document ID but no workspace/element, get document info
    if did and not (wid and eid):
        if "document" in t or "workspace" in t:
            result = get_document(did)
            if not result.get("ok"):
                return f"Failed to get document: {result.get('error')}"
            doc = result["data"]
            workspaces = doc.get("workspaces", [])
            lines = [f"Document: {doc.get('name', 'Unnamed')}"]
            lines.append(f"  Default workspace: {doc.get('defaultWorkspace', {}).get('name', 'N/A')}")
            if workspaces:
                lines.append("Workspaces:")
                for w in workspaces[:5]:
                    lines.append(f"  • {w.get('name')}  (id: {w.get('id')[:8]}...)")
            return "\n".join(lines)

    # Fallback: nothing actionable in the message (no document URL/IDs).
    # Return None so ask_with_tools falls through to the normal provider
    # chain instead of dead-ending in canned Onshape help for messages that
    # merely happen to route as 'cad'.
    if did and wid and eid:
        return (
            f"Found Part Studio (did={did[:8]}, wid={wid[:8]}, eid={eid[:8]}). "
            "Try: 'list features', 'what's the diameter of Hole 1', 'list parts', 'show elements'"
        )
    elif did:
        return (
            f"Found document (did={did[:8]}). "
            "Try: 'list elements', 'show workspaces', or paste a full Part Studio URL."
        )
    else:
        _debug("[CAD] No document URL/IDs in message — falling through to provider chain")
        return None


def ask_with_tools(user_message: str) -> str:
    with _process_lock:
        return _ask_with_tools_impl(user_message)


def _ask_with_tools_impl(user_message: str) -> str:
    global _last_provider_used, _last_model_used, _nemotron_usage_count

    ask_start = time.time()

    # ── Hybrid routing: local MLX first (simple chat, works offline) ──
    intent_start = time.time()
    intent = classify_intent(user_message)
    log_decision(
        phase="understand",
        decision="intent_classified",
        decision_source="classify_intent",
        measurable_inputs={"intent": intent},
        latency_ms=int((time.time() - intent_start) * 1000),
    )
    _debug(f"[Router] Intent: {intent}")
    local_reply = _try_local_routing(user_message, intent=intent)
    if local_reply is not None:
        log_decision(
            phase="act",
            decision="local_routing_used",
            decision_source="_try_local_routing",
            measurable_inputs={"intent": intent},
            outcome="success",
            latency_ms=int((time.time() - ask_start) * 1000),
        )
        return local_reply

    # ── CAD / Onshape intent (read-only Phase 1) ──
    if intent == "cad":
        cad_reply = _handle_cad_request(user_message)
        if cad_reply is not None:
            log_decision(
                phase="act",
                decision="cad_handler",
                decision_source="intent_equals_cad",
                measurable_inputs={"intent": intent},
            )
            return cad_reply
        # Nothing actionable (no URL/IDs) — fall through to the normal flow.

    if not _internet_available():
        log_decision(
            phase="act",
            decision="no_internet",
            decision_source="_internet_available",
            outcome="failure",
        )
        return "No internet connection. Cloud AI providers require internet access."

    # ── Daily cost budget guardrail (JARVIS_DAILY_BUDGET_USD; 0 = unlimited) ──
    _budget_hit, _spent, _limit = _daily_budget_exceeded()
    if _budget_hit:
        log_decision(
            phase="act",
            decision="budget_exhausted",
            decision_source="_daily_budget_exceeded",
            measurable_inputs={"spent": _spent, "limit": _limit},
            outcome="failure",
        )
        try:
            from local_provider import is_available, ask_local

            if is_available():
                reply = ask_local(user_message)
                if reply:
                    _debug("[Budget] Local reply (budget exhausted)")
                    _last_provider_used = "local_mlx"
                    _last_model_used = "local"
                    return reply
        except Exception:
            pass
        return (
            f"Daily AI budget exhausted (${_spent:.2f} of ${_limit:.2f}). "
            "I've paused cloud AI calls until tomorrow — set JARVIS_DAILY_BUDGET_USD "
            "higher (or 0 to disable) if you want to keep going."
        )

    if intent == "chat":
        try:
            from perf_router import semantic_cache_get

            _debug(f"[Cache] Read intent={intent}, checking semantic cache...")
            cached = semantic_cache_get(user_message, intent)
            if cached is not None:
                log_decision(
                    phase="act",
                    decision="cache_hit",
                    decision_source="semantic_cache_get",
                    measurable_inputs={"intent": intent},
                    outcome="success",
                    latency_ms=int((time.time() - ask_start) * 1000),
                )
                _debug(f"[Cache] HIT for intent={intent}, returning cached response")
                return cached
        except Exception:
            pass
    else:
        log_decision(
            phase="understand",
            decision="cache_bypass",
            decision_source="intent_not_chat",
            measurable_inputs={"intent": intent},
        )
        _debug(f"[Router] Semantic cache bypass for intent={intent}")

    # ── Latency policy: effort tier → provider-agnostic scoring ──
    # Pure routing-layer change (vNext): the E3/2B classifier and its gates
    # are untouched. This only decides WHO answers and HOW HARD they think.
    _effort_budget: int | None = None
    if JARVIS_LATENCY_POLICY and intent != "cad":
        from latency_policy import EFFORT_WEIGHTS, task_profile

        prof = task_profile(intent, user_message, _estimate_task_complexity(user_message))
        log_decision(
            phase="act",
            decision="task_profile",
            decision_source="latency_policy",
            measurable_inputs=dict(prof),
        )
        _debug(f"[Latency] profile: {prof}")
        if prof.get("short_circuit") or intent == "tool_use":
            prefetched_hint = _prefetch_tools_for_message(user_message)
        else:
            prefetched_hint = []
        if prof.get("short_circuit"):
            sc = _format_short_circuit(prefetched_hint)
            if sc is not None:
                log_decision(
                    phase="act",
                    decision="short_circuit_reply",
                    decision_source="latency_policy",
                    outcome="success",
                    measurable_inputs={"tier": "T1", "llm_calls": 0},
                )
                _debug("[Latency] T1 short-circuit reply (0 LLM calls)")
                _last_provider_used = "short_circuit"
                _last_model_used = "prefetched_tool_data"
                return sc
        if prof["effort"] == "low":
            ordered = _effort_candidate_order(intent, EFFORT_WEIGHTS["low"], prof)
            eligible = [e for e in ordered if e.get("status") == "eligible"]
            _debug(f"[Latency] low-effort chain: {[e['provider'] for e in eligible]}")
            for _cand in eligible:
                _key = _cand["provider"]
                _display, _avail, _ask_fn, _model = _POLICY_PRIMARY_CALLS[_key]
                try:
                    if not _avail() or not _provider_available(_display):
                        continue
                except Exception:
                    continue
                try:
                    _debug(f"[Latency] low-effort primary: {_display}")
                    _last_provider_used = _key
                    _last_model_used = _model
                    if _key == "groq":
                        reply = _retry(lambda: ask_groq(user_message, prefetched_hint), max_retries=1)
                    else:
                        reply = _retry(lambda: _ask_fn(user_message), max_retries=1)
                    _record_provider_success(_display)
                    return reply
                except Exception as e:
                    _debug(f"[Latency] {_display} low-effort attempt failed: {e}")
                    _handle_provider_failure(_display, e)
            _debug("[Latency] low-effort chain exhausted — falling through to standard chain")
        _effort_budget = prof.get("thinking_budget", 2048)

    # ── Primary: deterministic (policy or Nemotron-default) tool-first + final ──
    _nemotron_attempted = False
    if JARVIS_LLM_FIRST:
        provider_select_start = time.time()
        chosen = _select_primary_provider(user_message, intent)
        chosen = "nemotron_ultra" if chosen is None else chosen
        log_decision(
            phase="act",
            decision="provider_selected",
            decision_source=(
                "routing_policy" if JARVIS_ROUTER_POLICY else "deterministic_default"
            ),
            measurable_inputs={"provider": chosen, "intent": intent, "policy_on": JARVIS_ROUTER_POLICY},
            latency_ms=int((time.time() - provider_select_start) * 1000),
        )
        if chosen == "nemotron_ultra" and NVIDIA_NEMOTRON_API_KEY and _provider_available("Nemotron Ultra"):
            log_decision(
                phase="act",
                decision="provider_selected",
                decision_source="nemotron_primary",
                measurable_inputs={"provider": "Nemotron Ultra", "model": NEMOTRON_ULTRA_MODEL},
            )
            _nemotron_attempted = True
            try:
                _debug("[Router] Nemotron Ultra primary")
                _last_provider_used = "nemotron_ultra"
                _last_model_used = NEMOTRON_ULTRA_MODEL
                _nemotron_usage_count += 1
                reply = _retry(
                lambda: ask_nemotron_ultra(
                    user_message, [], reasoning_budget=_effort_budget
                ),
                max_retries=2,
            )
                if has_pending_safe():
                    with _pending_lock:
                        pending_tool = _pending_safe.get("tool", "that")
                    return f"I need your permission to {pending_tool.replace('_', ' ')}. Shall I go ahead?"
                _record_provider_success("Nemotron Ultra")
                return reply
            except Exception as e:
                _debug(f"Nemotron Ultra primary failed: {e}")
                _handle_provider_failure("Nemotron Ultra", e)
        elif JARVIS_ROUTER_POLICY and chosen in _POLICY_PRIMARY_CALLS:
            display, available_fn, ask_fn, model = _POLICY_PRIMARY_CALLS[chosen]
            if available_fn() and _provider_available(display):
                log_decision(
                    phase="act",
                    decision="provider_selected",
                    decision_source="routing_policy_primary",
                    measurable_inputs={"provider": display, "model": model},
                )
                try:
                    _debug(f"[Router] Policy primary: {display} (chosen={chosen})")
                    _last_provider_used = chosen
                    _last_model_used = model
                    reply = _retry(ask_fn, max_retries=2)
                    if has_pending_safe():
                        with _pending_lock:
                            pending_tool = _pending_safe.get("tool", "that")
                        return f"I need your permission to {pending_tool.replace('_', ' ')}. Shall I go ahead?"
                    _record_provider_success(display)
                    return reply
                except Exception as e:
                    _debug(f"{display} policy-primary failed: {e}")
                    _handle_provider_failure(display, e)
    # Fallback to original provider logic
    if not _nemotron_attempted and NVIDIA_NEMOTRON_API_KEY and _provider_available("Nemotron Ultra"):
        log_decision(
            phase="act",
            decision="provider_selected",
            decision_source="fallback_nemotron",
            measurable_inputs={"provider": "Nemotron Ultra", "model": NEMOTRON_ULTRA_MODEL},
        )
        try:
            _debug("[Router] Nemotron Ultra primary")
            _last_provider_used = "nemotron_ultra"
            _last_model_used = NEMOTRON_ULTRA_MODEL
            _nemotron_usage_count += 1
            reply = _retry(
                lambda: ask_nemotron_ultra(
                    user_message, [], reasoning_budget=_effort_budget
                ),
                max_retries=2,
            )
            if has_pending_safe():
                with _pending_lock:
                    pending_tool = _pending_safe.get("tool", "that")
                return f"I need your permission to {pending_tool.replace('_', ' ')}. Shall I go ahead?"
            _record_provider_success("Nemotron Ultra")
            return reply
        except Exception as e:
            _debug(f"Nemotron Ultra primary failed: {e}")
            _handle_provider_failure("Nemotron Ultra", e)

    # ── Plugin providers (after Nemotron, before built-in fallback) ──
    log_decision(
        phase="act",
        decision="plugin_providers_attempted",
        decision_source="plugin_manager",
        measurable_inputs={},
    )
    try:
        from plugin_manager import get_plugin_providers

        for pp in get_plugin_providers():
            pname = pp["name"]
            if not _provider_available(pname):
                continue
            try:
                log_decision(
                    phase="act",
                    decision="plugin_provider_attempted",
                    decision_source="plugin_manager",
                    measurable_inputs={"provider": pname},
                )
                _debug(f"[Router] Trying plugin provider: {pname}")
                reply = _retry(lambda: pp["handler"](user_message, []), max_retries=2)
                _last_provider_used = pname.lower().replace(" ", "_")
                _record_provider_success(pname)
                return reply
            except Exception as e:
                log_decision(
                    phase="act",
                    decision="plugin_provider_failed",
                    decision_source="plugin_manager",
                    measurable_inputs={"provider": pname, "error": str(e)},
                    outcome="failure",
                )
                _debug(f"Plugin provider {pname} failed: {e}")
                _handle_provider_failure(pname, e)
    except Exception:
        pass

    # ── Capture tool results from primary provider (Nemotron) for fallback context ──
    log_decision(
        phase="act",
        decision="context_preparation",
        decision_source="_prefetch_tools_for_message",
        measurable_inputs={"accumulated_count": len(_turn_memo_cache)},
    )
    _accumulated_tool_results = []
    _TOOL_RESULT_LIMIT = 2000
    for key, val in _turn_memo_cache.items():
        _accumulated_tool_results.append(f"{key}: {str(val)[:_TOOL_RESULT_LIMIT]}")

    prefetched = _prefetch_tools_for_message(user_message)
    combined_results = _accumulated_tool_results + prefetched
    # Phase 1.1: Truncate to ≤8000 tokens (~32000 chars) for richer fallback context
    _TOKEN_LIMIT = 8000
    _CHAR_LIMIT = _TOKEN_LIMIT * 4
    truncated = []
    char_count = 0
    for r in combined_results:
        r_str = str(r)
        if char_count + len(r_str) <= _CHAR_LIMIT:
            truncated.append(r)
            char_count += len(r_str)
        else:
            allowed = _CHAR_LIMIT - char_count
            if allowed > 80:
                truncated.append(r_str[:allowed] + "...")
            break
    combined_results = truncated
    if combined_results:
        log_decision(
            phase="act",
            decision="fallback_context_prepared",
            decision_source="combined_results",
            measurable_inputs={
                "total_count": len(combined_results),
                "accumulated_count": len(_accumulated_tool_results),
                "prefetched_count": len(prefetched),
                "char_count": char_count,
            },
        )
        _debug(f"Passing {len(combined_results)} tool results to fallbacks ({len(_accumulated_tool_results)} from primary + {len(prefetched)} prefetched, truncated to ~{char_count // 4}t)")
    else:
        _debug("No accumulated tool results from primary provider")

    # Sort providers by health score descending for intelligent fallback
    _provider_health_scores.setdefault("Gemini", 100)
    _provider_health_scores.setdefault("Groq", 100)
    _provider_health_scores.setdefault("NIM Fast", 100)
    _provider_health_scores.setdefault("NIM Coding", 100)
    _provider_health_scores.setdefault("OpenRouter", 100)
    _provider_health_scores.setdefault("Pollinations", 100)

    raw_providers = [
        ("Gemini", _gemini_available() and _provider_available("Gemini"), lambda: ask_gemini(user_message)),
        ("Groq", bool(GROQ_API_KEY), lambda: ask_groq(user_message, combined_results)),
        ("NIM Fast", bool(NVIDIA_NEMOTRON_API_KEY), lambda: ask_nim_fast(user_message, combined_results)),
        ("NIM Coding", bool(NVIDIA_NEMOTRON_API_KEY), lambda: ask_nim_coding(user_message, combined_results)),
        ("OpenRouter", bool(OPENROUTER_API_KEY), lambda: ask_openrouter(user_message, combined_results)),
        ("Pollinations", True, lambda: ask_pollinations(user_message, combined_results)),
    ]

    # Filter available, then sort: policy ordering (2A.3) or health descending
    available = [(n, a, f, _provider_health_scores.get(n, 100)) for n, a, f in raw_providers if a]
    if JARVIS_ROUTER_POLICY:
        try:
            from routing_policy import score_candidates

            policy_cands = [
                {
                    "provider": _PROVIDER_KEY_BY_DISPLAY.get(n, n),
                    "health": h,
                    "latency_ms": _observed_latency_ms(n),
                    "available": True,
                }
                for n, _a, _f, h in available
            ]
            ordered = score_candidates(intent, policy_cands)
            rank = {e["provider"]: i for i, e in enumerate(ordered)}
            available.sort(key=lambda x: rank.get(_PROVIDER_KEY_BY_DISPLAY.get(x[0], x[0]), 999))
            _log_policy_decision(intent, ordered)
        except Exception as e:
            _debug(f"[Router] policy fallback ordering failed, health sort: {e}")
            available.sort(key=lambda x: x[3], reverse=True)
    else:
        available.sort(key=lambda x: x[3], reverse=True)

    log_decision(
        phase="act",
        decision="fallback_chain_prepared",
        decision_source="routing_policy" if JARVIS_ROUTER_POLICY else "health_sorted_providers",
        measurable_inputs={"providers": [(n, h) for n, _, _, h in available]},
    )

    _failed_providers = []
    for provider_name, _is_available, fn, health in available:
        if not _provider_available(provider_name):
            log_decision(
                phase="act",
                decision="provider_skipped",
                decision_source="_provider_available",
                measurable_inputs={"provider": provider_name, "reason": "backoff/disabled"},
                outcome="skipped",
            )
            _debug(f"Skipping {provider_name} (backoff/disabled)")
            continue
        if health < 30:
            log_decision(
                phase="act",
                decision="provider_skipped",
                decision_source="health_threshold",
                measurable_inputs={"provider": provider_name, "health": health},
                outcome="skipped",
            )
            _debug(f"Skipping {provider_name} (low health: {health})")
            continue
        log_decision(
            phase="act",
            decision="fallback_provider_attempted",
            decision_source="health_sorted_fallback",
            measurable_inputs={"provider": provider_name, "health": health},
        )
        _debug(f"Using {provider_name} fallback (health={health})...")
        try:
            reply = _retry(fn, max_retries=2)
            _last_provider_used = provider_name.lower().replace(" ", "_")
            _record_provider_success(provider_name)
            log_decision(
                phase="act",
                decision="fallback_provider_succeeded",
                decision_source=provider_name,
                measurable_inputs={"provider": provider_name},
                outcome="success",
                latency_ms=0,
            )
            return reply
        except Exception as e:
            log_decision(
                phase="act",
                decision="fallback_provider_failed",
                decision_source=provider_name,
                measurable_inputs={"provider": provider_name, "error": str(e)},
                outcome="failure",
            )
            _debug(f"{provider_name} fallback failed: {e}")
            _handle_provider_failure(provider_name, e)
            _failed_providers.append(f"{provider_name}: {e}")

    # ── Last resort: on-device NN provider (never raises, works fully offline) ──
    if not JARVIS_MOCK_PROVIDERS:
        try:
            log_decision(
                phase="act",
                decision="local_nn_attempted",
                decision_source="_local_intent_predict",
                measurable_inputs={},
            )
            _local_offline_intent, _local_offline_conf = _local_intent_predict(user_message)
            reply = ask_local_offline(user_message, combined_results)
            _last_provider_used = "local_nn"
            log_decision(
                phase="act",
                decision="local_nn_succeeded",
                decision_source="ask_local_offline",
                measurable_inputs={"intent": _local_offline_intent, "confidence": _local_offline_conf},
                outcome="success",
            )
            _debug(f"[Local NN] Offline reply (intent={_local_offline_intent}, conf={_local_offline_conf:.2f})")
            return reply
        except Exception as e:
            log_decision(
                phase="act",
                decision="local_nn_failed",
                decision_source="ask_local_offline",
                measurable_inputs={"error": str(e)},
                outcome="failure",
            )
            _debug(f"[Local NN] Offline provider failed: {e}")

    if _failed_providers:
        log_decision(
            phase="act",
            decision="all_providers_failed",
            decision_source="fallback_chain_exhausted",
            measurable_inputs={"failed_providers": _failed_providers, "count": len(_failed_providers)},
            outcome="failure",
        )
        details = "; ".join(_failed_providers)
        return f"All configured AI providers are currently unavailable. Provider errors: {details}. Please try again shortly."
    log_decision(
        phase="act",
        decision="all_providers_unavailable",
        decision_source="no_available_providers",
        measurable_inputs={},
        outcome="failure",
    )
    return "All configured AI providers are currently unavailable. Please try again shortly."


def get_nemotron_usage_count() -> int:
    return _nemotron_usage_count


def get_provider_status_summary() -> str:
    parts = []
    if _nemotron_usage_count:
        parts.append(f"Nemotron: {_nemotron_usage_count}")
    gemini_count = _get_gemini_usage_count()
    if gemini_count:
        parts.append(f"Gemini: {gemini_count}/{GEMINI_DAILY_LIMIT}")
    return " | ".join(parts) if parts else ""


def set_pending_safe(tool, fn, args, level):
    with _pending_lock:
        _pending_safe["tool"] = tool
        _pending_safe["fn"] = fn
        _pending_safe["args"] = args
        _pending_safe["level"] = level
        _pending_safe["expires_at"] = time.time() + SAFETY_PENDING_TTL


def clear_pending_safe():
    with _pending_lock:
        _pending_safe["tool"] = None
        _pending_safe["fn"] = None
        _pending_safe["args"] = None
        _pending_safe["level"] = None
        _pending_safe["expires_at"] = 0


def has_pending_safe():
    with _pending_lock:
        if _pending_safe["fn"] is None:
            return False
        if time.time() > _pending_safe["expires_at"]:
            clear_pending_safe()
            return False
        return True


def execute_pending_safe():
    with _pending_lock:
        if not _pending_safe["fn"]:
            return None
        tool = _pending_safe["tool"]
        fn = _pending_safe["fn"]
        args = _pending_safe["args"]
        level = _pending_safe["level"]
    from safety import WARNING, log_audit, mark_session_confirmed

    if level == WARNING:
        mark_session_confirmed(tool)
    log_audit(tool, args, level, "CONFIRMED_BY_USER")
    clear_pending_safe()
    file_sandbox.clear_pending()
    try:
        real = action_sandbox.real_executor(tool)
        if real:
            return real(
                args.get("command") or args.get("code", ""),
                timeout=int(args.get("timeout", 30) or 30),
            )
        return fn(**args)
    except Exception as e:
        return f"Error: {e}"


def set_pending(fn, description):
    pending_action["fn"] = fn
    pending_action["description"] = description
    pending_action["expires_at"] = time.time() + SAFETY_PENDING_TTL


def clear_pending():
    pending_action["fn"] = None
    pending_action["description"] = None
    pending_action["expires_at"] = 0


def has_pending_action():
    if pending_action["fn"] is None:
        return False
    if time.time() > pending_action["expires_at"]:
        clear_pending()
        return False
    return True


def handle_memory_save(text):
    triggers = ["remember that", "remember my", "remember i", "keep in mind", "don't forget", "note that"]
    for trigger in triggers:
        if trigger in text.lower():
            content = text[text.lower().index(trigger) + len(trigger) :].strip()
            save_memory(content)
            return "Got it, I'll remember that."
    save_memory(text)
    return "Saved to memory."


def handle_memory_list():
    memories = get_all_memories()
    summaries = get_recent_summaries(3)
    if not memories and not summaries:
        return "I don't have any memories saved yet."
    parts = [c for _, c, _ in memories]
    result = "Here's what I remember: " + "; ".join(parts[:5])
    if summaries:
        result += ". Recent topics: " + "; ".join(s for s, _ in summaries[:2])
    return result


def _is_unknown_capability(text: str) -> bool:
    """Determine whether Jarvis lacks a capability for this request.

    NOTE: This used to call Gemini for a YES/NO check, but that consumed precious
    daily Gemini quota for an internal heuristic. We now avoid that extra request.
    """
    return False


def _summarize_with_gemini(history: str) -> str:
    """Summarize conversation history with Gemini if available.

    This helper is only used for background memory summarization and should
    not consume Gemini quota aggressively when the key is precious.
    """
    if not _gemini_available() or datetime.datetime.now().timestamp() < _gemini_backoff_until or _get_gemini_usage_count() >= GEMINI_DAILY_LIMIT - 2:
        return ""
    client = _get_client()
    if not client:
        return ""
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            usage = _increment_gemini_usage()
            if usage >= GEMINI_DAILY_LIMIT - 2:
                return ""
            response = client.models.generate_content(
                model=model_name,
                contents=f"Summarize in 1-2 sentences:\n{history}",
            )
            return (response.text or "").strip()
        except Exception as e:
            last_error = e
            continue
    if last_error:
        _debug(f"Summarize failed on all Gemini models: {last_error}")
    return ""


def _summarize_paste(content: str, max_chars: int = 2000) -> str:
    """Summarize pasted content for context preservation.

    Uses local LLM (Nemotron/NIM) if available, falls back to extractive summary.
    Returns a brief summary that captures the essence of the pasted content.
    """
    if not content or len(content) <= max_chars:
        return content

    # Try using the main provider chain for a quick summary
    try:
        from tts import speak as _speak_status

        # Use agent loop with a simple summarization goal
        summary = run_agent_loop(
            goal=f"Summarize this pasted content in 2-3 sentences, preserving key facts, names, and topics:\n\n{content[:8000]}",
            execute_tool_fn=_execute_tool,
            ask_llm_fn=ask_with_tools,
            speak_fn=_speak_status,
            max_iterations=1,
        )
        if summary and len(summary) > 20:
            return summary.strip()
    except Exception as e:
        _debug(f"Paste summarization via agent failed: {e}")

    # Fallback: extractive summary - first + last paragraphs + key lines
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return content[:max_chars] + "..."

    # Take first 2 paragraphs, last 1 paragraph, and any lines with keywords
    key_indicators = ["chapter", "prologue", "epilogue", "part ", "section ", "act ", "scene "]
    key_lines = []
    for p in paragraphs:
        pl = p.lower()
        if any(kw in pl for kw in key_indicators):
            key_lines.append(p[:200])

    summary_parts = []
    summary_parts.extend(paragraphs[:2])
    if key_lines:
        summary_parts.append("... Key sections: " + "; ".join(key_lines[:3]))
    if len(paragraphs) > 2:
        summary_parts.append(paragraphs[-1])

    summary = "\n\n".join(summary_parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "..."
    return summary


def _sanitize_user_message(text: str) -> str:
    """Strip meta-instructions about API keys or providers."""
    patterns = [
        r",?\s*use the [\w\s\.]+ api key for this[^.]*\.?",
        r",?\s*don't use any other keys[^.]*\.?",
        r",?\s*not any other keys[^.]*\.?",
        r",?\s*using the [\w\s]+ key[^.]*\.?",
        r",?\s*use [\w\s]+ api for this[^.]*\.?",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def process(text, session_id="default"):
    """Route a user turn through the brain, scoped to a persistent session.

    Holds the process lock for the whole turn so concurrent requests
    (terminal scheduler threads, server executor threads) never share
    the in-memory conversation unsafely. Session history is loaded from
    the persistent store under the lock and appended back on completion.
    """
    with _process_lock:
        _load_session_history(session_id)
        return _process_impl(text, session_id)


def _process_impl(text, session_id):
    global _turn_memo_cache, _current_request_id
    _turn_memo_cache = {}
    _tool_call_names.clear()
    _current_request_id = new_request_id()
    _start_time = time.time()
    # Clear intent cache for new turn
    conversation_context.intent_cache.clear()
    text = _sanitize_user_message(text)
    t = text.lower()

    log_decision(
        phase="understand",
        decision="process_start",
        decision_source="process",
        measurable_inputs={"user_message_len": len(text)},
    )

    # Imperative command routing: for "open X" / "launch X" / "start X", help model pick right tool
    imperative_verbs = ("open ", "launch ", "start ", "run ", "quit ", "close ", "kill ")
    if any(t.startswith(v) for v in imperative_verbs):
        # Inject a hint into the message so the model picks the correct app/quit tool
        # This is handled by the system prompt examples, but double-tap helps
        _debug(f"[Router] Imperative command detected: '{t.split()[0]}' — app routing enabled")
        # Let the normal intent router handle it, but ensure it's tool_use

    try:
        from procedural_memory import detect_procedure_trigger

        procedure = detect_procedure_trigger(text)
        if procedure:
            results = []
            for step in procedure.get("steps", []):
                result = _execute_tool_by_name(step)
                results.append(result)
            return ask_nim_with_context(
                f"Completed routine: {procedure['trigger']}",
                results,
            )
    except Exception:
        pass

    try:
        from priority import acknowledge

        acknowledge()
    except Exception:
        pass

    yes_words = [
        r"\byes\b",
        r"\byeah\b",
        r"\byep\b",
        r"\byup\b",
        r"\bdo it\b",
        r"\bgo ahead\b",
        r"\bconfirmed\b",
        r"\bsend it\b",
        r"\bapproved\b",
    ]
    no_words = [r"\bno\b", r"\bnope\b", r"\bcancel\b", r"\bstop\b", r"\bnevermind\b", r"\bdeny\b"]

    def _matches_any(patterns, text):
        return any(re.search(p, text) for p in patterns)

    # Track original user message (not confirmation responses)
    if not has_pending_safe() and not has_pending_action():
        global _last_user_message
        _last_user_message = text

    # Typed confirmation for self-modification (sandbox review)
    if action_sandbox.has_typed_pending():
        _tp = action_sandbox.get_typed_pending()
        if _tp and (t.strip().lower() == _tp.get("confirmation_text", "") or _matches_any(yes_words, t)):
            _tp_result = action_sandbox.apply_staged_selfmod()
            action_sandbox.clear_typed_pending()
            clear_pending_safe()
            file_sandbox.clear_pending()
            _debug(f"[Confirm] Self-mod applied: {_tp_result}")
            reply = ask_with_tools(_last_user_message)
            if conversation and conversation[-1].get("role") == "assistant":
                conversation[-1]["content"] = reply
            return reply
        if _tp and _matches_any(no_words, t):
            from safety import log_audit

            log_audit(_tp.get("tool", "modify_own_tool"), _tp.get("args", {}), "DANGEROUS", "DENIED_BY_USER")
            action_sandbox.clear_typed_pending()
            clear_pending_safe()
            file_sandbox.clear_pending()
            return "Okay, cancelled."
        return (
            f"Waiting for typed confirmation. Type: {_tp.get('confirmation_text', '')} "
            "— or say no to cancel."
        )

    if has_pending_safe():
        if _matches_any(yes_words, t):
            if action_sandbox.is_selfmod(_pending_safe["tool"], _pending_safe["args"]) and not action_sandbox.has_typed_pending():
                clear_pending_safe()
                return "Self-modification needs a typed confirmation — please ask me to make the change again."
            result = execute_pending_safe()
            if result:
                _debug("[Confirm] Confirmed, continuing with original request")
                reply = ask_with_tools(_last_user_message)
                # Replace the "I need your permission" message with the full response
                if conversation and conversation[-1].get("role") == "assistant":
                    conversation[-1]["content"] = reply
                return reply
            return "Something went wrong executing that action."
        elif _matches_any(no_words, t):
            tool = _pending_safe["tool"]
            clear_pending_safe()
            file_sandbox.clear_pending()
            action_sandbox.clear_typed_pending()
            from safety import log_audit

            log_audit(tool or "unknown", {}, "unknown", "DENIED_BY_USER")
            return "Okay, cancelled."

    if has_pending_action():
        if _matches_any(yes_words, t):
            fn = pending_action["fn"]
            clear_pending()
            return fn()
        elif _matches_any(no_words, t):
            clear_pending()
            return "Okay, cancelled."

    memory_triggers = ["remember that", "remember my", "remember i", "keep in mind", "don't forget", "note that"]
    if any(p in t for p in memory_triggers):
        return handle_memory_save(text)

    if any(p in t for p in ["what do you know", "what do you remember", "what have i told you"]):
        return handle_memory_list()

    if any(p in t for p in ["forget", "erase", "delete memory"]):
        for trigger in ["forget", "erase", "delete memory"]:
            if trigger in t:
                keyword = t.split(trigger, 1)[-1].strip()
                # Strip leading articles/possessives so "forget my favorite color" matches "favorite color is blue"
                keyword = re.sub(r"^(my|the|a|an|your|our)\s+", "", keyword, flags=re.IGNORECASE).strip()
                forget_memory(keyword)
                return f"Done, forgotten anything related to '{keyword}'."

    # Self-test routing — "test yourself" / "self test" / bug triage commands
    _self_test_patterns = [
        r"\btest (yourself|your(?: own)? (code|system|software|self))\b",
        r"\bself[- ]?test(?:ing)?\b",
        r"\b(check|look) (yourself|your (?:own )?code|your system) for bugs\b",
        r"\bfind bugs in (your|the) (code|system)\b",
        r"\b(?:run|start) (?:a )?self[- ]?test\b",
        r"\btest (status|progress|report|findings|history|stop|logs)\b",
        r"\b(confirm|dismiss) (bug|finding|issue) [a-f0-9]{12}\b",
    ]
    if any(re.search(p, t) for p in _self_test_patterns):
        try:
            from self_test.agent import handle_command

            return handle_command(text)
        except Exception as _st_err:
            _debug(f"[Self-Test] routing failed: {_st_err}")

    # Ops routing — backup / health check / memory prune (chat-accessible)
    if re.search(r"\bbackup\b", t) and re.search(r"\b(run|take|make|do|create|new)\b", t) \
            or t in ("backup", "backup now"):
        try:
            import backup

            result = backup.run_backup()
            return (
                f"Backup saved to {result['backup_dir']}. "
                f"Copied: {', '.join(result['copied']) or 'nothing'}."
                + (f" Skipped: {', '.join(result['skipped'])}." if result["skipped"] else "")
            )
        except Exception as _bk_err:
            _debug(f"[Ops] backup failed: {_bk_err}")

    if re.search(r"\bhealth\s*check\b", t) or t in ("health status", "how is your health"):
        try:
            from healthcheck import report_text

            return report_text()
        except Exception as _hc_err:
            _debug(f"[Ops] health check failed: {_hc_err}")

    if re.search(r"\bprune (old |stale )?(memor(y|ies))\b", t):
        try:
            from memory import prune_old_memories

            m = re.search(r"(\d+)\s*days", t)
            days = int(m.group(1)) if m else 30
            prune_old_memories(days=days)
            return f"Pruned memories older than {days} days."
        except Exception as _pr_err:
            _debug(f"[Ops] memory prune failed: {_pr_err}")

    # Explicit summarize command
    if re.search(r"\bsummar(iz|is)e\b.*(conversation|chat|history|discussion)", t) or re.search(
        r"(conversation|chat|history).*\bsummar(iz|is)e\b", t
    ):
        summary = _summarize_conversation(recent_turns=20)
        if summary:
            return f"Here's a summary of our recent conversation:\n\n{summary}"
        return "I couldn't generate a summary right now."

    if _is_unknown_capability(text):
        learner.learn_capability(text)
        return "I don't know how to do that yet, but I'm figuring it out. In the meantime, is there anything else I can help with?"

    # Fragment detection — catch "solution", "answer", "explain" with no context
    _fragment_msg = _check_fragment(text, conversation_context)
    if _fragment_msg:
        conversation_context.state = ConversationState.AWAITING_RESPONSE
        _debug(f"[Context] Fragment detected, returning clarification: {_fragment_msg[:80]}")
        return _fragment_msg
    # Clear fragment flag if input has enough context
    if conversation_context.fragment_awaiting_context:
        conversation_context.fragment_awaiting_context = False

    # Self-reflection: log pre-processing state
    if DEBUG:
        _pre_reflect = conversation_context.snapshot()
        _debug(f"[Reflect] PRE  state={_pre_reflect['state']} last_problem={'yes' if _pre_reflect['last_problem'] else 'no'}")

    # Capability/gap analysis — inject inspect_capabilities result directly
    _capability_gap_patterns = [
        r"\bwhat (should|could|can|do) (i |we |you )?(add|build|create|implement|make)\b",
        r"\bwhat'?s (missing|lacking|needed)\b",
        r"\b(what|single) (highest-leverage|most important|biggest|key)\b.*(missing|gap|lack|need)\b",
        r"\bgap\b.*(feature|capability|functionality)\b",
        r"\bwhat can you( not)? do\b",
        r"\bwhat are you (missing|lacking)\b",
        r"\b(how )?can i (improve|enhance|upgrade)\b.*(you|jarvis)\b",
    ]
    if any(re.search(p, t) for p in _capability_gap_patterns):
        _debug("[Router] Capability/gap analysis detected — calling inspect_capabilities")
        try:
            capabilities = _execute_tool("inspect_capabilities", {})
            conversation_context.add_message(
                "system",
                (f"[Capability context from inspect_capabilities()]:\n{capabilities}\n\nUse this data to answer the user's query. Do NOT call inspect_capabilities again."),
            )
            _debug(f"[Router] inspect_capabilities injected ({len(capabilities)} chars)")
        except Exception as e:
            _debug(f"inspect_capabilities injection failed: {e}")

    # Compound request detection — multiple distinct actions in one message
    _compound_verbs = {
        "open",
        "launch",
        "close",
        "quit",
        "send",
        "skip",
        "click",
        "type",
        "write",
        "search",
        "play",
        "pause",
        "create",
        "delete",
    }
    _found_verbs = set()
    for word in t.split():
        if word in _compound_verbs:
            _found_verbs.add(word)
    _has_conjunction = re.search(r"\band\b|\bthen\b|\balso\b", t)
    if len(_found_verbs) >= 2 and _has_conjunction:
        log_decision(
            phase="plan",
            decision="route_to_planner",
            decision_source="compound_request_detector",
            measurable_inputs={"verbs": list(_found_verbs), "has_conjunction": True},
        )
        try:
            from tts import speak as _speak_status

            max_iterations = _estimate_task_complexity(text)
            reply = run_planner_loop(
                goal=text,
                execute_tool_fn=_execute_tool,
                ask_llm_fn=ask_with_tools,
                speak_fn=_speak_status,
            )
            return reply
        except Exception as e:
            _debug(f"Planner loop for compound request failed: {e}, falling through")

    if needs_planner(text):
        log_decision(
            phase="plan",
            decision="route_to_planner",
            decision_source="needs_planner_heuristic",
            measurable_inputs={"text_preview": text[:100]},
        )
        try:
            from tts import speak as _speak_status

            max_iterations = _estimate_task_complexity(text)
            _debug(f"[Complexity] '{text[:50]}' iter={max_iterations} (planner)")
            reply = run_planner_loop(
                goal=text,
                execute_tool_fn=_execute_tool,
                ask_llm_fn=ask_with_tools,
                speak_fn=_speak_status,
            )
        except Exception as e:
            _debug(f"Planner loop failed: {e}, falling back to agent")
            try:
                from tts import speak as _speak_status

                max_iterations = _estimate_task_complexity(text)
                reply = run_agent_loop(
                    goal=text,
                    execute_tool_fn=_execute_tool,
                    ask_llm_fn=ask_with_tools,
                    speak_fn=_speak_status,
                    max_iterations=max_iterations,
                )
            except Exception as e2:
                _debug(f"Agent loop failed: {e2}, falling back")
                reply = ask_with_tools(text)
    elif needs_agent_loop(text):
        log_decision(
            phase="plan",
            decision="route_to_agent",
            decision_source="needs_agent_loop_heuristic",
            measurable_inputs={"text_preview": text[:100]},
        )
        try:
            from tts import speak as _speak_status

            max_iterations = _estimate_task_complexity(text)
            _debug(f"[Complexity] '{text[:50]}' iter={max_iterations} (agent)")
            reply = run_agent_loop(
                goal=text,
                execute_tool_fn=_execute_tool,
                ask_llm_fn=ask_with_tools,
                speak_fn=_speak_status,
                max_iterations=max_iterations,
            )
        except Exception as e:
            _debug(f"Agent loop failed: {e}, falling back")
            reply = ask_with_tools(text)
    else:
        log_decision(
            phase="plan",
            decision="route_to_direct",
            decision_source="no_planner_agent_trigger",
            measurable_inputs={},
        )
        reply = ask_with_tools(text)

    _session_append(session_id, "user", text)
    _session_append(session_id, "assistant", reply)

    # Periodic LLM conversation summarization (every 20 turns) to keep context compact
    try:
        if JARVIS_LLM_FIRST and len(conversation) % 40 == 0:
            summary = _summarize_conversation(recent_turns=10)
            if summary:
                _debug(f"[Summarizer] Inserted summary: {summary[:100]}")
                add_to_vector_memory(f"Conversation summary: {summary}", category="summary")
    except Exception as _sum_err:
        _debug(f"[Summarizer] Failed: {_sum_err}")

    # Update conversation context
    try:
        _ctx_elapsed = time.time() - _start_time
        conversation_context.update(
            text=text,
            reply=reply,
            intent=classify_intent(text),
            provider=_last_provider_used,
            tools=list(_turn_memo_cache.keys()),
            elapsed=_ctx_elapsed,
        )
        if has_pending_safe():
            conversation_context.state = ConversationState.AWAITING_CONFIRM
        if DEBUG:
            _post_reflect = conversation_context.snapshot()
            _debug(f"[Reflect] POST state={_post_reflect['state']} tools={_post_reflect['last_tools']} elapsed={_ctx_elapsed:.2f}s")
    except Exception as _ctx_err:
        _debug(f"[Context] Update error: {_ctx_err}")

    try:
        from associative_memory import record_concepts

        record_concepts(text)
        record_concepts(reply)
    except Exception:
        pass

    try:
        from graph_memory import extract_entities_relations

        if len(conversation) % 3 == 0:
            combined = f"User said: {text}. Jarvis replied: {reply[:500]}"
            extract_entities_relations(combined)
    except Exception:
        pass

    try:
        add_to_vector_memory(f"User: {text} | Jarvis: {reply[:200]}", category="conversation")
    except Exception:
        pass

    try:
        from perf_router import record_latency, semantic_cache_set

        _intent_for_cache = classify_intent(text)
        skip_reason = None
        if _intent_for_cache != "chat":
            skip_reason = f"intent={_intent_for_cache} (not chat)"
        elif _turn_memo_cache:
            skip_reason = f"tools_called={list(_turn_memo_cache.keys())}"
        elif has_pending_safe():
            skip_reason = "pending_safe=true"

        if skip_reason:
            _debug(f"[Cache] Write skipped: {skip_reason}")
        else:
            semantic_cache_set(text, _intent_for_cache, reply)
            _debug(f"[Cache] Write intent={_intent_for_cache}")
        _elapsed_inner = time.time() - _start_time
        record_latency(_last_provider_used, _elapsed_inner)
    except Exception:
        pass

    try:
        _elapsed = time.time() - _start_time
        _intent = classify_intent(text)
        _tool_calls_log = list(_turn_memo_cache.keys())
        _tokens_in = estimate_tokens(text)
        _tokens_out = estimate_tokens(reply)
        _error = None
        log_request(
            request_id=_current_request_id,
            provider=_last_provider_used,
            model=_last_model_used,
            user_message=text,
            reply=reply,
            intent=_intent,
            tokens_input=_tokens_in,
            tokens_output=_tokens_out,
            latency_seconds=_elapsed,
            tool_calls=_tool_calls_log,
            error=_error,
        )
    except Exception:
        pass

    log_decision(
        phase="respond",
        decision="final_response",
        decision_source="process",
        measurable_inputs={
            "provider": _last_provider_used,
            "model": _last_model_used,
            "tools_used": _tool_calls_log,
            "intent": _intent,
            "response_len": len(reply),
            "latency_seconds": round(time.time() - _start_time, 3),
        },
        outcome="success",
        latency_ms=int((time.time() - _start_time) * 1000),
    )
    
    return reply


def get_last_tool_calls() -> list[str]:
    """Return unique tool names called in the last process() invocation."""
    seen: set = set()
    return [t for t in _tool_call_names if not (t in seen or seen.add(t))]



