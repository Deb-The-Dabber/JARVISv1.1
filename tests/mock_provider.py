"""Mock AI Provider Server for testing.

Simulates all providers in the JARVIS provider chain:
  - OpenAI-compatible: Nemotron, DeepSeek, Groq, Kimi, OpenRouter, NIM, Pollinations
  - Google GenAI: Gemini (tool-first + final)
  - Vision: NVIDIA vision API (image → screen analysis)
  - Embeddings: NV-Embed (text → deterministic vectors)

Features:
  - Per-provider rate limiting (token bucket, realistic limits)
  - Forced failures (500, 429/rate-limit, timeouts)
  - Latency injection
  - Provider health scoring
  - Dynamic tool call generation from request tool definitions
"""

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

app = FastAPI(title="Mock Provider Server")

# ── Configuration ──

MOCK_PORT = int(os.getenv("MOCK_PROVIDER_PORT", "8888"))

RATE_LIMITS: dict[str, dict[str, float]] = {
    "Nemotron Ultra": {"req_min": 30, "burst": 5, "tokens_per_min": 10000},
    "DeepSeek": {"req_min": 60, "burst": 10, "tokens_per_min": 20000},
    "Groq": {"req_min": 1800, "burst": 30, "tokens_per_min": 100000},
    "Gemini": {"req_min": 60, "burst": 10, "tokens_per_min": 32000},
    "Kimi K2": {"req_min": 30, "burst": 5, "tokens_per_min": 15000},
    "NVIDIA NIM": {"req_min": 30, "burst": 5, "tokens_per_min": 10000},
    "OpenRouter": {"req_min": 100, "burst": 20, "tokens_per_min": 50000},
    "Pollinations": {"req_min": 1000, "burst": 100, "tokens_per_min": 200000},
}

PROVIDER_NAMES = [
    "Nemotron Ultra",
    "DeepSeek",
    "Groq",
    "Gemini",
    "Kimi K2",
    "NVIDIA NIM",
    "OpenRouter",
    "Pollinations",
]


def _model_to_provider(model: str) -> str:
    m = model.lower()
    if "nemotron-3-ultra" in m:
        return "Nemotron Ultra"
    if "nemotron-3-nano" in m or "nim-" in m:
        return "NVIDIA NIM"
    if "deepseek" in m:
        return "DeepSeek"
    if "kimi" in m:
        return "Kimi K2"
    if "llama" in m:
        return "Groq"
    if "gemini" in m:
        return "Gemini"
    if "pollinations" in m:
        return "Pollinations"
    if "openrouter" in m:
        return "OpenRouter"
    return "Gemini"


# ── State ──


class TokenBucket:
    def __init__(self, rate: float, burst: float):
        self.rate = rate / 60.0
        self.burst = burst
        self.tokens = burst
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


_provider_state: dict[str, dict] = {
    p: {
        "fail_mode": None,
        "fail_until": 0,
        "latency_ms": 0,
        "health_score": 100,
        "call_count": 0,
        "bucket": TokenBucket(
            RATE_LIMITS[p]["req_min"],
            RATE_LIMITS[p]["burst"],
        ),
    }
    for p in PROVIDER_NAMES
}

_state_lock = threading.Lock()


def _get_state(provider: str) -> dict:
    with _state_lock:
        return _provider_state.get(provider, {})


def _update_state(provider: str, **kwargs):
    with _state_lock:
        for k, v in kwargs.items():
            _provider_state.setdefault(provider, {})[k] = v


def _record_call(provider: str, success: bool):
    with _state_lock:
        s = _provider_state.setdefault(provider, {})
        s["call_count"] = s.get("call_count", 0) + 1
        if success:
            s["health_score"] = min(100, s.get("health_score", 100) + 5)
        else:
            s["health_score"] = max(0, s.get("health_score", 100) - 10)


def _is_failing(provider: str) -> str | None:
    s = _get_state(provider)
    if s.get("fail_mode") and time.time() > s.get("fail_until", 0):
        _update_state(provider, fail_mode=None)
        return None
    return s.get("fail_mode")


def _check_rate_limit(provider: str):
    s = _get_state(provider)
    bucket = s.get("bucket")
    if bucket and not bucket.consume():
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "message": f"Rate limit exceeded for {provider}",
                    "type": "rate_limit_error",
                },
                "retry_after": 60,
            },
        )


def _apply_latency(provider: str):
    s = _get_state(provider)
    ms = s.get("latency_ms", 0)
    if ms > 0:
        time.sleep(ms / 1000.0)


# ── Tool Call Generation ──


def _last_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part["text"]
                return ""
            return content or ""
    return ""


def _has_image(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _pick_tool(user_msg: str, tools: list[dict]) -> dict | None:
    if not tools:
        return None
    msg_lower = user_msg.lower()

    keyword_tool_map = [
        ("weather", ["get_weather_detailed", "get_weather"]),
        ("weather", ["weather"]),
        ("remember", ["save_memory"]),
        ("forget", ["forget_memory"]),
        ("search", ["web_search"]),
        ("open ", ["open_app"]),
        ("launch", ["open_app"]),
        ("quit ", ["quit_app"]),
        ("close ", ["quit_app"]),
        ("navigate", ["browser_navigate"]),
        ("go to", ["browser_navigate"]),
        ("spotify", ["spotify_play", "spotify_pause", "spotify_skip", "spotify_search"]),
        ("play", ["spotify_play"]),
        ("pause", ["spotify_pause"]),
        ("skip", ["spotify_skip"]),
        ("discord", ["discord_open_channel", "discord_send_message"]),
        ("send ", ["send_imessage"]),
        ("message", ["send_imessage"]),
        ("calendar", ["get_calendar_events", "add_calendar_event"]),
        ("schedule", ["add_calendar_event"]),
        ("event", ["get_calendar_events"]),
        ("screenshot", ["take_screenshot"]),
        ("screen", ["take_screenshot"]),
        ("click", ["click_on_screen", "move_and_click"]),
        ("mouse", ["move_and_click"]),
        ("type", ["type_text"]),
        ("press", ["press_key"]),
        ("system", ["get_system_usage", "get_disk_space"]),
        ("cpu", ["get_system_usage"]),
        ("ram", ["get_system_usage"]),
        ("memory", ["get_system_usage"]),
        ("disk", ["get_disk_space"]),
        ("file", ["read_file", "write_file", "find_file"]),
        ("read ", ["read_file"]),
        ("write ", ["write_file"]),
        ("list", ["list_files"]),
        ("find ", ["find_file"]),
        ("timer", ["set_timer", "cancel_timer"]),
        ("set a timer", ["set_timer"]),
        ("cancel timer", ["cancel_timer"]),
        ("ip address", ["get_ip_address"]),
        ("battery", ["get_battery_level"]),
        ("email", ["compose_gmail"]),
        ("gmail", ["compose_gmail"]),
        ("docs", ["google_docs_create", "google_docs_read"]),
        ("slides", ["google_slides_create"]),
        ("forms", ["google_forms_create"]),
        ("note", ["knowledge_query", "knowledge_add"]),
        ("learn", ["trigger_learning"]),
        ("capabilities", ["inspect_capabilities"]),
        ("tool", ["list_own_tools", "inspect_capabilities"]),
        ("translate", ["translate"]),
        ("joke", ["tell_joke"]),
        ("news", ["warwatch_news"]),
    ]

    for keyword, candidates in keyword_tool_map:
        if keyword in msg_lower:
            for tool_def in tools:
                name = tool_def.get("function", tool_def).get("name", "")
                for cand in candidates:
                    if cand in name or name in cand:
                        args = _infer_args(name, user_msg)
                        return {"name": name, "arguments": json.dumps(args)}
    return None


def _infer_args(tool_name: str, user_msg: str) -> dict:
    args: dict[str, Any] = {}
    if "query" in tool_name or tool_name in ("web_search",):
        args["query"] = user_msg.replace("search", "", 1).replace("search web for", "", 1).replace("search for", "", 1).strip() or user_msg
    elif tool_name in ("open_app", "quit_app"):
        apps = {"safari": "Safari", "chrome": "Chrome", "discord": "Discord", "spotify": "Spotify", "calculator": "Calculator", "terminal": "Terminal"}
        for k, v in apps.items():
            if k in user_msg.lower():
                args["app_name"] = v
                break
        if "app_name" not in args:
            args["app_name"] = user_msg.split()[-1].capitalize()
    elif tool_name == "send_imessage":
        args["recipient"] = "test"
        args["message"] = "Hello from test"
    elif tool_name == "web_search":
        args["query"] = user_msg
    elif tool_name == "read_file":
        args["path"] = "/tmp/test.txt"
        args.get("offset", 0)
        args.get("limit", 100)
    elif tool_name == "write_file":
        args["path"] = "/tmp/test.txt"
        args["content"] = "test content"
    elif tool_name in ("set_timer",):
        args["duration"] = 10
        args["label"] = "test"
    return args


def _build_openai_tool_call(tool_name: str, arguments: str) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": arguments,
        },
    }


def _build_gemini_tool_call(tool_name: str, arguments: dict) -> dict:
    return {
        "functionCall": {
            "name": tool_name,
            "args": arguments,
        }
    }


# ── Embeddings (deterministic) ──


def _embed(text: str, dims: int = 768) -> list[float]:
    h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(h)
    vec = rng.randn(dims).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ── Endpoints ──


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "unknown")
    messages = body.get("messages", [])
    tools = body.get("tools", body.get("functions", []))
    provider = _model_to_provider(model)

    _check_rate_limit(provider)
    _apply_latency(provider)

    fail_mode = _is_failing(provider)

    user_msg = _last_user_message(messages)
    has_tools = bool(tools)
    has_tool_results = any(m.get("role") in ("tool", "function") for m in messages)
    stream = body.get("stream", False)

    try:
        if fail_mode == "timeout":
            time.sleep(120)
            raise HTTPException(504, "Gateway Timeout")
        if fail_mode == "500":
            raise HTTPException(500, "Internal Server Error")
        if fail_mode == "429":
            raise HTTPException(429, {"error": {"message": "Rate limit", "type": "rate_limit_error"}, "retry_after": 60})
        if fail_mode == "402":
            raise HTTPException(402, "Payment Required: insufficient credits")
        if fail_mode:
            raise HTTPException(503, f"Service unavailable ({fail_mode})")

        # If tool results are present, return a final text response
        if has_tool_results:
            _record_call(provider, True)
            if stream:
                return StreamingResponse(_stream_text(model, f"Mock result for: {user_msg}"), media_type="text/event-stream")
            return _openai_text_response(model, f"Mock result for: {user_msg}")

        # If tools are available, try to pick one
        if has_tools:
            chosen = _pick_tool(user_msg, tools)
            if chosen:
                _record_call(provider, True)
                if stream:
                    return StreamingResponse(_stream_tool_call(model, chosen["name"], chosen["arguments"]), media_type="text/event-stream")
                return _openai_tool_response(model, chosen["name"], chosen["arguments"])

        # No tool match, or no tools defined — return text
        _record_call(provider, True)
        if stream:
            return StreamingResponse(_stream_text(model, f"Mock response for: {user_msg}"), media_type="text/event-stream")
        return _openai_text_response(model, f"Mock response for: {user_msg}")
    except HTTPException:
        _record_call(provider, False)
        raise


@app.post("/v1beta/models/{model}:generateContent")
async def gemini_generate(model: str, request: Request):
    body = await request.json()
    contents = body.get("contents", [])
    tools_config = body.get("tools", [])
    messages = []
    for c in contents:
        role = c.get("role", "user")
        parts = c.get("parts", [])
        for p in parts:
            if "text" in p:
                messages.append({"role": role, "content": p["text"]})
            if "functionCall" in p:
                messages.append({"role": "assistant", "function_call": p["functionCall"]})
            if "functionResponse" in p:
                messages.append({"role": "function", "content": json.dumps(p["functionResponse"]["response"])})

    provider = "Gemini"
    _check_rate_limit(provider)
    fail_mode = _is_failing(provider)

    try:
        if fail_mode == "timeout":
            time.sleep(120)
            raise HTTPException(504, "Gateway Timeout")
        if fail_mode == "500":
            raise HTTPException(500, "Internal Server Error")
        if fail_mode == "402":
            raise HTTPException(402, "Payment Required: insufficient credits")
        if fail_mode:
            raise HTTPException(503, f"Service unavailable ({fail_mode})")

        _apply_latency(provider)

        user_msg = _last_user_message(messages)
        has_tool_results = any(m.get("role") == "function" for m in messages)

        if has_tool_results:
            _record_call(provider, True)
            return _gemini_text_response(model, f"Mock result for: {user_msg}")

        # Check for function declarations from tools config
        declarations = []
        for t in tools_config:
            for fd in t.get("function_declarations", []):
                declarations.append(fd)

        if declarations:
            chosen = _pick_tool(user_msg, declarations)
            if chosen:
                _record_call(provider, True)
                return _gemini_tool_response(model, chosen["name"], json.loads(chosen["arguments"]))

        _record_call(provider, True)
        return _gemini_text_response(model, f"Mock response for: {user_msg}")
    except HTTPException:
        _record_call(provider, False)
        raise


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    model = body.get("model", "nv-embed")
    input_texts = body.get("input", [])
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    provider = "NVIDIA NIM"
    _check_rate_limit(provider)
    fail_mode = _is_failing(provider)
    if fail_mode:
        raise HTTPException(503, f"Service unavailable ({fail_mode})")

    dims = 768
    if "1024" in model:
        dims = 1024
    elif "2048" in model:
        dims = 2048

    data = []
    for i, text in enumerate(input_texts):
        vec = _embed(text, dims)
        data.append(
            {
                "object": "embedding",
                "index": i,
                "embedding": vec,
            }
        )
    _record_call(provider, True)
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in input_texts),
            "total_tokens": sum(len(t.split()) for t in input_texts),
        },
    }


# ── Control Endpoints ──


@app.get("/mock/health/{provider}")
async def mock_health(provider: str):
    s = _get_state(provider)
    return {
        "provider": provider,
        "available": s.get("health_score", 100) > 0 and _is_failing(provider) is None,
        "health_score": s.get("health_score", 100),
        "call_count": s.get("call_count", 0),
        "fail_mode": s.get("fail_mode"),
        "latency_ms": s.get("latency_ms", 0),
    }


@app.get("/mock/state")
async def mock_state():
    def _safe_state(s: dict) -> dict:
        return {k: v for k, v in s.items() if k != "bucket"}

    return {p: _safe_state(_get_state(p)) for p in PROVIDER_NAMES}


@app.post("/mock/fail/{provider}")
async def mock_fail(provider: str, mode: str = "500", duration: int = 30):
    _update_state(provider, fail_mode=mode, fail_until=time.time() + duration)
    return {"status": "ok", "provider": provider, "mode": mode, "duration": duration}


@app.post("/mock/unfail/{provider}")
async def mock_unfail(provider: str):
    _update_state(provider, fail_mode=None, fail_until=0)
    return {"status": "ok", "provider": provider}


@app.post("/mock/rate-limit/{provider}")
async def mock_rate_limit(provider: str, retry_after: int = 60):
    _update_state(provider, fail_mode="429", fail_until=time.time() + retry_after)
    return {"status": "ok", "provider": provider, "retry_after": retry_after}


@app.post("/mock/latency/{provider}")
async def mock_latency(provider: str, ms: int = 0):
    _update_state(provider, latency_ms=ms)
    return {"status": "ok", "provider": provider, "latency_ms": ms}


@app.post("/mock/validate-tool-call")
async def mock_validate_tool_call(request: Request):
    """Validate a tool call against known schemas."""
    body = await request.json()
    tool_name = body.get("tool_name", "")
    arguments = body.get("arguments", {})
    tool_schemas = body.get("tool_schemas", [])

    # Find matching schema
    for ts in tool_schemas:
        name = ts.get("function", ts).get("name", "")
        if name == tool_name:
            params = ts.get("function", ts).get("parameters", {})
            required = params.get("required", [])
            properties = params.get("properties", {})
            missing = [r for r in required if r not in arguments]
            extra = [k for k in arguments if k not in properties]
            return {
                "valid": len(missing) == 0,
                "tool_name": tool_name,
                "missing_required": missing,
                "extra_args": extra,
                "argument_count": len(arguments),
                "expected_required": required,
            }
    return {"valid": False, "tool_name": tool_name, "error": "No matching schema found"}


@app.post("/mock/reset")
async def mock_reset():
    global _provider_state
    with _state_lock:
        _provider_state = {
            p: {
                "fail_mode": None,
                "fail_until": 0,
                "latency_ms": 0,
                "health_score": 100,
                "call_count": 0,
                "bucket": TokenBucket(
                    RATE_LIMITS[p]["req_min"],
                    RATE_LIMITS[p]["burst"],
                ),
            }
            for p in PROVIDER_NAMES
        }
    return {"status": "ok", "message": "All provider state reset"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "mock-provider", "port": MOCK_PORT}


# ── Streaming Support ──


async def _stream_text(model: str, text: str):
    """SSE streaming for OpenAI-compatible chat completions."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    words = text.split()
    for i, word in enumerate(words):
        is_last = i == len(words) - 1
        data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": "stop" if is_last else None,
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.05)
    yield "data: [DONE]\n\n"


async def _stream_tool_call(model: str, tool_name: str, arguments: str):
    """SSE streaming for tool call responses."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    # First chunk: delta with tool_calls start
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': call_id, 'type': 'function', 'function': {'name': tool_name, 'arguments': ''}}]}, 'finish_reason': None}]})}\n\n"
    # Yield arguments in chunks
    chunk_size = 10
    for i in range(0, len(arguments), chunk_size):
        chunk = arguments[i : i + chunk_size]
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': chunk}}]}, 'finish_reason': None}]})}\n\n"
        await asyncio.sleep(0.02)
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
    yield "data: [DONE]\n\n"


# ── Response Builders ──


def _openai_text_response(model: str, text: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": len(text.split()),
            "total_tokens": 50 + len(text.split()),
        },
    }


def _openai_tool_response(model: str, tool_name: str, arguments: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_build_openai_tool_call(tool_name, arguments)],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
        },
    }


def _gemini_text_response(model: str, text: str) -> dict:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finish_reason": 1,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": len(text.split()),
            "totalTokenCount": 50 + len(text.split()),
        },
    }


def _gemini_tool_response(model: str, tool_name: str, arguments: dict) -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [_build_gemini_tool_call(tool_name, arguments)],
                },
                "finish_reason": 2,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 10,
            "totalTokenCount": 60,
        },
    }


# ── Main ──


@app.on_event("startup")
async def startup():
    """Apply per-provider latency from MOCK_PROVIDER_LATENCY_MS env var.
    Format: "Nemotron Ultra=100,Gemini=200,DeepSeek=50"
    """
    latency_env = os.getenv("MOCK_PROVIDER_LATENCY_MS", "")
    if latency_env:
        for entry in latency_env.split(","):
            entry = entry.strip()
            if "=" in entry:
                provider, ms_str = entry.split("=", 1)
                try:
                    _update_state(provider.strip(), latency_ms=int(ms_str.strip()))
                except (ValueError, IndexError):
                    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MOCK_PORT, log_level="info")
