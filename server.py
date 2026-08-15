import asyncio
import glob
import json
import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import proactive
from brain import get_conversation_context, get_runtime_status, process
from event_bus import subscribe, publish
from brain import init as brain_init
from config import (
    FACE_SIMILARITY_THRESHOLD,
    JARVIS_API_KEY,
    SCREEN_BUFFER_SIZE,
    SCREEN_CAPTURE_INTERVAL,
)
from tools.github_tools import github_auth_url, github_handle_callback
from tools.gmail_tools import gmail_auth_url, gmail_handle_callback
from tools.google_docs_tools import docs_auth_url, docs_handle_callback
from tools.google_drive_tools import gdrive_auth_url, gdrive_handle_callback
from tools.google_forms_tools import forms_auth_url, forms_handle_callback
from tools.google_sheets_tools import gsheets_auth_url, gsheets_handle_callback
from tools.google_slides_tools import slides_auth_url, slides_handle_callback
from tools.token_store import TokenStore
from trigger_engine import start as triggers_start
from tts import ELEVENLABS_API_KEY, ELEVENLABS_MODEL, ELEVENLABS_VOICE_ID, speak, stop_speaking

# ── App setup ────────────────────────────────
app = FastAPI(title="Jarvis API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

_screen_buffer: list[tuple[int, str]] = []
_screen_lock = threading.Lock()
_screenshot_dir = os.path.join(tempfile.gettempdir(), "jarvis_screen")
_face_enrolled: bool | None = None  # None = unchecked

# ── Start engines ─────────────────────────────
brain_init()
proactive.init(speak, process)
proactive.start()
triggers_start()
try:
    from self_test.monitor import start as self_test_monitor_start

    self_test_monitor_start()
except Exception:
    pass


# ── Models ───────────────────────────────────
class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    reply: str
    transcription: str = ""


# ── Routes ───────────────────────────────────
# ── WebSocket Event Stream ───────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return f.read()
    return "<h1>Jarvis is running but UI not found.</h1>"


@app.get("/static/manifest.json")
async def manifest():
    return FileResponse(os.path.join(static_dir, "manifest.json"))


@app.get("/health")
async def health():
    from healthcheck import run_healthcheck

    hc = run_healthcheck()
    return {"status": "online", "message": "Jarvis is running.", "runtime": get_runtime_status(), "checks": hc}


@app.get("/health/providers")
async def health_providers():
    from brain import get_provider_health

    return get_provider_health()


def _debug_http_enabled() -> bool:
    """/debug/* endpoints are off unless JARVIS_DEBUG_HTTP=1.

    The server binds 0.0.0.0 and may be exposed publicly (Tailscale Funnel);
    decision traces can contain commands/paths, so the surface is
    default-closed. JARVIS_DEBUG_TOKEN (if set) additionally requires
    `Authorization: Bearer <token>`.
    """
    return os.getenv("JARVIS_DEBUG_HTTP", "0").lower() in ("1", "true", "yes", "on")


async def _debug_gate(request: Request) -> None:
    if not _debug_http_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    token = os.getenv("JARVIS_DEBUG_TOKEN", "")
    if token:
        provided = (request.headers.get("Authorization", "") or "").removeprefix("Bearer ").strip()
        if not provided or provided != token:
            raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/debug/decisions")
async def debug_decisions(request: Request, request_id: str = None, limit: int = 100):
    """Return decision traces for a request or recent requests."""
    await _debug_gate(request)
    from decision_log import load_recent_decisions

    events = load_recent_decisions(limit=limit, request_id=request_id)
    return {"events": events}


@app.get("/debug/decisions/{request_id}")
async def debug_decision_trace(request: Request, request_id: str):
    """Full trace for a single request, ordered by timestamp."""
    await _debug_gate(request)
    from decision_log import get_decision_trace

    trace = get_decision_trace(request_id)
    return {"request_id": request_id, "trace": trace}


@app.post("/ask", response_model=TextResponse)
async def ask_text(req: TextRequest, tts: str = "server"):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, process, req.text)
    if tts == "server":
        speak(reply)
    return TextResponse(reply=reply)


@app.post("/ask-voice", response_model=TextResponse)
async def ask_voice(audio: UploadFile = File(...)):
    """
    Accepts audio upload, transcribes with whisper.cpp,
    processes through Jarvis brain, speaks reply.
    Uses pre-loaded model — no reload on each request.
    """
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Save to temp file
    suffix = ".webm"
    if audio.filename:
        ext = os.path.splitext(audio.filename)[-1]
        if ext:
            suffix = ext

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        # Convert to wav if needed (ffmpeg handles webm/m4a/etc)
        wav_path = tmp_path.replace(suffix, ".wav")
        try:
            import subprocess

            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
                capture_output=True,
                check=True,
            )
            transcribe_path = wav_path
        except Exception:
            # ffmpeg not available or failed — use original
            transcribe_path = tmp_path

        # Transcribe using pre-loaded whisper.cpp model
        from stt import transcribe_file

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, transcribe_file, transcribe_path)

    finally:
        # Clean up temp files
        for p in [tmp_path, wav_path if "wav_path" in locals() else None]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Couldn't transcribe audio.")

    # Process through Jarvis brain
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, process, text.strip())

    return TextResponse(reply=reply, transcription=text.strip())


@app.post("/tts/generate")
async def tts_generate(req: TextRequest):
    """Generate ElevenLabs TTS audio for client-side playback (fallback when iOS TTS unavailable)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="TTS not configured on server")
    try:
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=req.text,
            model_id=ELEVENLABS_MODEL,
        )
        from fastapi.responses import Response

        return Response(content=audio, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")


PARTIAL_INTERVAL = 2.0  # seconds between partial transcriptions


@app.websocket("/ws")
async def websocket_stream(ws: WebSocket):
    """WebSocket endpoint for streaming voice: audio chunks → partial transcription → reply + TTS audio."""
    await ws.accept()

    audio_chunks = bytearray()
    last_partial = time.time()
    partial_count = 0
    _tmp_files = []

    try:
        while True:
            data = await ws.receive_bytes()

            if data == b"END":
                break

            audio_chunks.extend(data)

            # Partial transcription every PARTIAL_INTERVAL seconds
            now = time.time()
            if now - last_partial >= PARTIAL_INTERVAL and len(audio_chunks) > 16000:
                partial_text = _transcribe_audio(audio_chunks, _tmp_files)
                if partial_text and partial_count < 3:
                    await ws.send_json({"type": "partial", "text": partial_text.strip()})
                    partial_count += 1
                last_partial = now

        # Full transcription + processing
        if not audio_chunks:
            await ws.send_json({"type": "final", "transcription": "", "reply": "No audio received."})
            await ws.close()
            return

        full_text = _transcribe_audio(audio_chunks, _tmp_files)
        if not full_text or not full_text.strip():
            await ws.send_json({"type": "final", "transcription": "", "reply": "Couldn't transcribe audio."})
            await ws.close()
            return

        full_text = full_text.strip()
        await ws.send_json({"type": "transcribed", "text": full_text})

        # Process through brain
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(None, process, full_text)

        await ws.send_json({"type": "final", "transcription": full_text, "reply": reply})

        # Stream TTS audio back via ElevenLabs
        if ELEVENLABS_API_KEY and reply:
            await ws.send_json({"type": "tts_start"})
            try:
                from elevenlabs import ElevenLabs

                client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                audio_stream = client.text_to_speech.stream(
                    voice_id=ELEVENLABS_VOICE_ID,
                    text=reply,
                    model_id=ELEVENLABS_MODEL,
                    output_format="mp3_44100_128",
                    optimize_streaming_latency=4,
                )
                for chunk in audio_stream:
                    if chunk:
                        await ws.send_bytes(chunk)
            except Exception as e:
                print(f"  WebSocket TTS stream error: {e}")
            await ws.send_json({"type": "tts_done"})

        await ws.close()

    except WebSocketDisconnect:
        pass
    finally:
        for p in _tmp_files:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def _transcribe_audio(audio_data: bytearray, tmp_files: list) -> str:
    """Write audio bytes to temp file, convert to wav, transcribe, return text."""
    suffix = ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        tmp_path = f.name
        tmp_files.append(tmp_path)

    wav_path = tmp_path.replace(suffix, ".wav")
    try:
        import subprocess

        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            capture_output=True,
            check=True,
        )
        tmp_files.append(wav_path)
    except Exception:
        wav_path = tmp_path

    from stt import transcribe_file

    try:
        return transcribe_file(wav_path)
    except Exception:
        return ""

# ── Event Stream for Push Notifications ────────────────────────────────────
# Global set of connected WebSocket clients for event broadcasting.
_connected_ws_clients: set[WebSocket] = set()
_ws_lock = threading.Lock()
_event_loop: asyncio.AbstractEventLoop | None = None

@app.on_event("startup")
async def _event_bus_startup():
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    # Subscribe to internal event bus and forward to WS clients
    from event_bus import subscribe

    def _forward(payload: dict):
        # Serialize payload and dispatch to all WS clients
        if not _event_loop:
            return
        msg = json.dumps(payload)
        async def _send():
            to_remove = []
            with _ws_lock:
                for ws in list(_connected_ws_clients):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        to_remove.append(ws)
            for ws in to_remove:
                _connected_ws_clients.discard(ws)
        # Schedule coroutine in the event loop
        asyncio.run_coroutine_threadsafe(_send(), _event_loop)

    # Forward the main event types we care about
    for et in ["proactive_alert", "subagent_started", "subagent_progress", "subagent_completed"]:
        subscribe(et, _forward)

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """WebSocket endpoint for real‑time event push.
    Clients receive JSON messages for each published event.
    """
    await ws.accept()
    # Register client
    with _ws_lock:
        _connected_ws_clients.add(ws)
    try:
        # Keep connection alive – client can send ping/pong or simple text.
        while True:
            msg = await ws.receive_text()
            if msg is None:
                break
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            _connected_ws_clients.discard(ws)

# REST endpoint for polling fallback (optional)
@app.get("/events/recent")
async def get_recent_events(event_type: str = "", limit: int = 20, since: str = ""):
    """Return recent events for polling clients.
    * ``event_type`` – filter by type (empty = all)
    * ``limit`` – max events per type
    * ``since`` – ISO timestamp; only events after this time are returned
    """
    from event_bus import get_all_recent, get_recent
    if event_type:
        events = get_recent(event_type, limit)
        return {"type": event_type, "events": events}
    else:
        all_events = get_all_recent(limit_per_type=limit)
        return {"events": all_events}



@app.get("/audit")
async def audit_endpoint():
    from safety import get_audit_log

    logs = get_audit_log(50)
    return {"logs": [{"tool": row[0], "args": row[1], "level": row[2], "decision": row[3], "time": row[4]} for row in logs]}


@app.get("/priorities")
async def priorities_endpoint():
    from priority import get_priority_stats

    stats = get_priority_stats()
    return {"priorities": [{"type": s[0], "base": s[1], "current": round(s[2], 2), "ignored": s[3], "acknowledged": s[4]} for s in stats]}


@app.get("/notices")
async def notices(unread_only: bool = True, limit: int = 50):
    from priority import get_notifications

    return {"notifications": get_notifications(unread_only=unread_only, limit=limit)}


@app.post("/notices/{notice_id}/read")
async def notice_read(notice_id: int):
    from priority import mark_notification_read

    mark_notification_read(notice_id)
    return {"status": "ok"}


@app.post("/notices/read-all")
async def notices_read_all():
    from priority import mark_all_notifications_read

    mark_all_notifications_read()
    return {"status": "ok"}


@app.post("/stop")
async def stop():
    stop_speaking()
    return {"status": "stopped"}


@app.post("/brain/reset")
async def brain_reset():
    from brain import clear_pending_safe, conversation, conversation_context

    clear_pending_safe()
    conversation.clear()
    conversation_context.__init__()
    return {"status": "reset"}


@app.get("/weather")
async def weather_endpoint():
    from tools.system_tools import get_weather_detailed

    return {"result": get_weather_detailed()}


@app.post("/vision/analyze")
async def vision_analyze(req: TextRequest):
    from tools.vision_tools import analyze_image

    try:
        data = json.loads(req.text)
        path = data.get("path", "")
        url = data.get("url", "")
        question = data.get("question", "Describe this image.")
    except Exception:
        path = req.text
        url = ""
        question = "Describe this image."
    result = analyze_image(path=path, url=url, question=question)
    return {"result": result}


@app.post("/vision/ocr")
async def vision_ocr(req: TextRequest):
    from tools.vision_tools import ocr_document

    result = ocr_document(req.text.strip())
    return {"result": result[:2000]}


@app.post("/vision/video")
async def vision_video(path: str = "", timestamps: str = "0"):
    from tools.vision_tools import analyze_video

    result = analyze_video(path, timestamps=timestamps)
    return {"result": result}


@app.get("/system")
async def system_endpoint():
    from brain import SYSTEM_INFO
    from tools.system_tools import get_system_info

    info = get_system_info()
    info.update(SYSTEM_INFO)
    return info


@app.get("/runtime")
async def runtime_endpoint():
    return get_runtime_status()


@app.get("/metrics")
async def metrics_endpoint():
    from jarvis_logger import get_cost_summary, get_metrics_snapshot

    m = get_metrics_snapshot()
    c = get_cost_summary()
    lines = [
        "# HELP jarvis_requests_total Total requests processed",
        "# TYPE jarvis_requests_total counter",
        f"jarvis_requests_total {m['requests_total']}",
        "",
        "# HELP jarvis_tokens_input_total Total input tokens",
        "# TYPE jarvis_tokens_input_total counter",
        f"jarvis_tokens_input_total {m['tokens_input_total']}",
        "",
        "# HELP jarvis_tokens_output_total Total output tokens",
        "# TYPE jarvis_tokens_output_total counter",
        f"jarvis_tokens_output_total {m['tokens_output_total']}",
        "",
        "# HELP jarvis_tool_calls_total Total tool calls",
        "# TYPE jarvis_tool_calls_total counter",
        f"jarvis_tool_calls_total {m['tool_calls_total']}",
        "",
        "# HELP jarvis_tool_errors_total Total tool errors",
        "# TYPE jarvis_tool_errors_total counter",
        f"jarvis_tool_errors_total {m['tool_errors_total']}",
        "",
        "# HELP jarvis_latency_seconds Average response latency",
        "# TYPE jarvis_latency_seconds gauge",
        f"jarvis_latency_seconds_overall {m['latency_avg_overall']}",
    ]
    for prov, avg in m.get("latency_avg_by_provider", {}).items():
        lines.append(f'jarvis_latency_seconds{{provider="{prov}"}} {avg}')
    for intent, count in sorted(m.get("requests_by_intent", {}).items()):
        lines.append(f'jarvis_requests_by_intent{{intent="{intent}"}} {count}')
    lines.append("")
    lines.append("# HELP jarvis_cost_usd_total Estimated total API cost (USD)")
    lines.append("# TYPE jarvis_cost_usd_total counter")
    lines.append(f"jarvis_cost_usd_total {c.get('cost_usd_total', 0):.4f}")
    lines.append("")
    lines.append("# HELP jarvis_cost_usd_monthly Estimated monthly API cost (USD)")
    lines.append("# TYPE jarvis_cost_usd_monthly gauge")
    lines.append(f"jarvis_cost_usd_monthly {c.get('estimated_monthly', 0):.4f}")
    return "\n".join(lines) + "\n"


@app.get("/intents")
async def intents_endpoint():
    from jarvis_logger import count_intents, get_metrics_snapshot

    persisted = count_intents()
    snapshot = get_metrics_snapshot().get("requests_by_intent", {})
    intents = sorted(set(persisted) | set(snapshot))
    return {
        "intents": [
            {
                "intent": name,
                "count": persisted.get(name, snapshot.get(name, 0)),
                "session_count": snapshot.get(name, 0),
                "persisted": name in persisted,
            }
            for name in intents
        ],
        "total": sum(persisted.values()),
    }


@app.get("/recap")
async def recap_endpoint():
    from watchlog import build_recap

    return {"result": build_recap(hours=8)}


# ── Mode / Queue / Context (Native App) ──

_QUEUE_ITEMS: list[str] = []
_MODE = "text"


class ModeRequest(BaseModel):
    mode: str


@app.post("/mode")
async def set_mode(req: ModeRequest):
    global _MODE
    mode = req.mode.lower()
    if mode not in ("text", "paste", "queue"):
        raise HTTPException(400, "Mode must be text, paste, or queue")
    _MODE = mode
    return {"mode": _MODE}


@app.get("/mode")
async def get_mode():
    return {"mode": _MODE}


class QueueItem(BaseModel):
    text: str


@app.get("/queue")
async def get_queue():
    return {"items": _QUEUE_ITEMS, "count": len(_QUEUE_ITEMS)}


@app.post("/queue")
async def add_to_queue(req: QueueItem):
    _QUEUE_ITEMS.append(req.text)
    return {"added": True, "count": len(_QUEUE_ITEMS)}


@app.post("/queue/execute")
async def execute_queue():
    items = list(_QUEUE_ITEMS)
    _QUEUE_ITEMS.clear()
    return {"executed": len(items), "items": items}


@app.post("/queue/clear")
async def clear_queue():
    _QUEUE_ITEMS.clear()
    return {"cleared": True}


@app.get("/context")
async def context_endpoint():
    return get_conversation_context()


@app.post("/confirm")
async def confirm_action(req: TextRequest):
    from brain import clear_pending_safe, execute_pending_safe, has_pending_safe

    if not has_pending_safe():
        return {"status": "no_pending"}
    answer = req.text.strip().lower()
    if answer in ("yes", "y", "true", "1", "approve", "confirm"):
        result = execute_pending_safe()
        return {"status": "confirmed", "result": str(result) if result else ""}
    else:
        clear_pending_safe()
        return {"status": "cancelled", "result": "Okay, cancelled."}


@app.get("/workflows")
async def workflows_list():
    from workflow_engine import list_workflows

    return {"workflows": list_workflows()}


@app.post("/workflows/run")
async def workflows_run(name: str = "", params: str = "{}"):
    from workflow_engine import run_workflow

    try:
        p = json.loads(params)
    except json.JSONDecodeError:
        p = {}
    result = run_workflow(name, p)
    return result


@app.get("/workflows/history")
async def workflows_history(limit: int = 20):
    from workflow_engine import get_run_history

    return {"runs": get_run_history(limit)}


@app.get("/workflows/runs/{run_id}")
async def workflows_run_detail(run_id: int):
    from workflow_engine import get_run_detail

    detail = get_run_detail(run_id)
    if not detail:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Run not found")
    return detail


@app.get("/eval/latest")
async def eval_latest():
    from eval_runner import load_latest_report

    report = load_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No eval report found")
    return report


@app.get("/eval/history")
async def eval_history(limit: int = 10):
    from eval_runner import load_eval_history

    return {"history": load_eval_history(limit)}


@app.post("/eval/run")
async def eval_run(no_api: bool = False):
    from eval_runner import run_eval_suite

    report = run_eval_suite(require_no_api=no_api)
    return report


@app.get("/graph/stats")
async def graph_stats():
    from graph_memory import get_graph_summary

    return {"summary": get_graph_summary()}


@app.post("/graph/extract")
async def graph_extract(req: TextRequest):
    from graph_memory import extract_entities_relations

    results = extract_entities_relations(req.text)
    return {"extracted": len(results), "relationships": results}


@app.get("/graph/neighbors")
async def graph_neighbors(entity: str):
    from graph_memory import query_relationships, search_neighbors

    return {
        "description": query_relationships(entity),
        "neighbors": search_neighbors(entity),
    }


@app.get("/graph/search")
async def graph_search(query: str):
    from graph_memory import hybrid_graph_search

    return {"results": hybrid_graph_search(query)}


# ── Learner ──


@app.get("/learner/tools")
async def learner_tools():
    from learner import get_learned_tools

    return {"tools": get_learned_tools()}


@app.delete("/learner/tools/{name}")
async def learner_delete_tool(name: str):
    from learner import delete_learned_tool

    return delete_learned_tool(name)


@app.get("/learner/stats")
async def learner_stats():
    from learner import get_learning_stats

    return get_learning_stats()


@app.post("/learner/trigger")
async def learner_trigger(req: TextRequest):
    from learner import trigger_learning

    return trigger_learning(req.text)


@app.get("/perf/providers")
async def perf_providers():
    from perf_router import get_provider_stats

    return {"providers": get_provider_stats()}


@app.get("/perf/cache")
async def perf_cache():
    from perf_router import get_semantic_cache_stats

    return get_semantic_cache_stats()


@app.get("/agents")
async def agents_list():
    from agent import list_agents

    return {"agents": list_agents()}


@app.post("/agents/spawn")
async def agent_spawn(goal: str):
    from agent import spawn_agent

    aid = spawn_agent(goal)
    return {"agent_id": aid, "goal": goal}


@app.get("/agents/{agent_id}")
async def agent_status(agent_id: str):
    from agent import get_agent

    a = get_agent(agent_id)
    if not a:
        raise HTTPException(status_code=404, detail="Agent not found")
    return a.to_dict()


@app.post("/agents/{agent_id}/stop")
async def agent_stop(agent_id: str):
    from agent import stop_agent

    if stop_agent(agent_id):
        return {"status": "stopped"}
    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/mobile/register")
async def mobile_register(token: str, platform: str = "web", name: str = ""):
    from push_notify import register_device

    return register_device(token, platform, name)


@app.post("/mobile/unregister")
async def mobile_unregister(token: str):
    from push_notify import unregister_device

    unregister_device(token)
    return {"status": "unregistered"}


@app.get("/mobile/devices")
async def mobile_devices():
    from push_notify import get_devices

    return {"devices": get_devices()}


@app.post("/mobile/send")
async def mobile_send(message: str, priority: int = 3):
    from push_notify import enqueue_message

    enqueue_message(message, priority)
    return {"status": "queued"}


@app.get("/mobile/outbox")
async def mobile_outbox():
    from push_notify import get_pending_messages

    return {"pending": get_pending_messages()}


@app.get("/plugins")
async def plugins_endpoint():
    from plugin_manager import get_loaded_plugins, list_available_plugins

    return {
        "available": list_available_plugins(),
        "loaded": get_loaded_plugins(),
    }


@app.get("/memories")
async def memories_endpoint():
    from memory import get_all_memories

    memories = get_all_memories()
    return {"memories": [{"type": t, "content": c, "date": d} for t, c, d in memories]}


@app.get("/rag/stats")
async def rag_stats():
    from rag_memory import get_rag_stats

    return get_rag_stats()


@app.post("/rag/index")
async def rag_index(folder: str = None):
    from rag_memory import index_folder

    path = folder or os.getenv("RAG_FOLDER", "~/Documents")
    loop = asyncio.get_event_loop()
    files, chunks = await loop.run_in_executor(None, index_folder, path)
    return {"status": "indexed", "files": files, "chunks": chunks}


@app.get("/rag/search")
async def rag_search(q: str = "", n: int = 5):
    from rag_memory import search_rag_structured

    if not q:
        return {"error": "Provide ?q=query", "results": []}
    result = search_rag_structured(q, n)
    return result


@app.post("/rag/prune")
async def rag_prune(days: int = 90):
    from rag_memory import prune_stale_entries

    deleted = prune_stale_entries(days)
    return {"deleted": deleted, "days": days}


@app.get("/triggers")
async def triggers_list(enabled: bool = False):
    from trigger_engine import list_triggers

    return list_triggers(enabled_only=enabled)


@app.get("/triggers/{trigger_id}")
async def triggers_get(trigger_id: int):
    from trigger_engine import get_trigger

    t = get_trigger(trigger_id)
    if not t:
        raise HTTPException(404, "Trigger not found")
    return t


@app.post("/triggers")
async def triggers_create(
    name: str,
    trigger_type: str,
    schedule: str,
    action_type: str,
    action_target: str,
    action_params: str = "{}",
    description: str = "",
):
    from trigger_engine import create_trigger

    try:
        params = json.loads(action_params)
    except json.JSONDecodeError:
        params = {}
    try:
        t = create_trigger(name, trigger_type, schedule, action_type, action_target, params, description)
        return t
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/triggers/{trigger_id}")
async def triggers_update(
    trigger_id: int,
    enabled: bool | None = None,
    name: str | None = None,
    description: str | None = None,
    schedule: str | None = None,
    trigger_type: str | None = None,
    action_type: str | None = None,
    action_target: str | None = None,
    action_params: str | None = None,
):
    from trigger_engine import update_trigger

    kwargs = {}
    if enabled is not None:
        kwargs["enabled"] = int(enabled)
    if name is not None:
        kwargs["name"] = name
    if description is not None:
        kwargs["description"] = description
    if schedule is not None:
        kwargs["schedule"] = schedule
    if trigger_type is not None:
        kwargs["trigger_type"] = trigger_type
    if action_type is not None:
        kwargs["action_type"] = action_type
    if action_target is not None:
        kwargs["action_target"] = action_target
    if action_params is not None:
        try:
            kwargs["action_params"] = json.dumps(json.loads(action_params))
        except json.JSONDecodeError:
            pass
    t = update_trigger(trigger_id, **kwargs)
    if not t:
        raise HTTPException(404, "Trigger not found")
    return t


@app.delete("/triggers/{trigger_id}")
async def triggers_delete(trigger_id: int):
    from trigger_engine import delete_trigger

    if not delete_trigger(trigger_id):
        raise HTTPException(404, "Trigger not found")
    return {"ok": True}


@app.get("/triggers/{trigger_id}/history")
async def triggers_history(trigger_id: int, limit: int = 50):
    from trigger_engine import get_trigger_history

    return get_trigger_history(trigger_id, limit)


@app.post("/triggers/{trigger_id}/fire")
async def triggers_fire(trigger_id: int):
    from trigger_engine import fire_trigger

    try:
        result = fire_trigger(trigger_id)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/events/fire")
async def events_fire(event_name: str):
    from trigger_engine import fire_event

    results = fire_event(event_name)
    return {"event": event_name, "triggers_fired": len(results), "results": results}


@app.get("/triggers/history/all")
async def triggers_history_all(limit: int = 50):
    from trigger_engine import get_trigger_history

    return get_trigger_history(limit=limit)


@app.get("/memory/stats")
async def memory_stats():
    from associative_memory import get_association_stats
    from memory import get_all_memories
    from procedural_memory import list_procedures
    from vector_memory import get_vector_memory_stats

    vec_stats = get_vector_memory_stats()
    assoc_stats = get_association_stats()
    try:
        from rag_memory import get_rag_stats

        rag_chunks = get_rag_stats().get("total_chunks", 0)
    except Exception:
        rag_chunks = 0
    try:
        from graph_memory import get_all_entities

        graph_entities = len(get_all_entities())
    except Exception:
        graph_entities = 0

    procs = list_procedures()
    proc_count = len(procs.splitlines()) if procs else 0
    return {
        "explicit_memories": len(get_all_memories()),
        "vector_entries": vec_stats.get("total_entries", 0),
        "rag_chunks": rag_chunks,
        "graph_entities": graph_entities,
        "procedures": proc_count,
        "associations": assoc_stats.get("total_pairs", 0),
    }


def _get_tailscale_ip() -> str | None:
    """Return this machine's Tailscale IPv4, if Tailscale is running."""
    import subprocess

    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            ip = (out.stdout or "").strip().splitlines()[0].strip()
            if ip and ip[0].isdigit():
                return ip
    except Exception:
        pass
    return None


def _get_lan_ip() -> str:
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────
# REMOTE ACCESS & AUTH
# ─────────────────────────────────────────────


def _require_api_key(x_api_key: str = Header(None)):
    """Dependency: reject if API key is configured but not provided correctly."""
    if JARVIS_API_KEY and x_api_key != JARVIS_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return True


REMOTE_WHITELIST = {
    "open_app": {"confirm": False},
    "quit_app": {"confirm": True},
    "system_status": {"confirm": False},
    "weather": {"confirm": False},
    "screen_check": {"confirm": False},
    "send_imessage": {"confirm": True},
    "gmail_search": {"confirm": False},
    "gmail_get_labels": {"confirm": False},
    "gmail_get_message": {"confirm": False},
    "github_search_code": {"confirm": False},
    "github_list_repos": {"confirm": False},
    "github_get_repo": {"confirm": False},
    "github_list_issues": {"confirm": False},
    "github_create_issue": {"confirm": True},
    "reminders_list": {"confirm": False},
    "reminders_get_lists": {"confirm": False},
    "reminders_search": {"confirm": False},
    "reminders_create": {"confirm": True},
    "reminders_complete": {"confirm": True},
    "reminders_delete": {"confirm": True},
    # Drive
    "gdrive_list": {"confirm": False},
    "gdrive_search": {"confirm": False},
    "gdrive_get": {"confirm": False},
    "gdrive_download": {"confirm": False},
    "gdrive_upload": {"confirm": True},
    "gdrive_create_folder": {"confirm": True},
    "gdrive_share": {"confirm": True},
    "gdrive_move": {"confirm": True},
    "gdrive_delete": {"confirm": True},
    # Sheets
    "gsheets_get": {"confirm": False},
    "gsheets_read_range": {"confirm": False},
    "gsheets_read_sheet": {"confirm": False},
    "gsheets_append": {"confirm": True},
    "gsheets_update_range": {"confirm": True},
    "gsheets_batch_update": {"confirm": True},
    "gsheets_create": {"confirm": True},
    "gsheets_add_sheet": {"confirm": True},
}


@app.post("/remote/command", dependencies=[Depends(_require_api_key)])
async def remote_command(req: TextRequest):
    """Execute a whitelisted remote command. Text should be JSON: {"command": "...", "params": {...}}"""
    try:
        data = json.loads(req.text)
    except Exception:
        raise HTTPException(400, "Expected JSON body with 'command' and 'params'")
    command = data.get("command", "")
    params = data.get("params", {})
    if not command or command not in REMOTE_WHITELIST:
        raise HTTPException(400, f"Unknown command. Allowed: {', '.join(REMOTE_WHITELIST)}")
    entry = REMOTE_WHITELIST[command]
    if entry["confirm"]:
        return {"status": "confirm", "command": command, "message": f"Confirm {command.replace('_', ' ')}?"}

    # Map command to tool
    import pyautogui

    from tools.communication_tools import send_imessage
    from tools.github_tools import (
        github_create_issue,
        github_get_repo,
        github_list_issues,
        github_list_repos,
        github_search_code,
    )
    from tools.gmail_tools import gmail_get_labels, gmail_get_message, gmail_search
    from tools.google_docs_tools import (
        docs_append_text,
        docs_create,
        docs_get,
        docs_search,
    )
    from tools.google_drive_tools import (
        gdrive_create_folder,
        gdrive_delete,
        gdrive_download,
        gdrive_get,
        gdrive_list,
        gdrive_move,
        gdrive_search,
        gdrive_share,
        gdrive_upload,
    )
    from tools.google_forms_tools import (
        forms_add_question,
        forms_create,
        forms_get,
        forms_get_responses,
    )
    from tools.google_sheets_tools import (
        gsheets_add_sheet,
        gsheets_append,
        gsheets_batch_update,
        gsheets_create,
        gsheets_get,
        gsheets_read_range,
        gsheets_read_sheet,
        gsheets_update_range,
    )
    from tools.google_slides_tools import (
        slides_add_slide,
        slides_create,
        slides_get,
        slides_replace_text,
        slides_search,
    )
    from tools.reminders_tools import (
        reminders_complete,
        reminders_create,
        reminders_delete,
        reminders_get_lists,
        reminders_list,
        reminders_search,
    )
    from tools.system_tools import get_system_info, get_weather_detailed, open_app, quit_app
    from tools.vision_tools import analyze_image

    tool_map = {
        "open_app": lambda: open_app(params.get("app_name", "")),
        "quit_app": lambda: quit_app(params.get("app_name", "")),
        "system_status": lambda: get_system_info(),
        "weather": lambda: get_weather_detailed(),
        "send_imessage": lambda: send_imessage(params.get("contact", ""), params.get("message", "")),
        "gmail_search": lambda: gmail_search(params.get("query", ""), params.get("max_results", 20)),
        "gmail_get_labels": lambda: gmail_get_labels(),
        "gmail_get_message": lambda: gmail_get_message(params.get("message_id", "")),
        "github_search_code": lambda: github_search_code(params.get("query", ""), params.get("per_page", 10)),
        "github_list_repos": lambda: github_list_repos(params.get("visibility", "all"), params.get("affiliation", "owner"), params.get("per_page", 30)),
        "github_get_repo": lambda: github_get_repo(params.get("owner", ""), params.get("repo", "")),
        "github_list_issues": lambda: github_list_issues(params.get("owner", ""), params.get("repo", ""), params.get("state", "open"), params.get("labels", ""), params.get("per_page", 20)),
        "github_create_issue": lambda: github_create_issue(params.get("owner", ""), params.get("repo", ""), params.get("title", ""), params.get("body", ""), params.get("labels", [])),
        "gdrive_list": lambda: gdrive_list(params.get("folder_id", "root"), params.get("page_size", 50)),
        "gdrive_search": lambda: gdrive_search(params.get("query", ""), params.get("page_size", 50)),
        "gdrive_get": lambda: gdrive_get(params.get("file_id", "")),
        "gdrive_download": lambda: gdrive_download(params.get("file_id", ""), params.get("dest_path", "")),
        "gdrive_upload": lambda: gdrive_upload(params.get("local_path", ""), params.get("folder_id", "root"), params.get("name", "")),
        "gdrive_create_folder": lambda: gdrive_create_folder(params.get("name", ""), params.get("parent_id", "root")),
        "gdrive_share": lambda: gdrive_share(params.get("file_id", ""), params.get("email", ""), params.get("role", "reader")),
        "gdrive_move": lambda: gdrive_move(params.get("file_id", ""), params.get("new_parent_id", "")),
        "gdrive_delete": lambda: gdrive_delete(params.get("file_id", "")),
        "gsheets_get": lambda: gsheets_get(params.get("spreadsheet_id", "")),
        "gsheets_read_range": lambda: gsheets_read_range(params.get("spreadsheet_id", ""), params.get("range_name", "")),
        "gsheets_read_sheet": lambda: gsheets_read_sheet(params.get("spreadsheet_id", ""), params.get("sheet_name", "")),
        "gsheets_append": lambda: gsheets_append(params.get("spreadsheet_id", ""), params.get("range_name", ""), params.get("values", [])),
        "gsheets_update_range": lambda: gsheets_update_range(params.get("spreadsheet_id", ""), params.get("range_name", ""), params.get("values", [])),
        "gsheets_batch_update": lambda: gsheets_batch_update(params.get("spreadsheet_id", ""), params.get("requests_list", [])),
        "gsheets_create": lambda: gsheets_create(params.get("title", "")),
        "gsheets_add_sheet": lambda: gsheets_add_sheet(params.get("spreadsheet_id", ""), params.get("title", ""), params.get("rows", 1000), params.get("cols", 26)),
        "reminders_list": lambda: reminders_list(params.get("list_name"), params.get("completed", False)),
        "reminders_get_lists": lambda: reminders_get_lists(),
        "reminders_search": lambda: reminders_search(params.get("query", "")),
        "reminders_create": lambda: reminders_create(params.get("title", ""), params.get("notes", ""), params.get("due_date"), params.get("list_name")),
        "reminders_complete": lambda: reminders_complete(params.get("reminder_id", "")),
        "reminders_delete": lambda: reminders_delete(params.get("reminder_id", "")),
        # Docs
        "docs_get": lambda: docs_get(params.get("document_id", "")),
        "docs_create": lambda: docs_create(params.get("title", "")),
        "docs_append_text": lambda: docs_append_text(params.get("document_id", ""), params.get("text", "")),
        "docs_search": lambda: docs_search(params.get("query", "")),
        # Slides
        "slides_get": lambda: slides_get(params.get("presentation_id", "")),
        "slides_create": lambda: slides_create(params.get("title", "")),
        "slides_add_slide": lambda: slides_add_slide(params.get("presentation_id", "")),
        "slides_replace_text": lambda: slides_replace_text(params.get("presentation_id", ""), params.get("old_text", ""), params.get("new_text", "")),
        "slides_search": lambda: slides_search(params.get("query", "")),
        # Forms
        "forms_get": lambda: forms_get(params.get("form_id", "")),
        "forms_create": lambda: forms_create(params.get("title", "")),
        "forms_add_question": lambda: forms_add_question(params.get("form_id", ""), params.get("question_text", ""), params.get("question_type", "text")),
        "forms_get_responses": lambda: forms_get_responses(params.get("form_id", "")),
    }
    if command == "screen_check":
        path = os.path.join(os.path.expanduser("~"), "Desktop", f"remote_screen_{int(time.time())}.png")
        pyautogui.screenshot().save(path)
        result = analyze_image(path=path, question="Describe what's on this screen.")
        return {"status": "ok", "command": command, "result": result}

    result = tool_map[command]()
    return {"status": "ok", "command": command, "result": str(result)}


@app.post("/remote/confirm", dependencies=[Depends(_require_api_key)])
async def remote_confirm(req: TextRequest):
    """Confirm a previously requested sensitive command. Text should be JSON: {"command": "...", "params": {...}}"""
    try:
        data = json.loads(req.text)
    except Exception:
        raise HTTPException(400, "Expected JSON body with 'command' and 'params'")
    command = data.get("command", "")
    params = data.get("params", {})
    from tools.communication_tools import send_imessage
    from tools.github_tools import github_create_issue
    from tools.google_drive_tools import gdrive_create_folder, gdrive_delete, gdrive_move, gdrive_share, gdrive_upload
    from tools.google_sheets_tools import (
        gsheets_add_sheet,
        gsheets_append,
        gsheets_batch_update,
        gsheets_create,
        gsheets_update_range,
    )
    from tools.reminders_tools import reminders_complete, reminders_create, reminders_delete
    from tools.system_tools import quit_app

    tool_map = {
        "quit_app": lambda: quit_app(params.get("app_name", "")),
        "send_imessage": lambda: send_imessage(params.get("contact", ""), params.get("message", "")),
        "github_create_issue": lambda: github_create_issue(params.get("owner", ""), params.get("repo", ""), params.get("title", ""), params.get("body", ""), params.get("labels", [])),
        "gdrive_upload": lambda: gdrive_upload(params.get("local_path", ""), params.get("folder_id", "root"), params.get("name", "")),
        "gdrive_create_folder": lambda: gdrive_create_folder(params.get("name", ""), params.get("parent_id", "root")),
        "gdrive_share": lambda: gdrive_share(params.get("file_id", ""), params.get("email", ""), params.get("role", "reader")),
        "gdrive_move": lambda: gdrive_move(params.get("file_id", ""), params.get("new_parent_id", "")),
        "gdrive_delete": lambda: gdrive_delete(params.get("file_id", "")),
        "gsheets_append": lambda: gsheets_append(params.get("spreadsheet_id", ""), params.get("range_name", ""), params.get("values", [])),
        "gsheets_update_range": lambda: gsheets_update_range(params.get("spreadsheet_id", ""), params.get("range_name", ""), params.get("values", [])),
        "gsheets_batch_update": lambda: gsheets_batch_update(params.get("spreadsheet_id", ""), params.get("requests_list", [])),
        "gsheets_create": lambda: gsheets_create(params.get("title", "")),
        "gsheets_add_sheet": lambda: gsheets_add_sheet(params.get("spreadsheet_id", ""), params.get("title", ""), params.get("rows", 1000), params.get("cols", 26)),
        "reminders_create": lambda: reminders_create(params.get("title", ""), params.get("notes", ""), params.get("due_date"), params.get("list_name")),
        "reminders_complete": lambda: reminders_complete(params.get("reminder_id", "")),
        "reminders_delete": lambda: reminders_delete(params.get("reminder_id", "")),
    }
    fn = tool_map.get(command)
    if not fn:
        raise HTTPException(400, f"Cannot confirm: unknown command '{command}'")
    result = fn()
    return {"status": "ok", "command": command, "result": str(result)}


@app.get("/remote/whitelist", dependencies=[Depends(_require_api_key)])
async def remote_whitelist():
    return {"commands": list(REMOTE_WHITELIST.keys())}


# ─────────────────────────────────────────────
# OAUTH INTEGRATIONS
# ─────────────────────────────────────────────


@app.get("/oauth/authorize/{provider}")
async def oauth_authorize(provider: str, request: Request):
    """Redirect to OAuth provider authorization URL with correct redirect_uri."""
    base = str(request.base_url).rstrip("/")

    callback_paths = {
        "gmail": "gmail",
        "github": "github",
        "google-drive": "google-drive",
        "google-sheets": "google-sheets",
        "google-docs": "google-docs",
        "google-slides": "google-slides",
        "google-forms": "google-forms",
        "google_drive": "google-drive",
        "google_sheets": "google-sheets",
        "google_docs": "google-docs",
        "google_slides": "google-slides",
        "google_forms": "google-forms",
    }
    cb_path = callback_paths.get(provider)
    if not cb_path:
        raise HTTPException(400, f"Unknown provider: {provider}")

    redirect_uri = f"{base}/oauth/callback/{cb_path}"
    url_map = {
        "gmail": gmail_auth_url(redirect_uri=redirect_uri),
        "github": github_auth_url(redirect_uri=redirect_uri),
        "google-drive": gdrive_auth_url(redirect_uri=redirect_uri),
        "google-sheets": gsheets_auth_url(redirect_uri=redirect_uri),
        "google-docs": docs_auth_url(redirect_uri=redirect_uri),
        "google-slides": slides_auth_url(redirect_uri=redirect_uri),
        "google-forms": forms_auth_url(redirect_uri=redirect_uri),
        "google_drive": gdrive_auth_url(redirect_uri=redirect_uri),
        "google_sheets": gsheets_auth_url(redirect_uri=redirect_uri),
        "google_docs": docs_auth_url(redirect_uri=redirect_uri),
        "google_slides": slides_auth_url(redirect_uri=redirect_uri),
        "google_forms": forms_auth_url(redirect_uri=redirect_uri),
    }
    auth_url = url_map.get(provider)
    return RedirectResponse(auth_url)


@app.get("/oauth/callback/{provider}")
async def oauth_callback(provider: str, code: str = Query(...), state: str = Query(...)):
    """Handle OAuth callback from provider."""
    callback_map = {
        "gmail": gmail_handle_callback,
        "github": github_handle_callback,
        "google-drive": gdrive_handle_callback,
        "google-sheets": gsheets_handle_callback,
        "google-docs": docs_handle_callback,
        "google-slides": slides_handle_callback,
        "google-forms": forms_handle_callback,
        "google_drive": gdrive_handle_callback,
        "google_sheets": gsheets_handle_callback,
        "google_docs": docs_handle_callback,
        "google_slides": slides_handle_callback,
        "google_forms": forms_handle_callback,
    }
    handler = callback_map.get(provider)
    if not handler:
        raise HTTPException(400, f"Unknown provider: {provider}")
    result = handler(code, state)
    return HTMLResponse(f"""
    <html><body>
    <script>
        window.opener.postMessage({{type: 'oauth_result', provider: '{provider}', result: {json.dumps(result)}}}, '*');
        window.close();
    </script>
    <p>{result}</p>
    </body></html>
    """)


@app.post("/oauth/disconnect/{provider}")
async def oauth_disconnect(provider: str):
    """Disconnect an OAuth integration."""
    provider_map = {
        "gmail": "gmail",
        "github": "github",
        "google-drive": "google_drive",
        "google-sheets": "google_sheets",
        "google_drive": "google_drive",
        "google_sheets": "google_sheets",
        "google-docs": "google_docs",
        "google-slides": "google_slides",
        "google-forms": "google_forms",
        "google_docs": "google_docs",
        "google_slides": "google_slides",
        "google_forms": "google_forms",
    }
    mapped = provider_map.get(provider)
    if not mapped:
        raise HTTPException(400, f"Unknown provider: {provider}")
    success = TokenStore.delete(mapped)
    return {"status": "ok", "provider": provider, "disconnected": success}


@app.get("/oauth/status")
async def oauth_status():
    """Get OAuth connection status for all providers."""
    return {"providers": TokenStore.get_all_status()}


# ─────────────────────────────────────────────
# iOS SHORTCUTS (webhook endpoints)
# ─────────────────────────────────────────────


@app.post("/ios/message", dependencies=[Depends(_require_api_key)])
async def ios_send_message(req: TextRequest):
    """iOS Shortcut: send iMessage. Text should be JSON: {"contact": "...", "message": "..."}"""
    try:
        data = json.loads(req.text)
        contact = data.get("contact", "")
        message = data.get("message", "")
    except Exception:
        raise HTTPException(400, "Expected JSON with 'contact' and 'message'")
    if not contact or not message:
        raise HTTPException(400, "Both 'contact' and 'message' are required")
    from tools.communication_tools import send_imessage

    result = send_imessage(contact, message)
    return {"status": "ok", "result": result}


@app.post("/ios/openapp", dependencies=[Depends(_require_api_key)])
async def ios_open_app(req: TextRequest):
    """iOS Shortcut: open an app on Mac. Text should be JSON: {"app_name": "..."}"""
    try:
        data = json.loads(req.text)
        app_name = data.get("app_name", "")
    except Exception:
        raise HTTPException(400, "Expected JSON with 'app_name'")
    if not app_name:
        raise HTTPException(400, "'app_name' is required")
    from tools.system_tools import open_app

    result = open_app(app_name)
    return {"status": "ok", "result": result}


# ─────────────────────────────────────────────
# FACE UNLOCK (DeepFace + ArcFace)
# ─────────────────────────────────────────────


def _check_face_enrolled() -> bool:
    """Check if a face embedding exists in macOS Keychain."""
    global _face_enrolled
    if _face_enrolled is not None:
        return _face_enrolled
    try:
        import keyring

        stored = keyring.get_password("jarvis", "face_embedding")
        _face_enrolled = bool(stored)
    except Exception:
        _face_enrolled = False
    return _face_enrolled


def _get_face_embedding(image_path: str) -> np.ndarray | None:
    """Extract face embedding from image using DeepFace + ArcFace."""
    try:
        from deepface import DeepFace

        result = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            enforce_detection=False,
        )
        if isinstance(result, list) and len(result) > 0 and "embedding" in result[0]:
            return np.array(result[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"  [Face] Embedding extraction error: {e}")
    return None


def _store_face_embedding(embedding: np.ndarray):
    """Store face embedding in macOS Keychain."""
    import keyring

    keyring.set_password("jarvis", "face_embedding", embedding.tobytes().hex())


def _load_face_embedding() -> np.ndarray | None:
    """Load face embedding from macOS Keychain."""
    try:
        import keyring

        stored = keyring.get_password("jarvis", "face_embedding")
        if stored:
            return np.frombuffer(bytes.fromhex(stored), dtype=np.float32)
    except Exception:
        pass
    return None


def _compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    if emb1.shape != emb2.shape or np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
        return 0.0
    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    return max(0.0, min(1.0, similarity))


@app.post("/enroll/face", dependencies=[Depends(_require_api_key)])
async def enroll_face(photos: list[UploadFile] = File(...)):
    """Enroll a face by uploading 1-10 photos. Extracts ArcFace embedding and stores in Keychain."""
    if not photos:
        raise HTTPException(400, "No photos uploaded. Send at least 1 photo file.")
    if len(photos) > 10:
        raise HTTPException(400, "Maximum 10 photos per enrollment.")

    embeddings = []
    errors = []
    for photo in photos:
        suffix = ".jpg"
        if photo.filename:
            ext = os.path.splitext(photo.filename)[-1]
            if ext:
                suffix = ext
        try:
            contents = await photo.read()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(contents)
                tmp_path = f.name
            emb = _get_face_embedding(tmp_path)
            if emb is not None:
                embeddings.append(emb)
            else:
                errors.append(f"No face detected in {photo.filename}")
            os.remove(tmp_path)
        except Exception as e:
            errors.append(f"Error processing {photo.filename}: {e}")

    if not embeddings:
        raise HTTPException(400, f"No face detected in any photo. Errors: {'; '.join(errors)}")

    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
    _store_face_embedding(avg_embedding)
    global _face_enrolled
    _face_enrolled = True

    return {
        "status": "ok",
        "enrolled": True,
        "photos_processed": len(embeddings),
        "photos_total": len(photos),
        "confidence_hint": "You can now use /unlock/face to verify.",
        "errors": errors if errors else None,
    }


@app.post("/unlock/face", dependencies=[Depends(_require_api_key)])
async def unlock_face(photo: UploadFile = File(None)):
    """Verify face: upload a selfie or let Mac capture from webcam. If match > threshold, returns authenticated."""
    if not _check_face_enrolled():
        raise HTTPException(400, "No face enrolled. POST to /enroll/face first.")

    stored_embedding = _load_face_embedding()
    if stored_embedding is None:
        raise HTTPException(500, "Failed to load stored embedding from Keychain.")

    if photo:
        contents = await photo.read()
        suffix = ".jpg"
        if photo.filename:
            ext = os.path.splitext(photo.filename)[-1]
            if ext:
                suffix = ext
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(contents)
            tmp_path = f.name
    else:
        # Capture from Mac webcam
        tmp_path = os.path.join(tempfile.gettempdir(), "jarvis_unlock.jpg")
        try:
            subprocess.run(
                ["imagesnap", "-q", tmp_path],
                capture_output=True,
                timeout=15,
            )
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1000:
                raise Exception("imagesnap failed or returned empty image")
        except Exception:
            try:
                subprocess.run(
                    ["ffmpeg", "-f", "avfoundation", "-i", "0", "-vframes", "1", tmp_path],
                    capture_output=True,
                    timeout=15,
                )
            except Exception as e:
                raise HTTPException(500, f"Could not capture from webcam: {e}")

    try:
        embedding = _get_face_embedding(tmp_path)
    finally:
        if not photo:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if embedding is None:
        return {
            "status": "denied",
            "authenticated": False,
            "confidence": 0.0,
            "reason": "No face detected in capture.",
            # Docs
            "docs_get": {"confirm": False},
            "docs_create": {"confirm": True},
            "docs_append_text": {"confirm": True},
            "docs_search": {"confirm": False},
            # Slides
            "slides_get": {"confirm": False},
            "slides_create": {"confirm": True},
            "slides_add_slide": {"confirm": True},
            "slides_replace_text": {"confirm": True},
            "slides_search": {"confirm": False},
            # Forms
            "forms_get": {"confirm": False},
            "forms_create": {"confirm": True},
            "forms_add_question": {"confirm": True},
            "forms_get_responses": {"confirm": False},
        }

    similarity = _compare_embeddings(embedding, stored_embedding)
    authenticated = similarity >= FACE_SIMILARITY_THRESHOLD

    if authenticated:
        try:
            # Notify via spoken confirmation on Mac
            speak(f"Face recognized. Confidence {similarity:.0%}.")
        except Exception:
            pass

    return {
        "status": "authenticated" if authenticated else "denied",
        "authenticated": authenticated,
        "confidence": round(similarity, 4),
        "threshold": FACE_SIMILARITY_THRESHOLD,
    }


@app.get("/face/status")
async def face_status():
    """Check whether a face embedding is enrolled."""
    enrolled = _check_face_enrolled()
    return {
        "enrolled": enrolled,
        "threshold": FACE_SIMILARITY_THRESHOLD,
        "model": "ArcFace",
    }


# ─────────────────────────────────────────────
# SCREEN VISION (Always-On Capture)
# ─────────────────────────────────────────────


def _init_screen_dir():
    os.makedirs(_screenshot_dir, exist_ok=True)
    for f in glob.glob(os.path.join(_screenshot_dir, "*.png")):
        try:
            os.remove(f)
        except Exception:
            pass


def _capture_screen_loop():
    """Background thread: capture screen every SCREEN_CAPTURE_INTERVAL seconds, keep rolling buffer."""
    _init_screen_dir()
    while True:
        try:
            ts = int(time.time())
            path = os.path.join(_screenshot_dir, f"screen_{ts}.png")
            subprocess.run(
                ["screencapture", "-x", path],
                capture_output=True,
                timeout=15,
            )
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                with _screen_lock:
                    _screen_buffer.append((ts, path))
                    if len(_screen_buffer) > SCREEN_BUFFER_SIZE:
                        old_ts, old_path = _screen_buffer.pop(0)
                        try:
                            os.remove(old_path)
                        except Exception:
                            pass
        except Exception:
            pass
        time.sleep(SCREEN_CAPTURE_INTERVAL)


@app.get("/screen/latest")
async def screen_latest():
    """Return the latest captured screenshot image."""
    with _screen_lock:
        if not _screen_buffer:
            # Capture one on demand
            try:
                path = os.path.join(_screenshot_dir, f"ondemand_{int(time.time())}.png")
                subprocess.run(["screencapture", "-x", path], capture_output=True, timeout=15)
                if os.path.exists(path):
                    return FileResponse(path, media_type="image/png")
            except Exception:
                pass
            raise HTTPException(404, "No screenshots available yet.")
        ts, path = _screen_buffer[-1]
        return FileResponse(path, media_type="image/png")


@app.post("/screen/analyze", dependencies=[Depends(_require_api_key)])
async def screen_analyze(req: TextRequest):
    """Analyze the latest screenshot using NVIDIA VL vision model."""
    with _screen_lock:
        if _screen_buffer:
            ts, screen_path = _screen_buffer[-1]
        else:
            screen_path = None

    if not screen_path or not os.path.exists(screen_path):
        try:
            screen_path = os.path.join(_screenshot_dir, f"ondemand_{int(time.time())}.png")
            subprocess.run(["screencapture", "-x", screen_path], capture_output=True, timeout=15)
        except Exception:
            raise HTTPException(500, "Could not capture screenshot")

    question = req.text.strip() or "Describe what's on this screen."
    try:
        from tools.vision_tools import analyze_image

        result = analyze_image(path=screen_path, question=question)
        return {"status": "ok", "analysis": result, "screenshot": os.path.basename(screen_path)}
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


@app.get("/screen/status")
async def screen_status():
    """Get screen capture status: interval, buffer count, last capture time."""
    with _screen_lock:
        count = len(_screen_buffer)
        last_ts = _screen_buffer[-1][0] if _screen_buffer else None
    return {
        "capture_interval_seconds": SCREEN_CAPTURE_INTERVAL,
        "buffer_size": SCREEN_BUFFER_SIZE,
        "current_buffer": count,
        "last_capture_unix_ts": last_ts,
        "screenshot_dir": _screenshot_dir,
    }


# ─────────────────────────────────────────────
# API KEY MANAGEMENT
# ─────────────────────────────────────────────


@app.get("/api/key", dependencies=[Depends(_require_api_key)])
async def get_api_key():
    """Return the API key (for phone setup)."""
    return {"api_key": JARVIS_API_KEY, "note": "Store this in the phone UI under Remote Control → Settings."}


# ── Startup ──────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("JARVIS_PORT", "8000"))
    lan_ip = _get_lan_ip()
    ts_ip = _get_tailscale_ip()

    print("\n" + "=" * 50)
    print("  J.A.R.V.I.S. Server starting...")
    print("=" * 50)
    print(f"\n  On this Mac:     http://localhost:{port}")
    print(f"  Same Wi‑Fi only: http://{lan_ip}:{port}")

    if ts_ip:
        print(f"\n  ★ Phone (Tailscale): http://{ts_ip}:{port}")
        print("    Install Tailscale on phone, same account, then open that URL.")
    else:
        print("\n  Phone (Tailscale): not detected.")
        print("    Install Tailscale on Mac + phone, then run: tailscale ip -4")
        print(f"    Or on same Wi‑Fi use: http://{lan_ip}:{port}")

    # Start screen capture background thread
    _capture_thread = threading.Thread(target=_capture_screen_loop, daemon=True)
    _capture_thread.start()
    print("  Screen capture: every 10s, buffer 5 frames.")

    face_status = "enrolled" if _check_face_enrolled() else "not enrolled"
    print(f"  Face unlock: {face_status} (ArcFace)")

    print("\n  Full guide: MOBILE.md")
    print("  TTS plays on this Mac's speakers.\n")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
