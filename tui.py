"""JARVIS Control Console — opencode-style Textual TUI.

Layout:
    header (session, status, mic) | footer (bindings)
    left: TabbedContent [MAIN | SYSTEM | BRAIN | MEMORY | VISION | WORKFLOWS | TOOLS]
      MAIN = transcript (session-backed) + multi-line composer
    right: Activity rail (tool calls, subagent progress, safety, latency)

Env:
    JARVIS_TUI=0                 force classic REPL (TUI is the default when stdin/stdout are TTYs)
    JARVIS_TUI_ANNOUNCE=1        speak panel-init flavor lines
    JARVIS_TUI_VISION=1          auto-start the camera when VISION opens
"""

from __future__ import annotations

import ast
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Footer, RichLog, Static, TabbedContent, TabPane, TextArea

import terminal

_ANNOUNCE = os.environ.get("JARVIS_TUI_ANNOUNCE", "0") == "1"
_AUTOSTART_VISION = os.environ.get("JARVIS_TUI_VISION", "1") == "1"

_RAMP = " .:-=+*#%@"


# ──────────────────────────────────────────────────────────
# STDOUT SINK — captures brain/agent prints into the rail
# ──────────────────────────────────────────────────────────
class StdoutSink:
    """Thread-safe file-like object that routes print() output to the Activity rail."""

    def __init__(self, on_line):
        self._on_line = on_line
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        if not data:
            return 0
        with self._lock:
            self._buf += str(data)
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                self._on_line(line)
        return len(data)

    def flush(self):
        with self._lock:
            if self._buf:
                self._on_line(self._buf)
                self._buf = ""

    def isatty(self) -> bool:
        return False


# ──────────────────────────────────────────────────────────
# COMPOSER
# ──────────────────────────────────────────────────────────
class Composer(TextArea):
    class Submit(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    BINDINGS = [
        Binding("ctrl+enter", "newline", "Newline"),
        Binding("ctrl+j", "newline", "Newline"),
        Binding("ctrl+up", "history_prev", "History"),
        Binding("ctrl+down", "history_next", "History"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._hist_idx = -1

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.action_submit()
            return
        await super()._on_key(event)

    def action_submit(self) -> None:
        text = self.text.strip()
        if text:
            self._history.append(text)
            self._hist_idx = -1
            self.post_message(self.Submit(text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_history_prev(self) -> None:
        if not self._history:
            return
        self._hist_idx = max(self._hist_idx - 1, -1) if self._hist_idx >= 0 else len(self._history) - 1
        self.text = self._history[self._hist_idx]

    def action_history_next(self) -> None:
        if self._hist_idx == -1 or not self._history:
            return
        self._hist_idx += 1
        if self._hist_idx >= len(self._history):
            self._hist_idx = -1
            self.text = ""
        else:
            self.text = self._history[self._hist_idx]


# ──────────────────────────────────────────────────────────
# ARTIFICIAL RETINA (read-only reference to the experiment)
# ──────────────────────────────────────────────────────────
def load_artificial_retina():
    """Load ArtificialRetina from jarvis_vision_experiment WITHOUT executing
    the module (it opens the camera and runs an infinite loop at import)."""
    path = Path(__file__).parent / "jarvis_vision_experiment" / "vision.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ArtificialRetina")
    mod = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(mod)
    import cv2
    import numpy as np

    ns = {"cv2": cv2, "np": np}
    exec(compile(mod, str(path), "exec"), ns)  # noqa: S102 — trusted local source
    return ns["ArtificialRetina"]


def _ascii_frame(gray, cols: int = 64, rows: int = 24) -> str:
    import cv2
    import numpy as np

    h, w = gray.shape
    small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
    small = np.clip(small.astype(np.float32), 0.0, 1.0)
    idx = (small * (len(_RAMP) - 1)).astype(np.intp)
    char_rows = ["".join(_RAMP[c] for c in row) for row in idx]
    return "\n".join(char_rows)


# ──────────────────────────────────────────────────────────
# VISION STREAM THREAD
# ──────────────────────────────────────────────────────────
class VisionStream(threading.Thread):
    """Camera → retina/camera ASCII frames at ~6 fps, only while active."""

    LEAK = 0.90
    THRESHOLD = 0.15

    def __init__(self, on_frame, on_status, active_check) -> None:
        super().__init__(daemon=True, name="jarvis-vision")
        self._on_frame = on_frame
        self._on_status = on_status
        self._active_check = active_check
        self._stop = threading.Event()
        self.retina_mode = False
        self.retina_cls = None
        self._retina = None
        self._membrane = None
        self._last_render = 0.0

    def stop(self) -> None:
        self._stop.set()

    def _emit_status(self, msg: str) -> None:
        try:
            self._on_status(msg)
        except Exception:
            pass

    def run(self) -> None:
        import cv2
        import numpy as np

        try:
            self.retina_cls = load_artificial_retina()
        except Exception as e:
            self._emit_status(f"RETINA OFFLINE — could not load ArtificialRetina: {e}")
            self.retina_cls = None

        camera = None
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError("could not open camera (index 0)")
        except Exception as e:
            self._emit_status(f"CAMERA OFFLINE — {e}")
            camera = None

        if camera is None:
            self._emit_status("Vision offline — no camera. Retina math idle.")
            return

        self._emit_status("Camera online.")

        while not self._stop.is_set():
            if not self._active_check():
                time.sleep(0.2)
                continue
            ok, frame = camera.read()
            if not ok:
                continue
            now = time.perf_counter()
            if now - self._last_render < 0.18:
                continue
            self._last_render = now

            if self.retina_mode and self.retina_cls is not None:
                try:
                    if self._retina is None:
                        self._retina = self.retina_cls(width=320, height=180)
                    changes = self._retina.process(frame)
                    if self._membrane is None:
                        self._membrane = np.zeros((180, 320), dtype=np.float32)
                    self._membrane *= self.LEAK
                    self._membrane += np.abs(changes)
                    spike_map = self._membrane >= self.THRESHOLD
                    spike_count = int(np.count_nonzero(spike_map))
                    self._membrane[spike_map] = 0.0
                    disp = np.clip(np.abs(changes) * 8.0, 0.0, 1.0)
                    frame_text = _ascii_frame(disp)
                    rate = spike_count / (320 * 180) * 100.0
                    self._on_frame(
                        frame_text,
                        f"RETINA 320x180 · 57,600 neurons · spikes {spike_count:,} ({rate:.1f}%) · "
                        f"threshold {self.THRESHOLD}",
                    )
                except Exception as e:  # noqa: BLE001
                    self._emit_status(f"Retina error: {e}")
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (64, 24), interpolation=cv2.INTER_AREA)
                norm = small.astype(np.float32) / 255.0
                frame_text = _ascii_frame(norm)
                self._on_frame(frame_text, "CAMERA VIEW — press R for artificial retina")

        if camera is not None:
            camera.release()


# ──────────────────────────────────────────────────────────
# VISION PANE
# ──────────────────────────────────────────────────────────
class VisionPane(Vertical):
    BINDINGS = [Binding("r", "toggle_retina", "Retina/Camera")]

    def compose(self) -> ComposeResult:
        yield Static("VISION — loading...", id="vision-status")
        yield Static("", id="vision-frame")
        yield Button("Toggle: Retina / Camera", id="vision-toggle", variant="primary")
        yield Static("R toggles retina ↔ camera · ctrl+v back to conversation")

    def action_toggle_retina(self) -> None:
        app = self.app
        if isinstance(app, JarvisConsole):
            app.toggle_vision_mode()


# ──────────────────────────────────────────────────────────
# CONSOLE APP
# ──────────────────────────────────────────────────────────
# CONSOLE APP
# ──────────────────────────────────────────────────────────
class JarvisConsole(App):
    TITLE = "J.A.R.V.I.S."
    CSS = """
    #header { height: 1; background: $panel; color: $text; padding: 0 1; }
    #header .brand { color: $accent; text-style: bold; }
    #header .badge { color: $success; text-style: bold; }
    #header .dim { color: $text-muted; }
    #transcript { border: round $panel; padding: 0 1; }
    #activity { border: round $panel; padding: 0 1; }
    #composer-wrap { height: auto; }
    Composer { height: 3; border: round $accent; }
    #vision-frame { height: 24; }
    #vision-status { color: $text-muted; }
    Static.pane-label { color: $text-muted; margin: 0 0 1 1; }
    #main-col { min-width: 60; }
    #rail-col { width: 44; }
    """

    BINDINGS = [
        Binding("ctrl+l", "focus_main", "Main"),
        Binding("ctrl+t", "focus_tools", "Tools"),
        Binding("ctrl+m", "focus_memory", "Memory"),
        Binding("ctrl+p", "focus_brain", "Brain"),
        Binding("ctrl+v", "focus_vision", "Vision"),
        Binding("ctrl+w", "focus_workflows", "Workflows"),
        Binding("ctrl+g", "focus_system", "System"),
        Binding("ctrl+r", "record", "Record"),
        Binding("ctrl+c", "cancel", "Cancel"),
        Binding("ctrl+q", "quit_app", "Quit"),
    ]

    def __init__(self, session_id: str = "default") -> None:
        super().__init__()
        self._session_id = session_id
        self._sink = None
        self._sink_queue: deque[str] = deque()
        self._busy = False
        self._pending: list[str] = []
        self._cancel = False
        self._recording = threading.Event()
        self._rail_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._vision: VisionStream | None = None
        self._vision_hint_shown = False
        self._last_session = session_id

    # ── live UI threading helpers ─────────────────────────
    def _rail(self, text: str, style: str = "dim") -> None:
        """Append a line to the Activity rail (safe from any thread)."""
        try:
            try:
                self.call_from_thread(self._rail_ui, text, style)
            except RuntimeError:
                self._rail_ui(text, style)
        except Exception:
            pass

    def _rail_ui(self, text: str, style: str = "dim") -> None:
        try:
            self.query_one("#activity", RichLog).write(Text(text, style=style))
        except Exception:
            pass

    def _transcript_line(self, role: str, text: str) -> None:
        try:
            try:
                self.call_from_thread(self._transcript_line_ui, role, text)
            except RuntimeError:
                self._transcript_line_ui(role, text)
        except Exception:
            pass

    def _transcript_line_ui(self, role: str, text: str) -> None:
        try:
            log = self.query_one("#transcript", RichLog)
            prefix = "YOU   »" if role == "user" else "JARVIS »"
            style = "bold cyan" if role == "user" else "green"
            log.write(Text(f"{prefix} {text}", style=style))
            log.scroll_end(animate=False)
        except Exception:
            pass

    # ── lifecycle ─────────────────────────────────────────
    def on_mount(self) -> None:
        import brain
        from event_bus import subscribe

        self._brain = brain
        subscribe("tool_call", self._ev_tool_call)
        subscribe("subagent_started", self._ev_subagent_started)
        subscribe("subagent_progress", self._ev_subagent_progress)
        subscribe("subagent_completed", self._ev_subagent_completed)
        subscribe("proactive_alert", self._ev_proactive_alert)
        subscribe("self_test", self._ev_self_test)

        # stdout sink → rail
        self.set_interval(0.4, self._sink_pump)
        self.set_interval(3.0, self._refresh_system_pane)
        self.set_interval(5.0, self._refresh_brain_pane)
        self.set_interval(5.0, self._refresh_memory_pane)
        self.set_interval(5.0, self._refresh_workflows_pane)
        self.set_interval(10.0, self._refresh_tools_pane)
        self.set_interval(1.0, self._refresh_header)

        self._load_transcript()
        self._refresh_header()

        try:
            self.query_one(Composer).focus()
        except Exception:
            pass

        if _ANNOUNCE:
            from tts import speak

            speak("Control console online.")

    def _sink_on_line(self, line: str) -> None:
        with self._rail_lock:
            self._sink_queue.append(line)

    def _sink_pump(self) -> None:
        try:
            self._flush_sink()
        except Exception:
            pass

    def _flush_sink(self) -> None:
        with self._rail_lock:
            lines = list(self._sink_queue)
            self._sink_queue.clear()
        for ln in lines:
            self._rail_ui(ln, "dim")

    # ── session transcript ────────────────────────────────
    def _load_transcript(self) -> None:
        try:
            from session_store import load_messages

            msgs = load_messages(self._session_id)
        except Exception as e:
            self._rail(f"Transcript error: {e}", "red")
            return
        log = self.query_one("#transcript", RichLog)
        log.clear()
        shown = 0
        for m in msgs[-80:]:
            role = m.get("role", "")
            content = (m.get("content") or "").strip()
            if not content or role not in ("user", "assistant"):
                continue
            if len(content) > 900:
                content = content[:900] + "…"
            self._transcript_line_ui(role, content)
            shown += 1
        if shown == 0:
            log.write(Text("Type a message below, or use //help.", style="dim"))

    def _refresh_transcript(self) -> None:
        # Session may have changed via local commands
        with self._session_lock:
            sid = terminal._session_id
        if sid != self._last_session:
            self._session_id = sid
            self._load_transcript()
            self._refresh_header()
            self._rail(f"Switched to session '{sid}'.", "yellow")

    # ── header ────────────────────────────────────────────
    def _refresh_header(self) -> None:
        try:
            header = self.query_one("#header", Static)
        except Exception:
            return
        rec = "●REC" if self._recording.is_set() else ""
        busy = "working…" if self._busy else "idle"
        header.update(
            Text.assemble(
                ("◈ J.A.R.V.I.S.", "bold cyan"),
                ("  session: ", "dim"),
                (self._session_id, ""),
                ("   ● ONLINE ", "bold green"),
                (" " + rec, "red bold") if rec else "",
                (f"  [{busy}]", "dim"),
            )
        )

    # ── submit flow ───────────────────────────────────────
    @on(Composer.Submit)
    def _on_submit(self, event: Composer.Submit) -> None:
        self._submit(event.text)

    def _submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        composer = self.query_one(Composer)
        composer.text = ""

        if self._handle_local(text):
            return

        self._transcript_line("user", text)
        if self._busy:
            self._pending.append(text)
            self._rail(f"Queued: {text[:80]}", "yellow")
            return
        self._busy = True
        self._refresh_header()
        threading.Thread(target=self._request_worker, args=(text,), daemon=True).start()

    def _handle_local(self, text: str) -> bool:
        lowered = text.lower()
        if lowered in ("quit", "exit", "q"):
            self.action_quit_app()
            return True
        if lowered == "wake":
            self._rail("Wake-word mode lives in the classic REPL (JARVIS_TUI=0). Here: ctrl+r records.", "yellow")
            return True
        if lowered.startswith(("/mode", "/m ")):
            self._rail(
                "One input mode in the console. Enter sends; paste drops whole; //paste reads clipboard.",
                "yellow",
            )
            return True
        if terminal.handle_local_command(text):
            self._refresh_transcript()
            return True
        return False

    def _request_worker(self, text: str) -> None:
        t0 = time.perf_counter()
        try:
            reply = self._brain.process(text, self._session_id)
            if self._cancel:
                self._cancel = False
                self._rail("Request cancelled — reply discarded.", "yellow")
            else:
                latency = (time.perf_counter() - t0) * 1000
                self._rail(f"reply in {latency:.0f} ms", "dim")
                self._transcript_line("assistant", reply)
                try:
                    from tts import speak, wait_for_speech

                    speak(reply)
                    wait_for_speech()
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            self._rail(f"Error: {e}", "red")
        finally:
            self.call_from_thread(self._request_done)

    def _request_done(self) -> None:
        self._busy = False
        self._refresh_header()
        if self._pending:
            nxt = self._pending.pop(0)
            self._submit(nxt)

    # ── event bus → rail ──────────────────────────────────
    def _ev_tool_call(self, payload: dict) -> None:
        tool = payload.get("tool", "?")
        args = payload.get("args", "")
        self._rail(f"📡 {tool}({args})", "cyan")

    def _ev_subagent_started(self, payload: dict) -> None:
        self._rail(f"▶ agent started: {payload.get('goal', '')[:80]}", "magenta")

    def _ev_subagent_progress(self, payload: dict) -> None:
        step = payload.get("step", "")
        tool = payload.get("tool", "")
        if tool:
            self._rail(f"· agent step {step}: {tool}", "magenta")

    def _ev_subagent_completed(self, payload: dict) -> None:
        self._rail("✔ agent completed", "magenta")

    def _ev_proactive_alert(self, payload: dict) -> None:
        msg = payload.get("message", "")
        if msg:
            self._rail(f"[Alert] {msg}", "yellow")

    def _ev_self_test(self, payload: dict) -> None:
        msg = payload.get("message", "")
        if msg:
            self._rail(f"[Self-Test] {msg}", "magenta")

    # ── actions ───────────────────────────────────────────
    def action_record(self) -> None:
        if self._recording.is_set():
            return
        self._recording.set()
        self._rail("Recording… ctrl+r again while recording stops.", "red")
        threading.Thread(target=self._record_worker, daemon=True).start()

    def _record_worker(self) -> None:
        try:
            from tts import stop_speaking

            stop_speaking()
            text = terminal.record_and_transcribe()
        except Exception as e:  # noqa: BLE001
            text = ""
            self._rail(f"Mic error: {e}", "red")
        finally:
            self._recording.clear()
        if text and text.strip():
            self.call_from_thread(self._submit, text.strip())

    def action_cancel(self) -> None:
        if self._busy:
            self._cancel = True
            self._rail("Cancelling current request…", "yellow")
        else:
            try:
                self.query_one(Composer).text = ""
            except Exception:
                pass

    def action_quit_app(self) -> None:
        if self._vision is not None:
            self._vision.stop()
        self.exit()

    def _focus_tab(self, pane_id: str) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = pane_id
            handler = {
                "main": lambda: self.query_one(Composer).focus(),
                "vision": self._ensure_vision,
            }.get(pane_id)
            if handler:
                handler()
            else:
                self.query_one(f"#{pane_id}-pane")
        except Exception:
            pass

    def action_focus_main(self) -> None:
        self._focus_tab("main")

    def action_focus_system(self) -> None:
        self._focus_tab("system")

    def action_focus_brain(self) -> None:
        self._focus_tab("brain")

    def action_focus_memory(self) -> None:
        self._focus_tab("memory")

    def action_focus_vision(self) -> None:
        self._focus_tab("vision")

    def action_focus_workflows(self) -> None:
        self._focus_tab("workflows")

    def action_focus_tools(self) -> None:
        self._focus_tab("tools")

    def _ensure_vision(self, start_thread: bool | None = None) -> None:
        if start_thread is None:
            start_thread = _AUTOSTART_VISION
        if self._vision is not None:
            return
        if not start_thread:
            if not self._vision_hint_shown:
                self._vision_hint_shown = True
                self._vision_status_ui("Vision module ready — press R or the button to start the camera.")
            return
        self._vision = VisionStream(
            on_frame=self._vision_frame,
            on_status=self._vision_status,
            active_check=lambda: True,
        )
        self._vision.start()
        self._rail("Vision subsystem initialized.", "cyan")
        if _ANNOUNCE:
            from tts import speak

            speak("Vision subsystem initialized. Retina: 320 by 180. Fifty seven thousand, six hundred neurons.")

    def toggle_vision_mode(self) -> None:
        self._ensure_vision(start_thread=True)
        if self._vision is None:
            return
        self._vision.retina_mode = not self._vision.retina_mode
        mode = "RETINA" if self._vision.retina_mode else "CAMERA"
        self._vision_status(f"Switched to {mode} view.")
        self._rail(f"Vision: {mode} view.", "cyan")

    def _vision_frame(self, ascii_frame: str, caption: str) -> None:
        try:
            try:
                self.call_from_thread(self._vision_frame_ui, ascii_frame, caption)
            except RuntimeError:
                self._vision_frame_ui(ascii_frame, caption)
        except Exception:
            pass

    def _vision_frame_ui(self, ascii_frame: str, caption: str) -> None:
        try:
            self.query_one("#vision-frame", Static).update(ascii_frame)
            self.query_one("#vision-status", Static).update(caption)
        except Exception:
            pass

    def _vision_status(self, msg: str) -> None:
        try:
            try:
                self.call_from_thread(self._vision_status_ui, msg)
            except RuntimeError:
                self._vision_status_ui(msg)
        except Exception:
            pass

    def _vision_status_ui(self, msg: str) -> None:
        try:
            self.query_one("#vision-status", Static).update(msg)
        except Exception:
            pass

    # ── panel refreshers ──────────────────────────────────
    def _sys_info_text(self) -> str:
        try:
            from tools.system_tools import disk_usage, get_system_info, get_top_processes

            si = get_system_info()
            du = disk_usage()
            lines = []
            cpu = si.get("cpu_percent")
            mem = si.get("memory_percent")
            if cpu is not None:
                lines.append(f"CPU: {cpu}")
            if mem is not None:
                lines.append(f"RAM: {mem}")
            if isinstance(du, dict):
                lines.append(f"Disk: {du.get('percent', '?')} used ({du.get('free_gb', '?')} GB free)")
            procs = get_top_processes(by="memory", count=5)
            if procs:
                lines.append("")
                lines.append("TOP PROCESSES (mem):")
                for p in procs:
                    lines.append(f"  {p}")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"System data unavailable: {e}"

    def _refresh_system_pane(self) -> None:
        try:
            self.query_one("#system-pane").update(self._sys_info_text())
        except Exception:
            pass

    def _brain_text(self) -> str:
        try:
            from brain import (
                _provider_backoff_until,
                _provider_consecutive_failures,
                _provider_health_scores,
                get_provider_status_summary,
                get_runtime_status,
            )

            st = get_runtime_status()
            lines = [f"Primary: {st.get('model_preferred', '?')}"]
            used = st.get("model_last_used")
            if used:
                lines.append(f"Last used: {used}")
            lines.append(get_provider_status_summary() or "")
            lines.append("")
            now = time.time()
            for provider in sorted(set(_provider_health_scores) | set(_provider_consecutive_failures)):
                health = _provider_health_scores.get(provider, 100)
                fails = _provider_consecutive_failures.get(provider, 0)
                backoff = _provider_backoff_until.get(provider, 0)
                if backoff > now:
                    status = f"BACKED OFF ({int(backoff - now)}s)"
                else:
                    status = "OK"
                lines.append(f"  {provider:<18} {status:<20} health {health}  fails {fails}")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"Brain data unavailable: {e}"

    def _refresh_brain_pane(self) -> None:
        try:
            self.query_one("#brain-pane").update(self._brain_text())
        except Exception:
            pass

    def _memory_text(self) -> str:
        try:
            from graph_memory import get_graph_summary
            from memory import get_all_memories
            from rag_memory import get_rag_stats

            rows = get_all_memories()
            by_type: dict[str, int] = {}
            for r in rows:
                by_type[r[0]] = by_type.get(r[0], 0) + 1
            lines = [f"Explicit memories: {len(rows)}"]
            for t, c in sorted(by_type.items()):
                lines.append(f"  {t}: {c}")
            stats = get_rag_stats()
            lines.append("")
            lines.append(f"RAG files: {stats.get('total_files', 0)}  chunks: {stats.get('total_chunks', 0)}")
            try:
                lines.append(get_graph_summary())
            except Exception:
                pass
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"Memory data unavailable: {e}"

    def _refresh_memory_pane(self) -> None:
        try:
            self.query_one("#memory-pane").update(self._memory_text())
        except Exception:
            pass

    def _workflows_text(self) -> str:
        try:
            from workflow_engine import get_run_history, list_workflows

            lines = [f"Workflows ({len(list_workflows())}):"]
            for w in list_workflows()[:12]:
                tag = "builtin" if w.get("builtin") else "user"
                lines.append(f"  [{tag}] {w['name']}: {w['description'][:70]}")
            lines.append("")
            lines.append("RUN HISTORY:")
            for h in get_run_history(8):
                lines.append(f"  #{h['id']} {h['workflow']} — {h['status']} ({h.get('started', '?')[:19]})")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"Workflow data unavailable: {e}"

    def _refresh_workflows_pane(self) -> None:
        try:
            self.query_one("#workflows-pane").update(self._workflows_text())
        except Exception:
            pass

    def _tools_text(self) -> str:
        try:
            from tools import TOOL_REGISTRY
            from tools.inspect_tools import inspect_capabilities

            caps = inspect_capabilities()
            return f"{len(TOOL_REGISTRY)} tools loaded\n\n{caps}"
        except Exception as e:  # noqa: BLE001
            return f"Tool data unavailable: {e}"

    def _refresh_tools_pane(self) -> None:
        try:
            self.query_one("#tools-pane").update(self._tools_text())
        except Exception:
            pass

    # ── compose ───────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Static("", id="header")

        with Horizontal():
            with Vertical(id="main-col"):
                with TabbedContent():
                    with TabPane("MAIN", id="main-tab"):
                        with VerticalScroll():
                            yield RichLog(id="transcript", markup=True, highlight=True, wrap=True, min_width=60)
                        with Vertical(id="composer-wrap"):
                            yield Composer(placeholder="Type a message or task… Enter to send, ctrl+enter newline")
                    with TabPane("SYSTEM", id="system-tab"):
                        yield Static("Loading system…", id="system-pane")
                    with TabPane("BRAIN", id="brain-tab"):
                        yield Static("Loading providers…", id="brain-pane")
                    with TabPane("MEMORY", id="memory-tab"):
                        yield Static("Loading memory…", id="memory-pane")
                    with TabPane("VISION", id="vision-tab"):
                        yield VisionPane()
                    with TabPane("WORKFLOWS", id="workflows-tab"):
                        yield Static("Loading workflows…", id="workflows-pane")
                    with TabPane("TOOLS", id="tools-tab"):
                        yield Static("Loading tools…", id="tools-pane")

            with Vertical(id="rail-col"):
                yield Static("ACTIVITY", classes="pane-label")
                yield RichLog(id="activity", markup=True, highlight=True, wrap=True)

        yield Footer()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        pane_id = event.pane.id or ""
        if pane_id == "vision-tab":
            self._ensure_vision(start_thread=_AUTOSTART_VISION)

    @on(Button.Pressed, "#vision-toggle")
    def _on_vision_toggle(self, event: Button.Pressed) -> None:
        self.toggle_vision_mode()


# ──────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────
def run_tui(session_id: str) -> None:
    """Run the console; installs the stdout sink and restores it on exit."""
    app = JarvisConsole(session_id=session_id)

    def on_line(line: str) -> None:
        app._sink_on_line(line)

    sink = StdoutSink(on_line)
    app._sink = sink

    real_out, real_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sink
        sys.stderr = sink
        app.run()
    finally:
        sys.stdout = real_out
        sys.stderr = real_err
