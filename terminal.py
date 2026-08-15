import datetime
import os
import queue
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import requests
import scipy.io.wavfile as wav
import scipy.signal as signal
import sounddevice as sd

import proactive
from brain import (
    _provider_backoff_until,
    _provider_consecutive_failures,
    _provider_health_scores,
    _summarize_paste,
    get_provider_status_summary,
    get_runtime_status,
    process,
)
from brain import init as brain_init
from config import MIC_DEVICE_INDEX, RECORD_SAMPLE_RATE, RECORD_SECONDS, SAMPLE_RATE
from trigger_engine import start as triggers_start
from tts import speak, stop_speaking, wait_for_speech

_INPUT_DEVICE_INDEX = None
_last_alert_ts = time.time()
_last_self_test_ts = time.time()


def _flush_proactive_alerts():
    """Print proactive alerts that arrived since the last prompt."""
    global _last_alert_ts
    try:
        from event_bus import get_recent

        events = get_recent("proactive_alert", 10)
        for ev in events:
            ts = ev.get("timestamp", 0) or 0
            if ts <= _last_alert_ts:
                continue
            msg = ev.get("message", "")
            if msg:
                print(f"\n  [Alert] {msg}")
        if events:
            _last_alert_ts = events[-1].get("timestamp", 0) or _last_alert_ts
    except Exception:
        pass


def _flush_self_test_events():
    """Print self-test progress events that arrived since the last prompt."""
    global _last_self_test_ts
    try:
        from event_bus import get_recent

        events = get_recent("self_test", 10)
        for ev in events:
            ts = ev.get("timestamp", 0) or 0
            if ts <= _last_self_test_ts:
                continue
            msg = ev.get("message", "")
            if msg:
                print(f"\n  [Self-Test] {msg}")
        if events:
            _last_self_test_ts = events[-1].get("timestamp", 0) or _last_self_test_ts
    except Exception:
        pass


def _resolve_input_device():
    """
    Pick a working input device instead of relying on platform default (-1).
    Respects MIC_DEVICE_INDEX env var if set to a valid index.
    """
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"  Mic device query failed: {e}")
        return None

    # Respect explicit config first
    if MIC_DEVICE_INDEX >= 0:
        try:
            info = sd.query_devices(MIC_DEVICE_INDEX)
            if info.get("max_input_channels", 0) > 0:
                return MIC_DEVICE_INDEX
        except Exception:
            pass

    # Try currently configured default input first, if valid.
    try:
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        if isinstance(default_in, int) and default_in >= 0:
            info = sd.query_devices(default_in)
            if info.get("max_input_channels", 0) > 0:
                return default_in
    except Exception:
        pass

    for i, info in enumerate(devices):
        if info.get("max_input_channels", 0) > 0:
            return i

    return None


def list_input_devices():
    """Print available input microphones and return their indices."""
    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        print("  Available microphones:")
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                marker = " (DEFAULT)" if isinstance(default_in, int) and i == default_in else ""
                print(f"    [{i}] {d['name']}{marker}")
        return [i for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    except Exception as e:
        print(f"  Could not list devices: {e}")
        return []


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────


def terminal_init():
    """Initialize terminal subsystems (brain, proactive, triggers, learner, RAG, network check)."""
    global _INPUT_DEVICE_INDEX

    # Interactive mic selection when MIC_DEVICE_INDEX is auto (-1)
    if MIC_DEVICE_INDEX < 0:
        input_devices = list_input_devices()
        if input_devices:
            try:
                choice = input("  Select mic index (Enter=auto): ").strip()
                if choice.isdigit() and int(choice) in input_devices:
                    _INPUT_DEVICE_INDEX = int(choice)
                else:
                    _INPUT_DEVICE_INDEX = _resolve_input_device()
            except (EOFError, KeyboardInterrupt):
                _INPUT_DEVICE_INDEX = _resolve_input_device()
        else:
            _INPUT_DEVICE_INDEX = _resolve_input_device()
    else:
        _INPUT_DEVICE_INDEX = _resolve_input_device()
    if _INPUT_DEVICE_INDEX is not None:
        try:
            mic = sd.query_devices(_INPUT_DEVICE_INDEX)
            print(f"  Mic selected: [{_INPUT_DEVICE_INDEX}] {mic.get('name', 'Unknown')}")
        except Exception:
            print(f"  Mic selected: [{_INPUT_DEVICE_INDEX}]")
    else:
        print("  Warning: no input microphone found; voice recording may fail.")

    brain_init()
    proactive.init(speak, process)
    proactive.start()
    triggers_start()
    try:
        from self_test.monitor import start as self_test_monitor_start

        self_test_monitor_start()
    except Exception:
        pass

    print("  Indexing RAG folder...")
    try:
        from config import RAG_FOLDER
        from rag_memory import get_rag_stats, index_folder

        files, chunks = index_folder(RAG_FOLDER)
        stats = get_rag_stats()
        print(f"  RAG: {stats.get('total_chunks', chunks)} chunks from {stats.get('total_files', files)} files")
    except Exception as e:
        print(f"  RAG index skipped: {e}")

    print("  Checking connections...")
    try:
        requests.get("https://www.google.com", timeout=3)
        print("  Online — using cloud AI provider chain.")
    except Exception:
        print("  Offline — cloud AI providers unavailable until internet returns.")

    try:
        from healthcheck import run_healthcheck

        hc = run_healthcheck()
        if hc["ok"]:
            print("  Health check: all systems OK")
        else:
            fails = [c["name"] for c in hc["checks"] if not c["ok"]]
            print(f"  Health check: {len(fails)} issue(s): {', '.join(fails)} (type 'health' for details)")
    except Exception:
        pass


# ─────────────────────────────────────────────
# RECORD + TRANSCRIBE
# ─────────────────────────────────────────────
def record_and_transcribe(seconds=RECORD_SECONDS) -> str:
    print(f"  🎙  Listening for {seconds} seconds...")
    try:
        audio = sd.rec(
            int(seconds * RECORD_SAMPLE_RATE),
            samplerate=RECORD_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=_INPUT_DEVICE_INDEX,
        )
        sd.wait()
        # Resample 44100 → 16000 for Whisper
        audio_resampled = signal.resample(audio, int(len(audio) * SAMPLE_RATE / RECORD_SAMPLE_RATE)).astype(np.int16)
    except Exception as e:
        print(f"  Mic error: {e}")
        # Fallback to default device
        try:
            print("  Trying default device...")
            audio = sd.rec(
                int(seconds * RECORD_SAMPLE_RATE),
                samplerate=RECORD_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=_INPUT_DEVICE_INDEX,
            )
            sd.wait()
            audio_resampled = signal.resample(audio, int(len(audio) * SAMPLE_RATE / RECORD_SAMPLE_RATE)).astype(np.int16)
        except Exception as e2:
            print(f"  Fallback mic error: {e2}")
            return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, SAMPLE_RATE, audio_resampled)
        tmp = f.name

    try:
        from stt import transcribe_file

        return transcribe_file(tmp).strip()
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────


class Priority(IntEnum):
    HIGH = 0  # stop, cancel, interrupt
    FOREGROUND = 1  # user commands (voice/text)
    BACKGROUND = 2  # proactive, learner, timers
    LOW = 3  # self-analysis, scans


class InputMode:
    TEXT = "text"
    PASTE = "paste"
    QUEUE = "queue"


_current_mode = InputMode.TEXT
_paste_buffer: list[str] = []
_queue_buffer: list[str] = []


# ─────────────────────────────────────────────
# MULTI-PROBLEM DETECTION
# ─────────────────────────────────────────────
def extract_problems(text: str) -> list[str]:
    lines = text.strip().split("\n")
    problem_headers = []
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"^(?:Problem|Question|Challenge)\s+\d+", s, re.IGNORECASE):
            problem_headers.append(i)
        elif re.match(r"^\d+[.)]\s", s) and len(s) > 4:
            problem_headers.append(i)
    if problem_headers and len(problem_headers) >= 2:
        problems = []
        for idx, start in enumerate(problem_headers):
            end = problem_headers[idx + 1] if idx + 1 < len(problem_headers) else len(lines)
            p = "\n".join(lines[start:end]).strip()
            if p:
                problems.append(p)
        return problems
    # Check for separator patterns like ---, ***, ___
    if re.search(r"^[-*_]{3,}\s*$", text, re.MULTILINE):
        paragraphs = [p.strip() for p in re.split(r"\n[-*_]{3,}\s*\n", text) if p.strip()]
        if len(paragraphs) >= 3:
            return paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 3:
        return paragraphs
    return [text]


def prompt_problem_strategy(problem_count: int) -> tuple[str, list[int] | None]:
    print(f"\n  {'=' * 45}")
    print(f"  Detected {problem_count} problems/questions")
    print(f"  {'=' * 45}")
    print("    [1] Sequential — ask before each")
    print("    [2] Auto — solve all without asking")
    print("    [3] Pick — select which to solve")
    print("    [4] First — solve only the first")
    print(f"  {'=' * 45}")
    choice = input("  Strategy (1-4, Enter=1): ").strip()
    choice = choice or "1"
    if choice == "3":
        picks = input("  Problem numbers to solve (e.g. 1,3,5): ").strip()
        try:
            selected = [int(x.strip()) for x in picks.split(",") if x.strip().isdigit()]
            return "pick", selected
        except (ValueError, TypeError):
            return "sequential", None
    return choice, None


def handle_multi_problem(problems: list[str], strategy: str, selected: list[int] | None = None):
    if strategy == "pick" and selected:
        indices = [i - 1 for i in selected if 1 <= i <= len(problems)]
    elif strategy == "4":
        indices = [0]
    elif strategy in ("2", "auto"):
        indices = list(range(len(problems)))
    else:
        indices = list(range(len(problems)))

    for idx in indices:
        problem = problems[idx]
        print(f"\n  {'─' * 40}")
        print(f"  Problem {idx + 1}/{len(problems)}:")
        preview = problem[:200] + "..." if len(problem) > 200 else problem
        print(f"  {preview}")

        if strategy in ("1", "sequential", "3", "pick") and strategy != "2":
            if strategy == "pick" and selected and idx not in indices:
                continue
            resp = input("  Solve this? (Enter=yes, n=skip, all=rest auto): ").strip().lower()
            if resp == "n":
                print("  Skipped.")
                continue
            elif resp == "all":
                strategy = "2"

        handle_input(problem)


@dataclass(order=True)
class TaskData:
    priority: Priority
    request_id: int = field(compare=False)
    text: str = field(compare=False)


class RequestScheduler:
    """Two-tier priority scheduler: foreground (ordered, 1 worker) + background (configurable workers)."""

    _stdout_lock = threading.Lock()

    def __init__(self, foreground_workers: int = 1, background_workers: int = 1):
        self.fg_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.bg_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running = threading.Event()
        self.running.set()
        self._counter = 0
        self._lock = threading.Lock()

        for i in range(foreground_workers):
            t = threading.Thread(target=self._fg_worker, name=f"jarvis-fg-{i}", daemon=True)
            t.start()

        for i in range(background_workers):
            t = threading.Thread(target=self._bg_worker, name=f"jarvis-bg-{i}", daemon=True)
            t.start()

    def submit(self, text: str, priority: Priority = Priority.FOREGROUND) -> int:
        with self._lock:
            self._counter += 1
            rid = self._counter
        task = TaskData(priority=priority, request_id=rid, text=text)
        if priority <= Priority.FOREGROUND:
            self.fg_queue.put(task)
        else:
            self.bg_queue.put(task)
        return rid

    def _fg_worker(self):
        while self.running.is_set():
            try:
                task = self.fg_queue.get(timeout=0.5)
                self._execute(task)
                self.fg_queue.task_done()
            except queue.Empty:
                continue

    def _bg_worker(self):
        while self.running.is_set():
            try:
                task = self.bg_queue.get(timeout=1.0)
                self._execute(task)
                self.bg_queue.task_done()
            except queue.Empty:
                continue

    def _execute(self, task: TaskData):
        tag = f"[#{task.request_id} {task.priority.name}]"
        with RequestScheduler._stdout_lock:
            try:
                print(f"\n  You → {task.text}  {tag}")
                print("  Jarvis thinking...")
                reply = process(task.text, _session_id)
                usage = get_provider_status_summary()
                if usage:
                    print(f"  [{usage}]")
                print(f"  Jarvis → {reply}  {tag}\n")
            except Exception as e:
                print(f"  Error {tag}: {e}")
        speak(reply, interrupt=True)
        wait_for_speech()

    def shutdown(self):
        self.running.clear()


# Feature flags (env defaults)
_USE_SCHEDULER = os.environ.get("JARVIS_USE_SCHEDULER", "1") == "1"
_FG_WORKERS = int(os.environ.get("JARVIS_FG_WORKERS", "1"))
_BG_WORKERS = int(os.environ.get("JARVIS_BG_WORKERS", "1"))

_scheduler: Optional[RequestScheduler] = None

_session_id = os.environ.get("JARVIS_SESSION", "default")


# ─────────────────────────────────────────────
# HANDLE INPUT
# ─────────────────────────────────────────────
def handle_input(text: str):
    if not text:
        return
    if not _USE_SCHEDULER:
        handle_input_legacy(text)
        return
    global _scheduler
    if _scheduler is None:
        _scheduler = RequestScheduler(foreground_workers=_FG_WORKERS, background_workers=_BG_WORKERS)
    _scheduler.submit(text, Priority.FOREGROUND)


def handle_input_legacy(text: str):
    """Original thread-per-request handler, kept for instant rollback."""
    if not text:
        return

    def _do_handle(t: str):
        try:
            print(f"\n  You → {t}")
            print("  Jarvis thinking...")
            reply = process(t, _session_id)
            usage = get_provider_status_summary()
            if usage:
                print(f"  [{usage}]")
            print(f"  Jarvis → {reply}\n")
            speak(reply, interrupt=True)
            wait_for_speech()
        except Exception as e:
            print(f"  Error handling input: {e}")

    threading.Thread(target=_do_handle, args=(text,), daemon=True).start()


# ─────────────────────────────────────────────
# WAKE WORD CALLBACK
# ─────────────────────────────────────────────
_wake_active = threading.Event()


def on_wake_word():
    if _wake_active.is_set():
        return
    _wake_active.set()
    stop_speaking()
    speak("Yes?", interrupt=True)
    wait_for_speech()
    text = record_and_transcribe()
    if text:
        handle_input(text)
    else:
        speak("Sorry, I didn't catch that.")
    _wake_active.clear()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def _sanitize_input(text: str) -> str | None:
    """Return cleaned text or None if too garbled to process."""
    if not text or not text.strip():
        return None
    stripped = text.strip()
    if len(stripped) > 10:
        char_counts = {}
        for c in stripped.lower():
            if c.isalpha():
                char_counts[c] = char_counts.get(c, 0) + 1
        max_count = max(char_counts.values()) if char_counts else 0
        if max_count / len(stripped) > 0.5:
            return None
    return stripped


def main():
    global _current_mode, _paste_buffer, _queue_buffer, _session_id
    terminal_init()
    print("\nModes:")

    print("  [w] Wake word — say 'Hey Jarvis' or 'Jarvis [command]'")
    print("  [m] Manual   — press Enter to speak, or type a message")
    print("  [q] Quit\n")

    mode = input("Choose mode (w/m/q): ").strip().lower()

    if mode == "q":
        sys.exit(0)

    elif mode == "w":
        import wakeword

        wakeword.start(on_wake_word)
        print("\n  Jarvis is listening. Say 'Hey Jarvis' to activate.")
        print(f"  Session: {_session_id}  (type 'session' to manage sessions)")
        print("  Press Ctrl+C to quit.\n")
        try:
            while True:
                _flush_proactive_alerts()
                _flush_self_test_events()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            speak("Goodbye.")
            wait_for_speech()
            wakeword.stop()

    else:
        print("\nPress Enter to speak, type a message, or 'quit' to exit.")
        print(f"Type 'wake' to switch to wake word mode mid-session. Session: {_session_id}\n")

        while True:
            try:
                _flush_proactive_alerts()
                _flush_self_test_events()
                # Mode-specific input handling
                if _current_mode == InputMode.PASTE:
                    raw_line = input("Paste → ")
                    if raw_line.startswith("/"):
                        raw_input = raw_line.strip()
                        user_input = _sanitize_input(raw_input)
                        if user_input is None:
                            continue
                    elif raw_line.strip() == "":
                        if _paste_buffer:
                            full_text = "\n".join(_paste_buffer)
                            _paste_buffer = []
                            # Summarize large pastes to avoid context pollution
                            if len(full_text) > 2000:
                                print(f"  Summarizing paste ({len(full_text)} chars)...")
                                full_text = _summarize_paste(full_text)
                            raw_input = full_text
                            user_input = _sanitize_input(raw_input)
                            if user_input is None:
                                continue
                            # Check for multi-problem content
                            problems = extract_problems(raw_input)
                            if len(problems) >= 2:
                                strategy, selected = prompt_problem_strategy(len(problems))
                                handle_multi_problem(problems, strategy, selected)
                                continue
                        else:
                            continue
                    else:
                        _paste_buffer.append(raw_line.rstrip())
                        continue

                elif _current_mode == InputMode.QUEUE:
                    raw_line = input("Queue → ").strip()
                    if raw_line.startswith("/"):
                        raw_input = raw_line
                        user_input = _sanitize_input(raw_input)
                        if user_input is None:
                            continue
                    elif raw_line.lower() in ("go", "run", "execute"):
                        if _queue_buffer:
                            print(f"  Executing {len(_queue_buffer)} queued items...")
                            q_items = list(_queue_buffer)
                            _queue_buffer.clear()
                            for item in q_items:
                                handle_input(item)
                        continue
                    elif raw_line.lower() in ("clear",):
                        _queue_buffer.clear()
                        print("  Queue cleared.")
                        continue
                    elif raw_line.lower() in ("show", "list", "queue"):
                        if _queue_buffer:
                            print(f"  Queue ({len(_queue_buffer)} items):")
                            for i, item in enumerate(_queue_buffer, 1):
                                print(f"    {i}. {item[:120]}")
                        else:
                            print("  Queue is empty.")
                        continue
                    elif raw_line:
                        _queue_buffer.append(raw_line)
                        print(f"  Queued ({len(_queue_buffer)}). Type 'go' to execute.")
                        continue
                    else:
                        continue

                else:
                    raw_input = input("You → ").strip()
                    user_input = _sanitize_input(raw_input)
                    if user_input is None:
                        continue

            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                speak("Goodbye.")
                wait_for_speech()
                sys.exit(0)

            if not user_input and _current_mode != InputMode.PASTE:
                text = record_and_transcribe()
                if text:
                    handle_input(text)
                else:
                    print("  Didn't catch that.\n")

            elif user_input.lower() in ("quit", "exit", "q"):
                speak("Goodbye.")
                wait_for_speech()
                sys.exit(0)

            elif user_input.lower() == "wake":
                import wakeword

                wakeword.start(on_wake_word)
                print("  Wake word active — say 'Hey Jarvis' anytime.\n")

            elif user_input.lower() in ("plugins", "plugins list"):
                from plugin_manager import print_plugin_status

                print_plugin_status()

            elif user_input.lower() == "plugins reload":
                from plugin_manager import reload_plugins

                print("  Reloading plugins...")
                results = reload_plugins()
                loaded = sum(1 for r in results if r.get("ok"))
                print(f"  Loaded {loaded}/{len(results)} plugins.")

            elif user_input.lower().startswith("plugin install "):
                source = user_input[len("plugin install ") :].strip()
                from plugin_manager import install_plugin

                result = install_plugin(source)
                if result.get("ok"):
                    print(f"  Installed plugin: {result['name']}")
                else:
                    print(f"  Failed: {result.get('error', 'unknown')}")

            elif user_input.lower() in ("workflows", "wf list"):
                from workflow_engine import list_workflows

                for w in list_workflows():
                    tag = "builtin" if w.get("builtin") else "user"
                    print(f"  [{tag}] {w['name']}: {w['description']}")

            elif user_input.lower().startswith("workflow run ") or user_input.lower().startswith("wf run "):
                parts = user_input.split(maxsplit=2)
                wf_name = parts[2] if len(parts) > 2 else ""
                from workflow_engine import run_workflow

                print(f"  Running workflow '{wf_name}'...")
                result = run_workflow(wf_name)
                if result.get("ok"):
                    print(f"  Done. Result: {result['result'][:300]}")
                else:
                    print(f"  Failed: {result.get('error', 'unknown')}")

            elif user_input.lower() in ("wf history", "workflow history"):
                from workflow_engine import get_run_history

                for h in get_run_history(10):
                    print(f"  #{h['id']} {h['workflow']} — {h['status']} ({h.get('started', '?')[:19]})")

            elif user_input.lower().startswith("ingest "):
                path = user_input[7:].strip()
                from rag_memory import index_folder

                print(f"  Indexing {path}...")
                files, chunks = index_folder(path)
                print(f"  Indexed {files} files, {chunks} chunks")

            elif user_input.lower().startswith("rag search "):
                query = user_input[11:].strip()
                from rag_memory import search_rag_structured

                result = search_rag_structured(query)
                if result.get("results"):
                    for r in result["results"]:
                        print(f"  [{r['source']}] (score={r['score']}): {r['text'][:200]}")
                else:
                    print("  No results found.")

            elif user_input.lower() in ("rag prune", "rag clean"):
                from rag_memory import prune_stale_entries

                deleted = prune_stale_entries(90)
                print(f"  Pruned {deleted} stale chunks older than 90 days.")

            elif user_input.lower() == "rag stats":
                from rag_memory import get_rag_stats

                stats = get_rag_stats()
                print(f"  Files: {stats.get('total_files', 0)}")
                print(f"  Chunks: {stats.get('total_chunks', 0)}")
                print(f"  BM25 ready: {stats.get('bm25_ready', '?')}")

            elif user_input.lower() in ("triggers", "trigger list"):
                from trigger_engine import list_triggers

                for t in list_triggers():
                    status = "✓" if t["enabled"] else "✗"
                    print(f"  [{t['id']}] {status} {t['name']} ({t['trigger_type']}/{t['action_type']}) — {t.get('description', '')}")
                    if t.get("next_fire"):
                        print(f"       next: {t['next_fire']}, count: {t['fire_count']}")

            elif user_input.lower().startswith("trigger add "):
                from trigger_engine import create_trigger

                parts = user_input.split(None, 5)
                if len(parts) < 5:
                    print("  Usage: trigger add <name> <type> <schedule> <action_type> <action_target> [description]")
                    print("  Types: cron, interval, once, event")
                    print("  Action types: workflow, tool, prompt")
                    print("  Examples:")
                    print("    trigger add cleanup interval 6h tool organize_downloads")
                    print("    trigger add briefing cron '0 8 * * *' prompt 'give me a morning briefing'")
                    print("    trigger add weather_once once 2026-06-22T09:00:00 tool get_weather")
                else:
                    try:
                        name = parts[2]
                        t = parts[3]
                        sched = parts[4]
                        a_type = parts[5].split()[0] if len(parts) > 5 else ""
                        rest = user_input.split(None, 6)
                        a_target = rest[6].split(" ", 1)[0] if len(rest) > 6 else ""
                        desc = rest[6].split(" ", 1)[1] if len(rest) > 6 and " " in rest[6] else ""
                        t = create_trigger(name, t, sched, a_type, a_target, {}, desc)
                        print(f"  Created trigger [{t['id']}] {t['name']}")
                    except (IndexError, ValueError) as e:
                        print(f"  Error: {e}")

            elif user_input.lower().startswith("trigger remove ") or user_input.lower().startswith("trigger delete "):
                from trigger_engine import delete_trigger

                try:
                    tid = int(user_input.split()[-1])
                    if delete_trigger(tid):
                        print(f"  Deleted trigger [{tid}]")
                    else:
                        print(f"  Trigger [{tid}] not found")
                except (IndexError, ValueError):
                    print("  Usage: trigger remove <id>")

            elif user_input.lower().startswith("trigger pause "):
                from trigger_engine import disable_trigger

                try:
                    tid = int(user_input.split()[-1])
                    trig = disable_trigger(tid)
                    print(f"  Paused trigger [{tid}] {trig.get('name', '')}")
                except (IndexError, ValueError):
                    print("  Usage: trigger pause <id>")

            elif user_input.lower().startswith("trigger resume "):
                from trigger_engine import enable_trigger

                try:
                    tid = int(user_input.split()[-1])
                    trig = enable_trigger(tid)
                    print(f"  Resumed trigger [{tid}] {trig.get('name', '')}")
                except (IndexError, ValueError):
                    print("  Usage: trigger resume <id>")

            elif user_input.lower() == "trigger history" or user_input.lower() == "trigger log":
                from trigger_engine import get_trigger, get_trigger_history

                for h in get_trigger_history(limit=20):
                    trig = get_trigger(h["trigger_id"]) or {}
                    trig_name = trig.get("name", f"id={h['trigger_id']}")
                    status_icon = "✓" if h["status"] == "done" else "✗"
                    print(f"  [{h['id']}] {status_icon} {trig_name} @ {h['triggered_at']} ({h['duration_ms']}ms)")
                    if h.get("error"):
                        print(f"       error: {h['error']}")

            elif user_input.lower().startswith("vision analyze "):
                from tools.vision_tools import analyze_image

                path = user_input[15:].strip()
                result = analyze_image(path=path)
                print(f"  {result[:300]}")

            elif user_input.lower().startswith("vision ocr "):
                from tools.vision_tools import ocr_document

                path = user_input[11:].strip()
                result = ocr_document(path)
                print(f"  {result[:300]}")

            elif user_input.lower().startswith("vision video "):
                from tools.vision_tools import analyze_video

                parts = user_input[13:].strip().split()
                path = parts[0] if parts else ""
                ts = parts[1] if len(parts) > 1 else "0"
                result = analyze_video(path, timestamps=ts)
                print(f"  {result[:300]}")

            elif user_input.lower() in ("agent list", "agents"):
                from agent import list_agents

                agents = list_agents()
                if not agents:
                    print("  No agents running.")
                else:
                    for a in agents:
                        ok = a.get("successful_steps", 0)
                        print(f"  [{a['id']}] {a['status']} — {a['goal'][:60]} ({a['step_count']} steps, {ok} ok)")

            elif user_input.lower().startswith("agent stop "):
                from agent import stop_agent

                aid = user_input.split(None, 2)[-1]
                if stop_agent(aid):
                    print(f"  Stopped agent {aid}")
                else:
                    print(f"  Agent {aid} not found")

            elif user_input.lower() in ("graph stats", "graph summary"):
                from graph_memory import get_graph_summary

                print(f"  {get_graph_summary()}")

            elif user_input.lower().startswith("graph extract "):
                from graph_memory import extract_entities_relations

                text = user_input.split(None, 2)[-1]
                results = extract_entities_relations(text)
                print(f"  Extracted {len(results)} relationship(s)")
                for r in results:
                    print(f"    {r['entity1']} --[{r['relationship']}]--> {r['entity2']}")

            elif user_input.lower().startswith("graph neighbors "):
                from graph_memory import query_relationships, search_neighbors

                entity = user_input.split(None, 2)[-1]
                print(f"  {query_relationships(entity)}")
                neighbors = search_neighbors(entity)
                if neighbors:
                    print(f"  Neighbors: {', '.join(n['entity'] for n in neighbors)}")

            elif user_input.lower().startswith("graph search "):
                from graph_memory import hybrid_graph_search

                query = user_input.split(None, 2)[-1]
                results = hybrid_graph_search(query)
                if results:
                    for r in results:
                        neighs = ", ".join(r.get("neighbors", []))
                        print(f"  [{r['entity']}] ({r['entity_type']}) score={r['score']} — neighbors: {neighs}")
                else:
                    print("  No graph matches.")

            elif user_input.lower() in ("mic", "switch mic"):
                input_devices = list_input_devices()
                if input_devices:
                    try:
                        choice = input("  Select mic index: ").strip()
                        if choice.isdigit() and int(choice) in input_devices:
                            _INPUT_DEVICE_INDEX = int(choice)
                            mic = sd.query_devices(_INPUT_DEVICE_INDEX)
                            print(f"  Mic switched to [{_INPUT_DEVICE_INDEX}] {mic.get('name', 'Unknown')}")
                        else:
                            print("  Invalid index.")
                    except (EOFError, KeyboardInterrupt):
                        print("  Mic selection cancelled.")

            elif user_input.lower() in ("/context", "context"):
                try:
                    from brain import get_conversation_context

                    ctx = get_conversation_context()
                    print(f"  State: {ctx['state']}")
                    print(f"  Problem: {ctx['last_problem'][:80] if ctx.get('last_problem') else 'none'}")
                    print(f"  Solution: {ctx['last_solution'][:80] if ctx.get('last_solution') else 'none'}")
                    print(f"  Intent: {ctx.get('last_intent', '?')}")
                    print(f"  Provider: {ctx.get('last_provider', '?')}")
                    print(f"  Tools: {', '.join(ctx.get('last_tools', [])) or 'none'}")
                    print(f"  Fragment awaiting: {ctx.get('fragment_awaiting_context', False)}")
                except Exception as e:
                    print(f"  Context error: {e}")

            elif user_input.lower() in ("/provider-status", "/ps", "/providers"):
                try:
                    status = get_runtime_status()
                    now = datetime.datetime.now().timestamp()
                    print("\n=== Provider Status ===")
                    print(f"{'Provider':<25} {'Status':<20} {'Health':<8} {'Failures':<8}")
                    print("-" * 65)
                    all_providers = set(_provider_health_scores.keys())
                    for p in status.get("providers", {}):
                        all_providers.add(p)
                    for provider in sorted(all_providers):
                        if provider == "huggingface":
                            continue
                        backoff = _provider_backoff_until.get(provider, 0)
                        health = _provider_health_scores.get(provider, 100)
                        failures = _provider_consecutive_failures.get(provider, 0)
                        if backoff > now:
                            remaining = int(backoff - now)
                            status_str = f"BACKED OFF ({remaining}s)"
                        else:
                            status_str = "OK"
                        print(f"  {provider:<25} {status_str:<20} {health:<8} {failures:<8}")
                    print()
                except Exception as e:
                    print(f"  Provider status error: {e}")

            elif user_input.lower().startswith("/mode") or user_input.lower().startswith("/m "):
                parts = user_input.split()
                mode_name = parts[-1].lower() if len(parts) > 1 else ""
                mode_map = {"text": InputMode.TEXT, "t": InputMode.TEXT, "paste": InputMode.PASTE, "p": InputMode.PASTE, "queue": InputMode.QUEUE, "q": InputMode.QUEUE}
                if mode_name in mode_map:
                    _current_mode = mode_map[mode_name]
                    _paste_buffer = []
                    print(f"  Mode switched to {_current_mode.upper()}")
                else:
                    print("  Modes: /mode text|paste|queue  (or /m t|p|q)")
                    print(f"  Current: {_current_mode.upper()}")

            elif user_input.lower() in ("/queue show", "/q show", "queue show"):
                if _queue_buffer:
                    print(f"  Queue ({len(_queue_buffer)} items):")
                    for i, item in enumerate(_queue_buffer, 1):
                        print(f"    {i}. {item[:120]}")
                else:
                    print("  Queue is empty.")

            elif user_input.lower() in ("/queue clear", "/q clear", "queue clear"):
                _queue_buffer.clear()
                print("  Queue cleared.")

            elif user_input.lower() in ("test", "test run", "test logs", "test status", "test report",
                                        "test findings", "test history", "test stop", "self-test", "selftest") \
                    or user_input.lower().startswith(("test confirm ", "test dismiss ", "self-test ", "selftest ")):
                from self_test.agent import handle_command

                print("  " + handle_command(user_input).replace("\n", "\n  "))

            elif user_input.lower() in ("backup", "backup now", "backups", "backup list"):
                import backup

                if user_input.lower() in ("backup", "backup now"):
                    print("  Backing up state...")
                    result = backup.run_backup()
                    print(f"  Backup: {result['backup_dir']}")
                    print(f"  Copied: {', '.join(result['copied']) or 'none'}")
                    if result["skipped"]:
                        print(f"  Skipped: {', '.join(result['skipped'])}")
                    if result["pruned"]:
                        print(f"  Pruned {len(result['pruned'])} old backup(s)")
                else:
                    backups = backup.list_backups()
                    if not backups:
                        print("  No backups yet. Run 'backup' to create one.")
                    else:
                        print(f"  Backups ({len(backups)}):")
                        for b in backups:
                            print(f"    {b['name']} ({b['size_bytes'] / 1024:.0f} KB)")

            elif user_input.lower() in ("health", "health check", "status check"):
                from healthcheck import report_text

                print("  " + report_text().replace("\n", "\n  "))

            elif user_input.lower() in ("selfmod log", "self-mod log", "audit selfmod", "selfmod audit"):
                from action_sandbox import get_self_mod_audit

                entries = get_self_mod_audit(limit=20)
                if not entries:
                    print("  No self-modifications recorded yet.")
                else:
                    print(f"  Self-modification audit ({len(entries)}):")
                    for e in entries:
                        print(f"    [{e['ts'][:19]}] {e['target']} → {e['outcome']}")

            elif user_input.lower().startswith("memory prune") or user_input.lower().startswith("prune memory"):
                from memory import prune_old_memories

                parts = user_input.split()
                days = 30
                for p in parts:
                    if p.isdigit():
                        days = int(p)
                print(f"  Pruning memories older than {days} days...")
                prune_old_memories(days=days)

            elif user_input.lower().strip() in ("session", "sessions", "session list"):
                _print_sessions()

            elif user_input.lower().startswith("session new"):
                from session_store import create_session

                name = user_input[11:].strip()
                meta = create_session(name or None)
                _session_id = meta["id"]
                print(f"  Switched to new session '{meta['name']}' ({meta['id']}).")

            elif user_input.lower().startswith("session switch") or user_input.lower().startswith("session use"):
                from session_store import list_sessions

                target = user_input.split(None, 2)[-1].strip().lower()
                found = None
                for s in list_sessions():
                    if target in (s["id"].lower(), s["name"].lower()):
                        found = s
                        break
                if found:
                    _session_id = found["id"]
                    print(f"  Switched to session '{found['name']}' ({found['id']}).")
                else:
                    print(f"  No session matches '{target}'. Use 'session list' to see sessions.")

            elif user_input.lower().startswith("session rename"):
                from session_store import rename_session

                parts = user_input.split(None, 3)
                if len(parts) < 4:
                    print("  Usage: session rename <id> <new name>")
                else:
                    meta = rename_session(parts[2], parts[3])
                    if not meta:
                        print(f"  Session '{parts[2]}' not found.")
                    else:
                        print(f"  Renamed to '{meta['name']}'.")

            elif user_input.lower().startswith("session delete"):
                from session_store import delete_session

                target = user_input.split(None, 2)[-1].strip()
                if not target:
                    print("  Usage: session delete <id>")
                elif target == _session_id:
                    print("  Can't delete the active session. Switch first (session switch <id>).")
                elif delete_session(target):
                    print(f"  Deleted session '{target}'.")
                else:
                    print(f"  Session '{target}' not found.")

            elif user_input.lower().startswith("session reset"):
                from brain import reset_conversation

                reset_conversation(_session_id)
                print(f"  Reset session '{_session_id}' — history cleared.")

            else:
                handle_input(user_input)


def _print_sessions():
    from session_store import ensure_default_session, list_sessions

    ensure_default_session()
    sessions = list_sessions()
    print(f"  Active session: {_session_id}  (sessions: {len(sessions)})")
    for s in sessions:
        marker = "→ " if s["id"] == _session_id else "  "
        preview = f" — \"{s['preview'][:60]}\"" if s["preview"] else ""
        print(
            f"  {marker}{s['id']:<20} {s['name']:<20} "
            f"{s['message_count']:>4} msgs{preview}"
        )
    print("  Commands: session new <name> | session switch <id> | "
          "session rename <id> <name> | session reset | session delete <id>")


if __name__ == "__main__":
    main()
