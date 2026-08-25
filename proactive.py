import atexit
import datetime
import os
import subprocess
import threading
import time

import psutil
import requests

import priority as priority_engine
from priority import queue_alert
from vector_memory import add_to_vector_memory
from watchlog import clear_old_logs, log_event, log_screen

CHECK_INTERVAL = 30
CPU_THRESHOLD = 80
RAM_THRESHOLD = 85
DISK_FREE_MIN_GB = 10
DOWNLOADS_MAX_GB = 5
DESKTOP_MAX_FILES = 20
APP_OPEN_HOURS = 3
WEATHER_CHECK_MINS = 30
SCREEN_CHECK_MINS = 30
BATTERY_MIN_PCT = 20

USER_LAT = 41.7606
USER_LON = -88.3201
USER_TZ = "America/Chicago"
HOME = os.path.expanduser("~")

_state = {
    "cpu_high_cycles": 0,
    "last_weather_check": 0,
    "last_rain_warning": 0,
    "last_screen_check": 0,
    "warned_disk": False,
    "warned_ram": False,
    "warned_downloads": False,
    "warned_desktop": False,
    "calendar_warned_15": set(),
    "calendar_warned_5": set(),
    "app_open_since": {},
    "app_warned": set(),
    "briefing_done": False,
    "last_evening_summary": None,
    "last_net_warning": 0,
    "caffeinate_proc": None,
    "user_active": True,
}

_speak_fn = None
PRIORITY_CRITICAL = "critical"


def add_alert(alert_type: str, message: str, priority: int = 3):
    queue_alert(alert_type, message, force=priority >= 5)
    try:
        from push_notify import enqueue_message

        if priority >= 3:
            enqueue_message(message, priority=priority)
    except Exception:
        pass


def init(speak_fn, process_fn):
    global _speak_fn
    _speak_fn = speak_fn
    _ = process_fn


def _emit_event(category: str, event: str, detail: str = "", *, alert_type: str | None = None, priority: str = "", speech: str | None = None):
    log_event(category, event, detail)
    vector_text = f"{category}: {event}"
    if detail:
        vector_text += f" | {detail}"
    add_to_vector_memory(vector_text, category="system_event")
    if speech and alert_type:
        queue_alert(alert_type, speech, force=priority == PRIORITY_CRITICAL)


# ─────────────────────────────────────────────
# CAFFEINATE — prevent idle sleep
# ─────────────────────────────────────────────
def start_caffeinate():
    """Prevent Mac from sleeping while Jarvis is running (idempotent — one per process)."""
    if os.getenv("JARVIS_EVAL_MODE") == "1":
        # Test servers: never spawn OS processes that outlive the server.
        return
    existing = _state.get("caffeinate_proc")
    if existing is not None and existing.poll() is None:
        return
    try:
        # Check if on battery
        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged and battery.percent < BATTERY_MIN_PCT:
            print("  Battery low — not caffeinating to preserve power.")
            return
        proc = subprocess.Popen(["caffeinate", "-i"])
        _state["caffeinate_proc"] = proc
        print("  Caffeinate started — Mac will stay awake.")
    except Exception as e:
        print(f"  Caffeinate failed: {e}")


def stop_caffeinate():
    proc = _state.get("caffeinate_proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    _state["caffeinate_proc"] = None


atexit.register(stop_caffeinate)


# ─────────────────────────────────────────────
# USER ACTIVITY DETECTION
# ─────────────────────────────────────────────
def check_user_activity():
    """Detect if user is actively using the Mac."""
    try:
        result = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "HIDIdleTime" in line:
                idle_ns = int(line.split("=")[-1].strip())
                idle_seconds = idle_ns / 1e9
                was_active = _state["user_active"]
                _state["user_active"] = idle_seconds < 300  # Active if idle < 5 min

                # User just came back
                if not was_active and _state["user_active"]:
                    log_event("presence", "User returned", "Was away for a while")
                # User just left
                elif was_active and not _state["user_active"]:
                    log_event("presence", "User went away")
                return idle_seconds
    except Exception:
        pass
    return 0


# ─────────────────────────────────────────────
# MONITORS
# ─────────────────────────────────────────────
def check_cpu():
    cpu = psutil.cpu_percent(interval=1)
    if cpu > CPU_THRESHOLD:
        _state["cpu_high_cycles"] += 1
        if _state["cpu_high_cycles"] == 2:
            procs = sorted(psutil.process_iter(["name", "cpu_percent"]), key=lambda p: p.info["cpu_percent"] or 0, reverse=True)
            top = procs[0].info["name"] if procs else "unknown"
            msg = f"CPU spike at {int(cpu)}% — top process: {top}"
            _emit_event(
                "system",
                "CPU spike",
                msg,
                alert_type="cpu_spike",
                priority="high",
                speech=f"Heads up, CPU is at {int(cpu)} percent. The top process is {top}.",
            )
    else:
        _state["cpu_high_cycles"] = 0


def check_ram():
    ram = psutil.virtual_memory()
    if ram.percent > RAM_THRESHOLD and not _state["warned_ram"]:
        msg = f"RAM at {ram.percent:.0f}%"
        _emit_event(
            "system",
            "High RAM usage",
            msg,
            alert_type="ram_high",
            priority="high",
            speech=f"RAM is at {int(ram.percent)} percent. You might want to close some apps.",
        )
        _state["warned_ram"] = True
    elif ram.percent < RAM_THRESHOLD - 10:
        _state["warned_ram"] = False


def check_disk():
    usage = psutil.disk_usage(HOME)
    free_gb = usage.free / (1024**3)
    if free_gb < DISK_FREE_MIN_GB and not _state["warned_disk"]:
        detail = f"{free_gb:.1f}GB free"
        _emit_event(
            "system",
            "Low disk space",
            detail,
            alert_type="disk_low",
            priority="high",
            speech=f"You're running low on disk space — only {free_gb:.1f} gigabytes free.",
        )
        _state["warned_disk"] = True
    elif free_gb > DISK_FREE_MIN_GB + 5:
        _state["warned_disk"] = False


def check_network():
    try:
        requests.get("https://www.google.com", timeout=3)
    except Exception:
        now = time.time()
        if now - _state["last_net_warning"] > 300:
            _emit_event(
                "network",
                "Internet connection lost",
                "",
                alert_type="internet_down",
                priority=PRIORITY_CRITICAL,
                speech="It looks like you've lost your internet connection.",
            )
            _state["last_net_warning"] = now


def check_downloads_folder():
    folder = os.path.join(HOME, "Downloads")
    if not os.path.exists(folder):
        return
    total = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
    total_gb = total / (1024**3)
    if total_gb > DOWNLOADS_MAX_GB and not _state["warned_downloads"]:
        detail = f"{total_gb:.1f}GB"
        _emit_event(
            "files",
            "Downloads folder large",
            detail,
            alert_type="downloads_large",
            priority="normal",
            speech=f"Your Downloads folder is {total_gb:.1f} gigabytes. Want me to organize it?",
        )
        _state["warned_downloads"] = True
    elif total_gb < DOWNLOADS_MAX_GB - 1:
        _state["warned_downloads"] = False


def check_desktop_clutter():
    desktop = os.path.join(HOME, "Desktop")
    if not os.path.exists(desktop):
        return
    files = [f for f in os.listdir(desktop) if not f.startswith(".")]
    if len(files) > DESKTOP_MAX_FILES and not _state["warned_desktop"]:
        detail = f"{len(files)} items"
        _emit_event(
            "files",
            "Desktop cluttered",
            detail,
            alert_type="desktop_clutter",
            priority="low",
            speech=f"Your Desktop has {len(files)} items on it. Want me to help organize it?",
        )
        _state["warned_desktop"] = True
    elif len(files) <= DESKTOP_MAX_FILES:
        _state["warned_desktop"] = False


def check_weather():
    now = time.time()
    if now - _state["last_weather_check"] < WEATHER_CHECK_MINS * 60:
        return
    _state["last_weather_check"] = now
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={USER_LAT}&longitude={USER_LON}&hourly=precipitation_probability,weathercode&forecast_days=1&timezone={USER_TZ}"
        data = requests.get(url, timeout=10).json()
        times = data["hourly"]["time"]
        precip = data["hourly"]["precipitation_probability"]
        now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:00")
        for i, t in enumerate(times):
            if t == now_str:
                upcoming = precip[i : i + 2]
                if any(p >= 70 for p in upcoming):
                    if now - _state["last_rain_warning"] > 3600:
                        detail = f"{max(upcoming)}% chance"
                        _emit_event(
                            "weather",
                            "Rain incoming",
                            detail,
                            alert_type="rain_incoming",
                            priority="normal",
                            speech="Rain is likely in the next hour or two. Just a heads up.",
                        )
                        _state["last_rain_warning"] = now
                break
    except Exception:
        pass


def check_calendar():
    try:
        script = """
        set output to ""
        set theNow to current date
        set in15 to theNow + 900
        tell application "Calendar"
            set allCals to every calendar
            repeat with aCal in allCals
                set theEvents to (every event of aCal whose start date >= theNow and start date <= in15)
                repeat with e in theEvents
                    set eid to uid of e
                    set etitle to summary of e
                    set estart to start date of e as string
                    set output to output & eid & "|" & etitle & "|" & estart & "~"
                end repeat
            end repeat
        end tell
        return output
        """
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
        out = result.stdout.strip()
        if not out:
            return
        events = [e for e in out.split("~") if "|" in e]
        now = datetime.datetime.now()
        for event_str in events:
            parts = event_str.split("|")
            if len(parts) < 3:
                continue
            eid, title, start_str = parts[0], parts[1], parts[2]
            try:
                start = datetime.datetime.strptime(start_str.strip()[:19], "%A, %B %d, %Y %I:%M:%S")
            except Exception:
                continue
            mins_away = (start - now).total_seconds() / 60
            if 13 <= mins_away <= 16 and eid not in _state["calendar_warned_15"]:
                _emit_event(
                    "calendar",
                    f"Event in 15 min: {title}",
                    alert_type="calendar_15min",
                    priority=PRIORITY_CRITICAL,
                    speech=f"Heads up — {title} starts in about 15 minutes.",
                )
                _state["calendar_warned_15"].add(eid)
            if 4 <= mins_away <= 6 and eid not in _state["calendar_warned_5"]:
                _emit_event(
                    "calendar",
                    f"Event in 5 min: {title}",
                    alert_type="calendar_5min",
                    priority=PRIORITY_CRITICAL,
                    speech=f"{title} starts in 5 minutes.",
                )
                _state["calendar_warned_5"].add(eid)
    except Exception:
        pass


def check_open_apps():
    try:
        script = 'tell application "System Events" to get name of every process whose background only is false'
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
        open_apps = [a.strip() for a in result.stdout.strip().split(",")]
        now = time.time()
        interesting = ["Xcode", "Visual Studio Code", "Terminal", "PyCharm", "Cursor", "Antigravity"]
        for app in open_apps:
            if app in interesting:
                if app not in _state["app_open_since"]:
                    _state["app_open_since"][app] = now
                    log_event("apps", f"{app} opened")
                    add_to_vector_memory(f"apps: {app} opened", category="system_event")
                elif now - _state["app_open_since"][app] > APP_OPEN_HOURS * 3600:
                    if app not in _state["app_warned"]:
                        hours = (now - _state["app_open_since"][app]) / 3600
                        queue_alert("app_open_long", f"You've had {app} open for {hours:.0f} hours. Don't forget to take a break.")
                        _state["app_warned"].add(app)
        for app in list(_state["app_open_since"].keys()):
            if app not in open_apps:
                log_event("apps", f"{app} closed")
                add_to_vector_memory(f"apps: {app} closed", category="system_event")
                del _state["app_open_since"][app]
                _state["app_warned"].discard(app)
    except Exception:
        pass


def check_screen():
    """Periodic screen check — only when user is away."""
    now = time.time()
    if now - _state["last_screen_check"] < SCREEN_CHECK_MINS * 60:
        return
    _state["last_screen_check"] = now
    try:
        from screen import (
            capture_screenshot,
            describe_screen_with_moondream,
            get_frontmost_app,
        )

        app = get_frontmost_app()
        path = capture_screenshot()
        description = describe_screen_with_moondream(path, "Briefly describe what's on screen in 1 sentence. Focus on main content.")
        log_screen(description, app)
        # If something urgent and user is away, log it prominently
        if any(word in description.lower() for word in ["error", "warning", "alert", "failed", "crash", "urgent", "unread"]):
            log_event("screen", f"Notable content in {app}", description[:200])
            add_to_vector_memory(f"screen: Notable content in {app} | {description[:200]}", category="system_event")
    except Exception:
        pass


# ─────────────────────────────────────────────
# STARTUP BRIEFING
# ─────────────────────────────────────────────
def startup_briefing():
    time.sleep(3)
    now = datetime.datetime.now()
    hour = now.hour

    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    queue_alert("startup_briefing", f"{greeting}, Debasish. Jarvis is online.", force=True)

    time.sleep(0.3)

    # Weather
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={USER_LAT}&longitude={USER_LON}&current=temperature_2m,weathercode&temperature_unit=fahrenheit&timezone={USER_TZ}"
        data = requests.get(url, timeout=10).json()["current"]
        conditions = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 61: "rainy", 63: "rainy", 95: "stormy"}
        condition = conditions.get(data["weathercode"], "mixed")
        queue_alert("startup_briefing", f"It's {data['temperature_2m']} degrees and {condition} outside.", force=True)
    except Exception:
        pass

    time.sleep(0.3)

    # Recap hint
    from watchlog import get_events_since

    recent = get_events_since(hours=12)
    away_events = [e for e in recent if e[0] != "system"]
    if len(away_events) > 3:
        queue_alert("startup_briefing", f"While you were away I logged {len(away_events)} events. Ask me what did I miss for a full recap.", force=True)

    time.sleep(0.3)

    # System health
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    disk_free = psutil.disk_usage(HOME).free / (1024**3)
    queue_alert("startup_briefing", f"System: CPU at {int(cpu)} percent, RAM at {int(ram)} percent, {disk_free:.0f} gigabytes free.", force=True)

    time.sleep(0.3)

    # Calendar
    try:
        script = """
        set output to ""
        set theNow to current date
        set endOfDay to theNow - (time of theNow) + 86400
        tell application "Calendar"
            set allCals to every calendar
            repeat with aCal in allCals
                set theEvents to (every event of aCal whose start date >= theNow and start date < endOfDay)
                repeat with e in theEvents
                    set output to output & summary of e & " at " & (start date of e as string) & ", "
                end repeat
            end repeat
        end tell
        return output
        """
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
        out = result.stdout.strip().rstrip(",")
        if out:
            queue_alert("startup_briefing", f"You have the following events today: {out}.", force=True)
        else:
            queue_alert("startup_briefing", "Your calendar is clear for the rest of today.", force=True)
    except Exception:
        pass

    _state["briefing_done"] = True
    log_event("system", "Startup briefing complete")
    clear_old_logs(days=7)


# ─────────────────────────────────────────────
# EVENING SUMMARY
# ─────────────────────────────────────────────
def maybe_evening_summary():
    now = datetime.datetime.now()
    today = now.date()
    if now.hour == 21 and now.minute < 30:
        if _state["last_evening_summary"] != today:
            _state["last_evening_summary"] = today
            from watchlog import get_events_since

            events = get_events_since(hours=12)
            log_event("system", "Evening summary generated")
            queue_alert("evening_summary", f"Good evening. Today I logged {len(events)} events. Ask me 'what happened today?' for details. Have a good night.")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def _monitor_loop():
    threading.Thread(target=startup_briefing, daemon=True).start()
    while True:
        try:
            check_user_activity()
            check_cpu()
            check_ram()
            check_disk()
            check_network()
            check_downloads_folder()
            check_desktop_clutter()
            check_weather()
            check_calendar()
            check_open_apps()
            check_screen()
            maybe_evening_summary()
        except Exception:
            pass
        time.sleep(CHECK_INTERVAL)


def start():
    # Start priority engine first
    priority_engine.init(_speak_fn, lambda: _state["user_active"])
    priority_engine.start()

    start_caffeinate()
    t = threading.Thread(target=_monitor_loop, daemon=True)
    t.start()
    print("  Proactive engine started.")
