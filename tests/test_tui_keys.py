"""Regression tests: TUI keymap must not collide with the composer's
TextArea bindings, and info panes must refresh off the UI thread."""

import os
import threading
import time

os.environ.setdefault("JARVIS_TUI_VISION", "0")

from textual.widgets import TabbedContent, TextArea  # noqa: E402

from tui import Composer, JarvisConsole  # noqa: E402

# Keys TextArea (and our Composer subclass) consumes — pane switches must
# never use these, otherwise the shortcuts die while typing.
_TEXTAREA_KEYS = {
    k
    for b in list(TextArea.BINDINGS) + list(Composer.BINDINGS)
    for k in b.key.split(",")
}


def test_pane_switches_do_not_collide_with_composer_keys():
    console_keys = {k for b in JarvisConsole.BINDINGS for k in b.key.split(",")}
    overlap = console_keys & _TEXTAREA_KEYS
    # ctrl+c is deliberately NOT in the console bindings: Composer owns it
    # (copy-with-text / cancel-when-empty), so no collision can leak through.
    assert not overlap, f"keymap collision with composer: {sorted(overlap)}"


def test_pane_switches_exist_for_every_tab():
    keys = {k for b in JarvisConsole.BINDINGS for k in b.key.split(",")}
    assert {"ctrl+1", "ctrl+2", "ctrl+3", "ctrl+4", "ctrl+5", "ctrl+6", "ctrl+7"} <= keys
    assert "ctrl+w" not in keys and "ctrl+v" not in keys


def test_pane_refresh_happens_off_the_ui_thread():
    """Pane text must be computed on a worker thread — psutil.cpu_percent
    (interval=1) and SQLite reads must never block the Textual loop."""
    app = object.__new__(JarvisConsole)
    app._pane_refreshing = set()
    app._PANE_TEXT_FN = dict(JarvisConsole._PANE_TEXT_FN)
    app._apply_pane_ui = lambda *a, **k: None
    app.call_from_thread = lambda *a, **k: None
    ui_thread = threading.get_ident()
    computed_on: dict[str, int] = {}

    def fake_text(self):
        computed_on["id"] = threading.get_ident()
        return "data"

    app._PANE_TEXT_FN["system"] = "_fake"
    JarvisConsole._fake = fake_text

    app._reveal_pane("system-tab")
    for _ in range(100):
        if "id" in computed_on:
            break
        time.sleep(0.02)
    assert computed_on.get("id") is not None
    assert computed_on["id"] != ui_thread


def test_pane_apply_respects_active_tab():
    """Hidden panes must never receive updates (skip the UI write entirely)."""
    app = object.__new__(JarvisConsole)
    app._pane_refreshing = set()
    writes: list[str] = []

    class Recorder:
        def update(self, text):
            writes.append(text)

    class FakeTabs:
        def __init__(self, active):
            self.active = active

    def fake_query(selector):
        if selector == TabbedContent:
            return FakeTabs(active[0])
        return Recorder()

    app.query_one = fake_query
    active = ["main-tab"]
    app._apply_pane_ui("system", "data-while-hidden")
    assert writes == []
    active[0] = "system-tab"
    app._apply_pane_ui("system", "data-while-open")
    assert writes == ["data-while-open"]
    # Textual 8.2 may also report the active tab as the bare key ("system")
    active[0] = "system"
    app._apply_pane_ui("system", "data-key-form")
    assert writes[-1] == "data-key-form"


def test_pane_content_updates_after_tab_activation():
    """Real TabbedContent: opening SYSTEM/BRAIN/MEMORY/WORKFLOWS/TOOLS must
    replace the placeholder with computed content (regression: active-tab
    id mismatch left panes stuck on 'Loading …')."""
    import asyncio

    from textual.widgets import Static, TabbedContent

    app = JarvisConsole(session_id="default")
    placeholders = {
        "system": "Loading system…",
        "brain": "Loading providers…",
        "memory": "Loading memory…",
        "workflows": "Loading workflows…",
        "tools": "Loading tools…",
    }

    async def drive():
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            for num, key in (("ctrl+2", "system"), ("ctrl+3", "brain"), ("ctrl+4", "memory"),
                             ("ctrl+6", "workflows"), ("ctrl+7", "tools")):
                await pilot.press(num)
                await pilot.pause(2.0)  # allow the off-thread refresh to land
                pane = app.query_one(f"#{key}-pane", Static)
                content = str(pane.render())
                assert content != placeholders[key], f"{key} pane stuck on placeholder"
                assert tabs.active == f"{key}-tab", (
                    f"active must be the pane id form, got {tabs.active!r}"
                )
            # Mouse path: clicking the tab header must also refresh the pane.
            await pilot.click("#--content-tab-system-tab")
            await pilot.pause(2.0)
            pane = app.query_one("#system-pane", Static)
            assert str(pane.render()) != placeholders["system"], "click path stuck"
            assert tabs.active == "system-tab", f"click active {tabs.active!r}"

    asyncio.run(drive())


def test_on_demand_refresh_hides_under_interval_churn():
    """Info panes must refresh on demand, NOT on periodic intervals — the
    one deliberate exception is the SYSTEM pane's mini Activity Monitor
    (_monitor_tick, live every 2s while its tab is open)."""
    import tui

    src = tui.__file__
    text = open(src, encoding="utf-8").read()
    assert "set_interval" in text
    for stale in ("_refresh_system_pane", "_refresh_brain_pane", "_refresh_memory_pane",
                  "_refresh_workflows_pane", "_refresh_tools_pane"):
        assert stale not in text, f"stale interval refresher still present: {stale}"


def test_system_monitor_modes_and_live_tick():
    """SYSTEM pane is a mini Activity Monitor: c/m/d/n/e switch the metric
    (only while the pane owns focus), and the live tick refreshes only
    while the pane is open."""
    import asyncio

    from tui import SystemMonitor

    app = JarvisConsole(session_id="default")

    async def drive():
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            mon = app.query_one(SystemMonitor)
            tabs = app.query_one(TabbedContent)

            await pilot.press("ctrl+2")
            await pilot.pause(0.2)
            assert tabs.active == "system-tab"
            assert mon.has_focus, "monitor must own focus so c/m/d/n/e switch modes"

            # mode keys must land on the monitor, not the composer
            await pilot.press("m")
            assert mon.mode == "mem"
            await pilot.press("c")
            assert mon.mode == "cpu"
            await pilot.press("d")
            assert mon.mode == "disk"
            await pilot.press("e")
            assert mon.mode == "energy"
            await pilot.press("n")
            assert mon.mode == "net"

            # a mode switch triggers an immediate refresh while open
            for _ in range(100):
                if not app._pane_refreshing:
                    break
                await pilot.pause(0.02)
            app._monitor_tick()
            assert "system" in app._pane_refreshing
            for _ in range(100):
                if "system" not in app._pane_refreshing:
                    break
                await pilot.pause(0.02)
            text = str(mon.render())
            assert "NET" in text and "pid" in text

            # while on MAIN the tick must not touch the pane
            await pilot.press("ctrl+1")
            await pilot.pause(0.2)
            app._monitor_tick()
            assert app._pane_refreshing == set()

    asyncio.run(drive())


def test_clipboard_bridge_pastes_os_clipboard():
    """super+v and ctrl+v must insert the real OS clipboard (Textual's own
    clipboard is an internal string; super+v is unbound upstream, which is
    why cmd+v used to type a bare 'v'). Copies must also reach the OS."""
    import asyncio

    import pytest

    try:
        import pyperclip
    except ImportError:
        pytest.skip("pyperclip not installed")
    try:
        saved = pyperclip.paste()
        pyperclip.copy("probe-on-clipboard")
        assert pyperclip.paste() == "probe-on-clipboard"
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no OS clipboard available: {e}")

    from textual import events

    from tui import Composer

    app = JarvisConsole(session_id="default")

    async def drive():
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.focus()
            await pilot.press("super+v")
            await pilot.pause()
            assert "probe-on-clipboard" in composer.text, "super+v must paste OS clipboard"
            composer.text = ""
            await pilot.press("ctrl+v")
            await pilot.pause()
            assert "probe-on-clipboard" in composer.text, "ctrl+v must paste OS clipboard"
            # a real terminal paste (driver → app → focused widget) inserts once
            composer.text = ""
            app.post_message(events.Paste("paste-event-text"))
            await pilot.pause()
            assert composer.text == "paste-event-text", repr(composer.text)
            # copy reaches the OS clipboard
            composer.text = "select-me"
            from textual.widgets.text_area import Selection

            composer.selection = Selection((0, 0), (0, 6))
            await pilot.press("super+c")
            await pilot.pause()
            assert pyperclip.paste() == "select", "super+c must push to OS clipboard"

    try:
        asyncio.run(drive())
    finally:
        try:
            pyperclip.copy(saved)
        except Exception:  # noqa: BLE001
            pass


def test_ctrl_c_cancels_when_composer_empty_but_copies_with_text():
    import asyncio

    from tui import JarvisConsole

    app = JarvisConsole(session_id="default")

    async def drive():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.focus()
            app._busy = True
            app._cancel = False
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app._cancel is True, "empty composer ctrl+c should cancel"
            # with text, TextArea's copy behavior wins (no cancel, text stays)
            app._cancel = False
            composer.text = "hello"
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app._cancel is False
            assert composer.text == "hello"
            app._busy = False
            app._cancel = False

    asyncio.run(drive())
