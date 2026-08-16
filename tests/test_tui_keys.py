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


def test_on_demand_refresh_hides_under_interval_churn():
    """The pane refreshers must not be wired to periodic intervals anymore."""
    import tui

    src = tui.__file__
    text = open(src, encoding="utf-8").read()
    assert "set_interval" in text
    for stale in ("_refresh_system_pane", "_refresh_brain_pane", "_refresh_memory_pane",
                  "_refresh_workflows_pane", "_refresh_tools_pane"):
        assert stale not in text, f"stale interval refresher still present: {stale}"


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
