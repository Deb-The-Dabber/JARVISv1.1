"""Headless smoke test for the JARVIS Control Console TUI (tui.py).

Runs inside Textual's built-in headless pilot — no terminal, no camera
(JARVIS_TUI_VISION=0 keeps the vision thread off), no brain round-trips
(sample text routes through the local command surface).
"""

import asyncio
import os

os.environ.setdefault("JARVIS_TUI_VISION", "0")

from textual.widgets import RichLog, Static, TabbedContent  # noqa: E402

from tui import Composer, JarvisConsole, command_hint  # noqa: E402


def test_tui_boots_and_composes():
    app = JarvisConsole(session_id="default")

    async def drive() -> None:
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            assert app.query_one("#header", Static)
            assert app.query_one("#transcript", RichLog)
            assert app.query_one("#activity", RichLog)
            assert app.query_one(Composer)
            # Default tab is MAIN
            tabs = app.query_one(TabbedContent)
            assert "main-tab" in tabs.active
            # Switch across every pane
            for key in ("ctrl+g", "ctrl+p", "ctrl+m", "ctrl+w", "ctrl+t"):
                await pilot.press(key)
                await pilot.pause()
            # Vision tab loads the retina class without starting the camera
            await pilot.press("ctrl+v")
            await pilot.pause()
            status = app.query_one("#vision-status", Static).content
            assert status is not None
            # Composer → local command → transcript refresh on session switch
            composer = app.query_one(Composer)
            composer.focus()
            composer.text = "session new smoke-test"
            await pilot.press("enter")
            await pilot.pause()
            transcript = app.query_one("#transcript", RichLog)
            assert len(transcript.lines) > 0
            # Sink plumbing: a print() line lands in the Activity rail
            app._sink_on_line("hello from a worker thread")
            app._flush_sink()
            rail = app.query_one("#activity", RichLog)
            assert any("hello from a worker thread" in str(line) for line in rail.lines)
            # Fast-local surface is synchronous; full commands run in a worker
            assert app._fast_local("quit") is True
            assert app._fast_local("session list") is False
            assert command_hint("session ls") is not None
            assert command_hint("not a command, please route to brain") is None
            # Vision toggle flips the mode (stream logic exercised with a stub)
            class FakeStream:
                retina_mode = False

            app._vision = FakeStream()
            app.toggle_vision_mode()
            assert app._vision.retina_mode is True
            app._vision = None
            await pilot.pause()

    asyncio.run(drive())


def test_retina_class_loads_without_executing_module():
    from tui import load_artificial_retina

    cls = load_artificial_retina()
    retina = cls(width=320, height=180)
    assert retina.width == 320
    assert retina.height == 180
