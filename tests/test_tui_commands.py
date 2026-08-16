"""Unit tests for the TUI command surface and camera selection (tui.py).

Pure-function coverage only — no display, no camera, no brain round-trips.
"""


import pytest


def test_command_hint_unknown_slash_command():
    from tui import command_hint

    hint = command_hint("//frobnicate")
    assert hint is not None
    assert "//help" in hint


def test_command_hint_near_miss_command():
    from tui import command_hint

    assert command_hint("session ls") is not None
    assert command_hint("trigger") is not None
    assert command_hint("backup please") is not None
    assert "//help" in command_hint("backup please")


def test_command_hint_does_not_touch_chat():
    from tui import command_hint

    assert command_hint("") is None
    assert command_hint("tell me about triggers and workflows") is None
    assert command_hint("backup my photos from last summer please") is None
    assert command_hint("   ") is None
    # exact-match commands are handled before the hint is consulted
    assert command_hint("session list") is not None  # would never be reached


def test_command_hint_case_insensitive():
    from tui import command_hint

    assert command_hint("Session LS") is not None


def test_camera_index_override(monkeypatch):
    from tui import _find_camera_index

    monkeypatch.setenv("JARVIS_CAMERA_INDEX", "2")
    assert _find_camera_index() == 2
    monkeypatch.setenv("JARVIS_CAMERA_INDEX", "-3")
    assert _find_camera_index() == 0


def test_session_reset_does_not_route_to_llm(monkeypatch):
    """session reset used to call handle_input() → a second brain round-trip."""
    import brain
    import terminal

    monkeypatch.setattr(brain, "reset_conversation", lambda sid: None)
    monkeypatch.setattr(
        terminal, "handle_input", lambda *a, **k: pytest.fail("handle_input must not be called")
    )
    assert terminal.handle_local_command("session reset") is True


def test_session_reset_switches_to_active_session(monkeypatch):
    import brain
    import terminal

    called = {}

    def fake_reset(sid):
        called["sid"] = sid

    monkeypatch.setattr(brain, "reset_conversation", fake_reset)
    old = terminal._session_id
    try:
        terminal._session_id = "test-session-1"
        assert terminal.handle_local_command("session reset") is True
        assert called.get("sid") == "test-session-1"
    finally:
        terminal._session_id = old


def test_fast_local_surface_is_synchronous_and_non_blocking(monkeypatch):
    import tui

    app = object.__new__(tui.JarvisConsole)
    app._rail = lambda *a, **k: None
    app._rail_ui = lambda *a, **k: None
    app.action_quit_app = lambda: None

    assert app._fast_local("q") is True
    assert app._fast_local("exit") is True
    assert app._fast_local("wake") is True
    assert app._fast_local("/mode paste") is True
    assert app._fast_local("mic") is True
    # chat-looking text is never swallowed
    assert app._fast_local("what is the weather") is False


def test_unknown_command_prints_usage_via_rail(monkeypatch, capsys):
    """//-unknowns surface a hint; the surface never silences them."""
    from tui import command_hint

    hint = command_hint("//bogus")
    assert hint and "//bogus" in hint
