"""PTY regression tests for the non-canonical tty line reader.

The whole point of this module: macOS/BSD canonical-mode ttys cap a single
line at 1024 bytes and silently discard the excess, so a long one-line
Cmd+V paste was truncated before Jarvis ever read it (verified with a
3000-char line -> 1024 chars received). `_RawTTYLineReader` reads in
non-canonical mode so paste length is bounded only by _PASTE_MAX_CHARS.
"""

import os
import sys
import termios
import threading

import pytest

import terminal


@pytest.fixture
def pty_pair():
    master, slave = os.openpty()
    yield master, slave
    for fd in (master, slave):
        try:
            os.close(fd)
        except OSError:
            pass


def _feed(master, data: bytes, chunk: int = 512):
    """Write paste data to the pty master from a thread.

    A blocking os.write() to a pty master stalls once the slave-side input
    queue fills (~4KB on macOS), so a huge paste must be streamed WHILE the
    reader drains the slave side — exactly what happens when a human pastes.
    Returns the feeder thread.
    """
    t = threading.Thread(target=_write_all, args=(master, data, chunk), daemon=True)
    t.start()
    return t


def _write_all(master, data: bytes, chunk: int):
    view = memoryview(data)
    for i in range(0, len(view), chunk):
        os.write(master, view[i : i + chunk])
        if i + chunk < len(view):
            threading.Event().wait(0.005)


class TestRawTTYLineReader:
    def test_long_single_line_survives_whole(self, pty_pair):
        """The bug: >1024-char single line must arrive intact."""
        master, slave = pty_pair
        payload = "A" * 5000
        with terminal._RawTTYLineReader(slave) as reader:
            _feed(master, (payload + "\n").encode())
            line = reader.read_line("")
        assert len(line) == 5000
        assert line == payload

    def test_exactly_around_old_cap(self, pty_pair):
        master, slave = pty_pair
        for n in (1023, 1024, 1025, 4096, 4097):
            payload = "B" * n
            with terminal._RawTTYLineReader(slave) as reader:
                _feed(master, (payload + "\n").encode())
                line = reader.read_line("")
            assert len(line) == n

    def test_paste_with_newlines_preserved(self, pty_pair):
        """One chunk with several lines: each line must survive, in order."""
        master, slave = pty_pair
        payload = "line one\nline two\n\nline four\n"
        with terminal._RawTTYLineReader(slave) as reader:
            _feed(master, payload.encode(), chunk=64)
            lines = [reader.read_line("") for _ in range(4)]
        assert lines == ["line one", "line two", "", "line four"]

    def test_utf8_multibyte_preserved(self, pty_pair):
        master, slave = pty_pair
        payload = "héllo wörld — 日本語 ✓"
        with terminal._RawTTYLineReader(slave) as reader:
            _feed(master, (payload + "\n").encode("utf-8"))
            line = reader.read_line("")
        assert line == payload

    def test_backspace_edits_ascii(self, pty_pair):
        master, slave = pty_pair
        with terminal._RawTTYLineReader(slave) as reader:
            os.write(master, b"hel\x7flo\n")
            assert reader.read_line("") == "helo"

    def test_backspace_edits_multibyte(self, pty_pair):
        """Backspace must remove a full UTF-8 codepoint, not one byte."""
        master, slave = pty_pair
        with terminal._RawTTYLineReader(slave) as reader:
            os.write(master, "café\x7f\n".encode())
            assert reader.read_line("") == "caf"

    def test_ctrl_c_raises_keyboard_interrupt(self, pty_pair):
        master, slave = pty_pair
        with terminal._RawTTYLineReader(slave) as reader:
            os.write(master, b"partial\x03")
            with pytest.raises(KeyboardInterrupt):
                reader.read_line("")

    def test_ctrl_d_flushes_partial_line(self, pty_pair):
        master, slave = pty_pair
        with terminal._RawTTYLineReader(slave) as reader:
            os.write(master, b"flushed\x04")
            assert reader.read_line("") == "flushed"

    def test_bare_ctrl_d_raises_eof(self, pty_pair):
        master, slave = pty_pair
        with terminal._RawTTYLineReader(slave) as reader:
            os.write(master, b"\x04")
            with pytest.raises(EOFError):
                reader.read_line("")

    def test_crlf_paste_handled(self, pty_pair):
        """CRLF pastes: the tty maps \\r to \\n (ICRNL), so CRLF yields a
        blank line between entries — exactly like a canonical terminal."""
        master, slave = pty_pair
        with terminal._RawTTYLineReader(slave) as reader:
            os.write(master, b"one\r\ntwo\r\n")
            assert reader.read_line("") == "one"
            assert reader.read_line("") == ""
            assert reader.read_line("") == "two"

    def test_termios_restored_on_exit(self, pty_pair):
        master, slave = pty_pair
        with terminal._RawTTYLineReader(slave) as reader:
            os.write(master, b"x\n")
            reader.read_line("")
        attrs = termios.tcgetattr(slave)
        assert attrs[3] & termios.ICANON  # canonical restored
        assert attrs[3] & termios.ECHO  # echo restored
        assert attrs[3] & termios.ISIG  # signal chars restored


class _FakeStdin:
    """Wrap a pty slave fd so _read_batched sees a real tty stdin."""

    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        return self._fd

    def isatty(self):
        return os.isatty(self._fd)


def _slave_raw(slave):
    """Put the slave in non-canonical/no-echo mode, as the reader does."""
    attrs = termios.tcgetattr(slave)
    attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
    termios.tcsetattr(slave, termios.TCSANOW, attrs)


class TestReadBatchedRawPath:
    def test_long_single_line_through_read_batched(self, monkeypatch, pty_pair):
        """End-to-end: _read_batched must not truncate a >1024-char line."""
        master, slave = pty_pair
        _slave_raw(slave)
        monkeypatch.setattr(sys, "stdin", _FakeStdin(slave))
        payload = "C" * 9000
        _feed(master, (payload + "\n").encode())
        text, is_burst, truncated = terminal._read_batched("You → ")
        assert truncated is False
        assert text == payload

    def test_burst_merges_via_raw_path(self, monkeypatch, pty_pair):
        master, slave = pty_pair
        _slave_raw(slave)
        monkeypatch.setattr(sys, "stdin", _FakeStdin(slave))
        monkeypatch.setattr(terminal, "_PASTE_SETTLE_S", 0.05)
        _feed(master, b"line one\nline two\n\nline four\n", chunk=64)
        text, is_burst, truncated = terminal._read_batched("Paste → ")
        assert is_burst is True
        assert truncated is False
        assert text == "line one\nline two\n\nline four"

    def test_paste_cap_still_enforced(self, monkeypatch, pty_pair):
        master, slave = pty_pair
        _slave_raw(slave)
        monkeypatch.setattr(sys, "stdin", _FakeStdin(slave))
        monkeypatch.setattr(terminal, "_PASTE_SETTLE_S", 0.05)
        monkeypatch.setattr(terminal, "_PASTE_MAX_CHARS", 15)
        _feed(master, b"0123456789\nabcdefghij\n", chunk=64)
        text, is_burst, truncated = terminal._read_batched("You → ")
        assert is_burst is True
        assert truncated is True
        assert text == "0123456789"


class TestRawToggle:
    def test_tty_raw_flag_off_uses_input(self, monkeypatch, pty_pair):
        """JARVIS_TTY_RAW=0 must restore the canonical input() path."""
        master, slave = pty_pair
        monkeypatch.setattr(sys, "stdin", _FakeStdin(slave))
        monkeypatch.setattr(terminal, "_TTY_RAW", False)
        calls = []

        def fake_input(prompt=""):
            calls.append(prompt)
            return "scripted"

        monkeypatch.setattr("builtins.input", fake_input)
        text, is_burst, _ = terminal._read_batched("You → ")
        assert text == "scripted"
        assert calls == ["You → "]
