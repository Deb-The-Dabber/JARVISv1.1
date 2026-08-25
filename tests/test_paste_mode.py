"""Unit tests for the classic REPL paste/batch input handling.

Covers `_read_batched` (terminal paste bursts merge into one message),
the paste-mode submit/cancel predicates, and `_submit_paste_buffer`
(one message per submit, summarize gate, literal "/"-content lines).
"""

import builtins
import sys

import pytest

import terminal


def _patch_input(monkeypatch, lines):
    """Replace builtins.input with a queue of scripted lines."""
    calls = []

    def fake_input(prompt=""):
        calls.append(prompt)
        if not lines:
            raise AssertionError(f"input() called past scripted lines (prompt={prompt!r})")
        return lines.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)
    return calls


def _patch_tty(monkeypatch, isatty=True, pending_seq=None):
    """Script stdin tty-ness and the pending-data signal."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: isatty)
    seq = list(pending_seq or [])

    def fake_pending():
        return seq.pop(0) if seq else False

    monkeypatch.setattr(terminal, "_stdin_pending", fake_pending)


@pytest.fixture(autouse=True)
def _fast_settle(monkeypatch):
    """Keep the settle window tiny so burst tests don't sleep."""
    monkeypatch.setattr(terminal, "_PASTE_SETTLE_S", 0.001)


class TestReadBatched:
    def test_single_line_no_burst(self, monkeypatch):
        _patch_input(monkeypatch, ["hello jarvis"])
        _patch_tty(monkeypatch, isatty=True, pending_seq=[])
        text, is_burst, truncated = terminal._read_batched("You → ")
        assert text == "hello jarvis"
        assert is_burst is False
        assert truncated is False

    def test_piped_stdin_never_batches(self, monkeypatch):
        _patch_input(monkeypatch, ["hello"])
        _patch_tty(monkeypatch, isatty=False, pending_seq=[True, True, True])
        text, is_burst, truncated = terminal._read_batched("You → ")
        assert text == "hello"
        assert is_burst is False
        assert truncated is False

    def test_burst_merges_into_single_unit(self, monkeypatch):
        _patch_input(monkeypatch, ["line one", "line two", "", "line four"])
        _patch_tty(monkeypatch, isatty=True, pending_seq=[True, True, True, True])
        text, is_burst, truncated = terminal._read_batched("Paste → ")
        assert is_burst is True
        assert truncated is False
        assert text == "line one\nline two\n\nline four"

    def test_burst_preserves_blank_lines(self, monkeypatch):
        _patch_input(monkeypatch, ["first", "", "", "last"])
        _patch_tty(monkeypatch, isatty=True, pending_seq=[True, True, True, True])
        text, _, _ = terminal._read_batched("Paste → ")
        assert "\n\n" in text
        assert text.startswith("first\n")
        assert text.endswith("last")

    def test_eof_during_drain_stops_burst(self, monkeypatch):
        def fake_input(prompt=""):
            if prompt == "Paste → ":
                return "first"
            raise EOFError

        monkeypatch.setattr(builtins, "input", fake_input)
        _patch_tty(monkeypatch, isatty=True, pending_seq=[True, True])
        text, is_burst, truncated = terminal._read_batched("Paste → ")
        assert is_burst is True
        assert text == "first"
        assert truncated is False

    def test_truncation_cap(self, monkeypatch):
        monkeypatch.setattr(terminal, "_PASTE_MAX_CHARS", 15)
        _patch_input(monkeypatch, ["0123456789", "abcdefghij"])
        _patch_tty(monkeypatch, isatty=True, pending_seq=[True, True])
        text, is_burst, truncated = terminal._read_batched("You → ")
        assert is_burst is True
        assert truncated is True
        assert text == "0123456789"

    def test_first_line_carries_prompt_only(self, monkeypatch):
        calls = _patch_input(monkeypatch, ["one", "two"])
        _patch_tty(monkeypatch, isatty=True, pending_seq=[True, True])
        terminal._read_batched("Paste → ")
        assert calls == ["Paste → ", ""]


class TestPastePredicates:
    @pytest.mark.parametrize(
        "line", ["/", "/go", "go", "GO", "/Go"],
    )
    def test_submit_lines(self, line):
        assert terminal._is_submit_line(line)

    @pytest.mark.parametrize(
        "line", ["/mode t", "//comment", "/etc/hosts", "g o"],
    )
    def test_non_submit_lines(self, line):
        assert not terminal._is_submit_line(line)

    @pytest.mark.parametrize(
        "line", ["/cancel", "/discard", "cancel", "discard", "  /CANCEL  "],
    )
    def test_cancel_lines(self, line):
        assert terminal._is_cancel_line(line)

    @pytest.mark.parametrize(
        "line", ["/", "go", "/mode t", "//paste", ""],
    )
    def test_non_cancel_lines(self, line):
        assert not terminal._is_cancel_line(line)


class TestSubmitPasteBuffer:
    def test_joins_buffer_into_one_message(self, monkeypatch):
        sent = []
        monkeypatch.setattr(terminal, "handle_input", lambda t: sent.append(t))
        terminal._paste_buffer = ["a", "b", "c"]
        terminal._submit_paste_buffer()
        assert sent == ["a\nb\nc"]
        assert terminal._paste_buffer == []

    def test_empty_buffer_noop(self, monkeypatch):
        sent = []
        monkeypatch.setattr(terminal, "handle_input", lambda t: sent.append(t))
        terminal._paste_buffer = []
        terminal._submit_paste_buffer()
        assert sent == []

    def test_under_threshold_not_summarized(self, monkeypatch):
        sent = []
        summarized = []
        monkeypatch.setattr(terminal, "handle_input", lambda t: sent.append(t))
        monkeypatch.setattr(terminal, "_summarize_paste", lambda c: summarized.append(c) or "SUM")
        terminal._paste_buffer = ["some short pasted text"]
        terminal._submit_paste_buffer()
        assert sent == ["some short pasted text"]
        assert summarized == []

    def test_over_threshold_summarized_once(self, monkeypatch):
        sent = []
        monkeypatch.setattr(terminal, "handle_input", lambda t: sent.append(t))
        monkeypatch.setattr(terminal, "_summarize_paste", lambda c: "DIGEST")
        monkeypatch.setattr(terminal, "_PASTE_SUMMARIZE_MIN", 10)
        terminal._paste_buffer = ["0123456789", "0123456789"]
        terminal._submit_paste_buffer()
        assert sent == ["DIGEST"]
