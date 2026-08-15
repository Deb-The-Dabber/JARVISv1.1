"""brain.process session isolation tests (no providers, no real memory writes).

ask_with_tools / classify_intent are canned; the post-processing hooks that
would hit real providers or the real ChromaDB index are no-ops.
"""
import threading
import time

import pytest

import brain
from session_store import load_messages


@pytest.fixture()
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_SESSIONS_DIR", str(tmp_path))
    return str(tmp_path)


@pytest.fixture()
def hermetic_brain(monkeypatch):
    monkeypatch.setattr(brain, "classify_intent", lambda text: "chat")
    monkeypatch.setattr(brain, "add_to_vector_memory", lambda *a, **k: None)
    monkeypatch.setattr("graph_memory.extract_entities_relations", lambda text: [])
    monkeypatch.setattr("associative_memory.record_concepts", lambda text: None)
    return brain


def _as_pairs(session_id: str) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in load_messages(session_id)]


def test_sessions_are_isolated(hermetic_brain, sessions_dir, monkeypatch):
    monkeypatch.setattr(
        hermetic_brain, "ask_with_tools", lambda text: f"reply-to-{text}"
    )

    hermetic_brain.process("first question", "one")
    hermetic_brain.process("second question", "two")

    assert _as_pairs("one") == [
        ("user", "first question"),
        ("assistant", "reply-to-first question"),
    ]
    assert _as_pairs("two") == [
        ("user", "second question"),
        ("assistant", "reply-to-second question"),
    ]


def test_session_history_reloads_into_context(hermetic_brain, sessions_dir, monkeypatch):
    seen = []

    def fake_ask(text):
        seen.append(list(hermetic_brain.conversation))
        return f"reply-to-{text}"

    monkeypatch.setattr(hermetic_brain, "ask_with_tools", fake_ask)

    hermetic_brain.process("turn one", "sess")
    hermetic_brain.process("turn two", "sess")

    # Second call must see the first turn's history loaded from the store.
    assert seen[-1] == [
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "reply-to-turn one"},
    ]


def test_reset_clears_only_target_session(hermetic_brain, sessions_dir, monkeypatch):
    monkeypatch.setattr(
        hermetic_brain, "ask_with_tools", lambda text: f"reply-to-{text}"
    )

    hermetic_brain.process("hello", "s1")
    hermetic_brain.process("hi", "s2")
    hermetic_brain.reset_conversation("s1")

    assert load_messages("s1") == []
    assert len(load_messages("s2")) == 2


def test_reset_does_not_wedge_on_busy_lock(hermetic_brain, sessions_dir, monkeypatch):
    """reset_conversation must not block forever when a request holds the lock."""
    monkeypatch.setattr(
        hermetic_brain, "ask_with_tools", lambda text: f"reply-to-{text}"
    )

    # Occupy the process lock the way a slow in-flight /ask would.
    held = threading.Event()

    def hold_lock():
        with hermetic_brain._process_lock:
            held.set()
            time.sleep(5)

    t = threading.Thread(target=hold_lock)
    t.start()
    assert held.wait(5)

    start = time.time()
    hermetic_brain.reset_conversation("default", timeout=1.0)
    elapsed = time.time() - start
    assert elapsed < 3.0  # bounded wait, no wedge
    t.join(timeout=10)

    # Persisted history was still cleared even though the buffer was busy.
    assert load_messages("default") == []


def test_concurrent_sessions_do_not_interleave(hermetic_brain, sessions_dir, monkeypatch):
    """Two threads on different sessions must never see each other's turns.

    fake_ask inspects the in-memory conversation mid-turn and sleeps, so an
    interleaving (missing _process_lock) would surface as foreign turns in
    the snapshot or corrupted session files.
    """
    seen_a: list = []
    seen_b: list = []
    snap_lock = threading.Lock()

    def fake_ask(text):
        snapshot = list(hermetic_brain.conversation)
        with snap_lock:
            (seen_a if text.startswith("A-") else seen_b).append(snapshot)
        time.sleep(0.15)
        return f"reply-to-{text}"

    monkeypatch.setattr(hermetic_brain, "ask_with_tools", fake_ask)

    results: dict = {}
    errors: list = []

    def run(session_id, messages):
        try:
            for msg in messages:
                results[msg] = hermetic_brain.process(msg, session_id)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    ta = threading.Thread(
        target=run, args=("sessA", ["A-1", "A-2", "A-3", "A-4", "A-5"])
    )
    tb = threading.Thread(
        target=run, args=("sessB", ["B-1", "B-2", "B-3", "B-4", "B-5"])
    )
    ta.start()
    tb.start()
    ta.join(timeout=60)
    tb.join(timeout=60)

    assert not errors
    assert len(results) == 10

    # Mid-turn snapshots must only ever contain the caller's own session.
    for snapshot in seen_a:
        assert all(
            m["content"].startswith("A-")
            or m["content"].startswith("reply-to-A-")
            for m in snapshot
        ), f"session A saw foreign turns: {snapshot}"
    for snapshot in seen_b:
        assert all(
            m["content"].startswith("B-")
            or m["content"].startswith("reply-to-B-")
            for m in snapshot
        ), f"session B saw foreign turns: {snapshot}"

    # Store files: strict per-session ordering, no cross-contamination.
    a_pairs = _as_pairs("sessA")
    b_pairs = _as_pairs("sessB")
    assert a_pairs == [
        pair
        for i in range(1, 6)
        for pair in [("user", f"A-{i}"), ("assistant", f"reply-to-A-{i}")]
    ]
    assert b_pairs == [
        pair
        for i in range(1, 6)
        for pair in [("user", f"B-{i}"), ("assistant", f"reply-to-B-{i}")]
    ]
