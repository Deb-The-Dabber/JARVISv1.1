"""Unit tests for session_store: CRUD, path safety, cross-process concurrency."""
import os
import subprocess
import sys

import pytest

import session_store as store


@pytest.fixture()
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_SESSIONS_DIR", str(tmp_path))
    return str(tmp_path)


def test_default_session_is_created(sessions_dir):
    meta = store.ensure_default_session()
    assert meta["id"] == "default"
    assert os.path.exists(os.path.join(sessions_dir, "index.json"))
    store.append_message("default", "user", "hello")
    assert os.path.exists(os.path.join(sessions_dir, "default.jsonl"))


def test_append_and_load_roundtrip(sessions_dir):
    store.ensure_default_session()
    store.append_message("default", "user", "hello")
    store.append_message("default", "assistant", "hi there")
    messages = store.load_messages("default")
    assert messages == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_invalid_role_rejected(sessions_dir):
    store.ensure_default_session()
    with pytest.raises(ValueError):
        store.append_message("default", "system", "nope")


def test_create_list_rename_delete(sessions_dir):
    meta = store.create_session("School Project")
    assert meta["id"] == "school-project"
    store.append_message(meta["id"], "user", "what's due")
    sessions = store.list_sessions()
    by_id = {s["id"]: s for s in sessions}
    assert by_id[meta["id"]]["name"] == "School Project"
    assert by_id[meta["id"]]["message_count"] == 1
    assert by_id[meta["id"]]["preview"] == "what's due"

    renamed = store.rename_session(meta["id"], "College")
    assert renamed["name"] == "College"
    assert store.rename_session("ghost", "x") is None

    assert store.delete_session(meta["id"]) is True
    assert store.delete_session(meta["id"]) is False
    assert not os.path.exists(os.path.join(sessions_dir, "school-project.jsonl"))
    assert store.get_session(meta["id"]) is None


def test_create_duplicate_name_gets_suffix(sessions_dir):
    a = store.create_session("Work")
    b = store.create_session("Work")
    assert a["id"] != b["id"]
    assert a["id"].startswith("work")


def test_clear_session_empties_history(sessions_dir):
    store.ensure_default_session()
    store.append_message("default", "user", "hello")
    store.clear_session("default")
    assert store.load_messages("default") == []


def test_path_traversal_ids_are_sanitized(sessions_dir):
    evil = "../../../etc/passwd"
    safe = store._safe_id(evil)
    assert "/" not in safe and ".." not in safe
    store.append_message(evil, "user", "x")
    assert os.path.exists(os.path.join(sessions_dir, f"{safe}.jsonl"))


def test_bad_lines_are_skipped(sessions_dir):
    store.ensure_default_session()
    path = os.path.join(sessions_dir, "default.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write("not json\n")
        f.write('{"role": "user"}\n')
        f.write('{"role": "user", "content": "ok"}\n')
    assert store.load_messages("default") == [{"role": "user", "content": "ok"}]


def test_concurrent_append_two_processes(sessions_dir):
    """Two OS processes appending concurrently must not lose or corrupt lines."""
    store.ensure_default_session()
    child_code = (
        "import os, sys, json\n"
        "sys.path.insert(0, os.environ['JARVIS_ROOT'])\n"
        "import session_store as s\n"
        "for i in range(20):\n"
        "    s.append_message('default', 'user', f'child-{i}')\n"
    )
    env = os.environ.copy()
    env["JARVIS_SESSIONS_DIR"] = sessions_dir
    env["JARVIS_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    child = subprocess.Popen([sys.executable, "-c", child_code], env=env)
    for i in range(20):
        store.append_message("default", "user", f"parent-{i}")
    child.wait(timeout=30)
    assert child.returncode == 0

    messages = store.load_messages("default")
    assert len(messages) == 40
    contents = [m["content"] for m in messages]
    assert sorted(c for c in contents if c.startswith("child-")) == sorted(
        f"child-{i}" for i in range(20)
    )
    assert sorted(c for c in contents if c.startswith("parent-")) == sorted(
        f"parent-{i}" for i in range(20)
    )
