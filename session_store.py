"""Persistent named conversation sessions, shared across processes.

Layout (default ~/.jarvis/sessions/, override with JARVIS_SESSIONS_DIR):
    index.json       session metadata (id, name, created, updated)
    <id>.jsonl       one JSON line per message: {"role", "content", "ts"}
    .lock            flock file serializing index mutations across processes

Message appends use O_APPEND single-write lines; index updates are
flock-protected with atomic rename. Safe for concurrent writers
(server.py + terminal.py may both be alive).
"""
import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import IO

_INDEX = "index.json"
_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")

Message = dict


def _sessions_dir() -> str:
    return os.path.expanduser(os.environ.get("JARVIS_SESSIONS_DIR", "~/.jarvis/sessions/"))


def _index_path() -> str:
    return os.path.join(_sessions_dir(), _INDEX)


def _lock_path() -> str:
    return os.path.join(_sessions_dir(), ".lock")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_dir() -> None:
    os.makedirs(_sessions_dir(), exist_ok=True)


def _read_index() -> dict:
    try:
        with open(_index_path(), encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
            return {"sessions": {}}
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {"sessions": {}}


def _write_index(data: dict) -> None:
    _ensure_dir()
    fd, tmp = tempfile.mkstemp(dir=_sessions_dir(), prefix=".index-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _index_path())
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _lock() -> IO[str]:
    _ensure_dir()
    f = open(_lock_path(), "a+", encoding="utf-8")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def _unlock(f) -> None:
    fcntl.flock(f, fcntl.LOCK_UN)
    f.close()


def _safe_id(session_id: str) -> str:
    session_id = (session_id or "").strip().lower().replace(" ", "-")
    if _ID_RE.match(session_id):
        return session_id
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]


def _name_to_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-").lower()
    if not slug:
        slug = "session"
    candidate = slug[:48]
    suffix = 0
    while True:
        candidate_id = candidate if suffix == 0 else f"{candidate[:40]}-{suffix}"
        if candidate_id not in _read_index()["sessions"]:
            return candidate_id
        suffix += 1


def _session_file(session_id: str) -> str:
    return os.path.join(_sessions_dir(), f"{_safe_id(session_id)}.jsonl")


def ensure_default_session() -> dict:
    f = _lock()
    try:
        data = _read_index()
        sessions = data["sessions"]
        if "default" in sessions:
            return sessions["default"]
        now = _now()
        meta = {"id": "default", "name": "Default", "created": now, "updated": now}
        sessions["default"] = meta
        _write_index(data)
        return meta
    finally:
        _unlock(f)


def list_sessions() -> list[dict]:
    _ensure_dir()
    sessions = _read_index()["sessions"]
    out = []
    for meta in sorted(sessions.values(), key=lambda m: m.get("updated", ""), reverse=True):
        item = dict(meta)
        item["message_count"] = _count_lines(_session_file(item["id"]))
        item["preview"] = _preview(item["id"])
        out.append(item)
    return out


def get_session(session_id: str) -> dict | None:
    meta = _read_index()["sessions"].get(_safe_id(session_id))
    if not meta:
        return None
    item = dict(meta)
    item["message_count"] = _count_lines(_session_file(item["id"]))
    item["preview"] = _preview(item["id"])
    return item


def create_session(name: str | None = None) -> dict:
    f = _lock()
    try:
        data = _read_index()
        name = (name or "").strip() or "New session"
        session_id = _name_to_id(name)
        now = _now()
        meta = {"id": session_id, "name": name, "created": now, "updated": now}
        data["sessions"][session_id] = meta
        _write_index(data)
        return dict(meta)
    finally:
        _unlock(f)


def rename_session(session_id: str, name: str) -> dict | None:
    f = _lock()
    try:
        data = _read_index()
        sessions = data["sessions"]
        key = _safe_id(session_id)
        if key not in sessions:
            return None
        sessions[key]["name"] = (name or "").strip() or sessions[key]["name"]
        sessions[key]["updated"] = _now()
        _write_index(data)
        return dict(sessions[key])
    finally:
        _unlock(f)


def delete_session(session_id: str) -> bool:
    f = _lock()
    try:
        data = _read_index()
        sessions = data["sessions"]
        key = _safe_id(session_id)
        if key not in sessions:
            return False
        del sessions[key]
        _write_index(data)
    finally:
        _unlock(f)
    path = _session_file(key)
    if os.path.exists(path):
        os.unlink(path)
    return True


def clear_session(session_id: str) -> None:
    f = _lock()
    try:
        data = _read_index()
        key = _safe_id(session_id)
        if key in data["sessions"]:
            data["sessions"][key]["updated"] = _now()
            _write_index(data)
    finally:
        _unlock(f)
    open(_session_file(key), "w", encoding="utf-8").close()


def load_messages(session_id: str) -> list[Message]:
    path = _session_file(session_id)
    messages = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    messages.append({"role": role, "content": content})
    except FileNotFoundError:
        pass
    return messages


def append_message(session_id: str, role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        raise ValueError(f"invalid role: {role!r}")
    _ensure_dir()
    line = json.dumps(
        {"role": role, "content": content, "ts": _now()}, ensure_ascii=False
    ).encode("utf-8")
    path = _session_file(session_id)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line.decode("utf-8") + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    f = _lock()
    try:
        data = _read_index()
        key = _safe_id(session_id)
        if key in data["sessions"]:
            data["sessions"][key]["updated"] = _now()
            _write_index(data)
    finally:
        _unlock(f)


def _count_lines(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def _preview(session_id: str) -> str:
    messages = load_messages(session_id)
    if not messages:
        return ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"][:120]
    return messages[-1]["content"][:120]
