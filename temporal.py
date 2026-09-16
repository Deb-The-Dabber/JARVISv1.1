"""Authoritative temporal context for JARVIS (V5 task/session rewrite).

Provides the single source of truth for "what date/time is it" that is
injected into the internal control-plane LLM calls (planner / evaluator /
replanner / agent loop), which previously received no date at all and fell
back to the model's training-data assumptions (e.g. "winter break" -> 2025).

Also detects explicit temporal corrections from the user ("the date is
wrong", "this is 2026, not 2025") and persists a per-session override so the
corrected date is authoritative until the user changes it again.
"""

import datetime
import json
import os
import re
import threading
from typing import Dict, Optional


_OVERRIDES_FILE = os.getenv(
    "JARVIS_TEMPORAL_FILE", os.path.expanduser("~/.jarvis/temporal_context.json")
)
_lock = threading.RLock()
_overrides: Dict[str, Dict] = {}


def _load() -> None:
    global _overrides
    try:
        if os.path.exists(_OVERRIDES_FILE):
            with open(_OVERRIDES_FILE, "r", encoding="utf-8") as f:
                _overrides = json.load(f)
    except Exception:
        _overrides = {}


def _save() -> None:
    try:
        os.makedirs(os.path.dirname(_OVERRIDES_FILE), exist_ok=True)
        with open(_OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(_overrides, f, indent=2)
    except Exception:
        pass


_load()


def _local_tz() -> str:
    try:
        import time as _time

        offset = _time.localtime().tm_gmtoff
        name = _time.tzname[0] if _time.daylight and _time.localtime().tm_isdst else _time.tzname[0]
        return f"{name} (UTC{offset // 3600:+d})"
    except Exception:
        return "local"


def now_context(session_id: str = "default") -> str:
    """Authoritative date block for prompt injection.

    Uses the real system date unless a user-corrected override exists for
    this session. Returns a compact block the planner/evaluator can read.
    """
    with _lock:
        ov = _overrides.get(session_id) or {}
    base = datetime.datetime.now()
    year = ov.get("year") or base.year
    try:
        effective = base.replace(year=year)
    except ValueError:  # Feb 29 in a non-leap override year
        effective = base
    date_str = effective.strftime("%Y-%m-%d (%A)")
    lines = [f"Current date: {date_str}"]
    lines.append(f"Timezone: {_local_tz()}")
    if ov.get("note"):
        lines.append(f"User-corrected temporal note: {ov['note']}")
    if ov.get("year") and ov["year"] != base.year:
        lines.append(f"NOTE: user explicitly corrected the year to {ov['year']} (system date is {base.year}).")
    return "\n".join(lines)


def detect_temporal_correction(text: str) -> Optional[Dict]:
    """Detect an explicit user correction of the current date/year.

    Returns {"year": int, "note": str} or None. Deterministic regex rules;
    never guesses a year without an explicit statement.
    """
    t = re.sub(r"\s+", " ", (text or "").lower()).strip()
    patterns = [
        # "the date is wrong", "today is 2026", "it's 2026, not 2025"
        r"\b(it'?s|this is|today is|the year is|currently|we are in|we'?re in|it is)\b[^.]{0,40}?\b(20\d\d)\b",
        # "not 2025", "not in 2025", "you said 2025"
        r"\bnot(?: in)?\s+(20\d\d)\b",
        # "year is wrong", "date is wrong", "wrong year"
        r"\b(year|date)\b[^.]{0,20}?\bwrong\b[^.]{0,40}?\b(20\d\d)\b",
        r"\bwrong\b[^.]{0,20}?\b(20\d\d)\b",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            # Prefer the LAST 20XX token in the match (the corrected year).
            years = [int(x) for x in re.findall(r"20\d\d", m.group(0))]
            if years:
                year = years[-1]
                if 2020 <= year <= 2100:
                    return {"year": year, "note": text[:200]}
    return None


def apply_temporal_correction(session_id: str, correction: Dict) -> Dict:
    """Persist a user-corrected temporal context for a session."""
    with _lock:
        current = dict(_overrides.get(session_id) or {})
        current.update(correction)
        _overrides[session_id] = current
        _save()
        return dict(current)


def set_temporal_context(session_id: str, context: Dict) -> Dict:
    """Explicitly set (or clear) a session's temporal context."""
    with _lock:
        if context:
            _overrides[session_id] = dict(context)
        else:
            _overrides.pop(session_id, None)
        _save()
        return dict(_overrides.get(session_id) or {})


def get_temporal_context(session_id: str = "default") -> Dict:
    with _lock:
        return dict(_overrides.get(session_id) or {})


def clear_all_overrides() -> None:
    """Test-hygiene helper: clear persisted overrides."""
    global _overrides
    with _lock:
        _overrides = {}
        _save()