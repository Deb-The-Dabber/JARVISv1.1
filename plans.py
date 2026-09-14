"""Persistent Plan / PlanStep / ExecutionRecord storage (Phase 2.2).

Follows existing JARVIS SQLite patterns:
- Dedicated DB file per domain (~/.jarvis/plans.db)
- _connect() returns sqlite3.Connection
- init_db() creates schema with IF NOT EXISTS
- Context-manager connections: with _connect() as conn:
- Thread-safe with RLock
- Foreign keys enabled via PRAGMA
"""

import datetime
import json
import os
import sqlite3
import threading
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.expanduser("~"), ".jarvis", "plans.db")

_lock = threading.RLock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Initialize the plans database schema."""
    with _lock:
        with _connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    original_goal TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    current_step INTEGER NOT NULL DEFAULT 0,
                    final_answer TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active_task_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_steps (
                    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
                    step_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    goal TEXT NOT NULL,
                    tool_hint TEXT NOT NULL,
                    args_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    result TEXT DEFAULT '',
                    evaluation TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (plan_id, step_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_records (
                    execution_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
                    step_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    tool TEXT NOT NULL,
                    args_json TEXT NOT NULL DEFAULT '{}',
                    result TEXT DEFAULT '',
                    evaluation TEXT DEFAULT '',
                    success BOOLEAN NOT NULL DEFAULT 0,
                    error TEXT,
                    timestamp TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (plan_id, step_id) REFERENCES plan_steps(plan_id, step_id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_steps_plan_id ON plan_steps(plan_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_records_plan_id ON execution_records(plan_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_records_step ON execution_records(plan_id, step_id)
            """)
            conn.commit()


def _row_to_plan(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "plan_id": row["plan_id"],
        "original_goal": row["original_goal"],
        "status": row["status"],
        "current_step": row["current_step"],
        "final_answer": row["final_answer"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "active_task_id": row["active_task_id"],
    }


def _row_to_plan_step(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "step_id": row["step_id"],
        "plan_id": row["plan_id"],
        "step_index": row["step_index"],
        "goal": row["goal"],
        "tool_hint": row["tool_hint"],
        "args": json.loads(row["args_json"]) if row["args_json"] else {},
        "status": row["status"],
        "result": row["result"],
        "evaluation": row["evaluation"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_execution_record(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "execution_id": row["execution_id"],
        "plan_id": row["plan_id"],
        "step_id": row["step_id"],
        "step_index": row["step_index"],
        "tool": row["tool"],
        "args": json.loads(row["args_json"]) if row["args_json"] else {},
        "result": row["result"],
        "evaluation": row["evaluation"],
        "success": bool(row["success"]),
        "error": row["error"],
        "timestamp": row["timestamp"],
        "attempt": row["attempt"],
        "duration_ms": row["duration_ms"],
    }


# ─────────────────────────────────────────────
# PLAN CRUD
# ─────────────────────────────────────────────

def save_plan(
    plan_id: str,
    original_goal: str,
    status: str = "running",
    current_step: int = 0,
    final_answer: str = "",
    active_task_id: Optional[str] = None,
) -> None:
    """Insert or replace a plan."""
    now = datetime.datetime.now().isoformat()
    with _lock:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO plans (plan_id, original_goal, status, current_step, final_answer, created_at, updated_at, active_task_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    original_goal = excluded.original_goal,
                    status = excluded.status,
                    current_step = excluded.current_step,
                    final_answer = excluded.final_answer,
                    updated_at = excluded.updated_at,
                    active_task_id = excluded.active_task_id
                """,
                (plan_id, original_goal, status, current_step, final_answer, now, now, active_task_id),
            )
            conn.commit()


def load_plan(plan_id: str) -> Optional[Dict[str, Any]]:
    """Load a plan by ID."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,)).fetchone()
        return _row_to_plan(row) if row else None


def update_plan_status(plan_id: str, status: str, current_step: Optional[int] = None, final_answer: Optional[str] = None) -> None:
    """Update plan status and optionally current_step/final_answer."""
    now = datetime.datetime.now().isoformat()
    with _lock:
        with _connect() as conn:
            if current_step is not None and final_answer is not None:
                conn.execute(
                    "UPDATE plans SET status = ?, current_step = ?, final_answer = ?, updated_at = ? WHERE plan_id = ?",
                    (status, current_step, final_answer, now, plan_id),
                )
            elif current_step is not None:
                conn.execute(
                    "UPDATE plans SET status = ?, current_step = ?, updated_at = ? WHERE plan_id = ?",
                    (status, current_step, now, plan_id),
                )
            elif final_answer is not None:
                conn.execute(
                    "UPDATE plans SET status = ?, final_answer = ?, updated_at = ? WHERE plan_id = ?",
                    (status, final_answer, now, plan_id),
                )
            else:
                conn.execute(
                    "UPDATE plans SET status = ?, updated_at = ? WHERE plan_id = ?",
                    (status, now, plan_id),
                )
            conn.commit()


def set_plan_active_task(plan_id: str, active_task_id: str) -> None:
    """Link a plan to an ActiveTask."""
    now = datetime.datetime.now().isoformat()
    with _lock:
        with _connect() as conn:
            conn.execute(
                "UPDATE plans SET active_task_id = ?, updated_at = ? WHERE plan_id = ?",
                (active_task_id, now, plan_id),
            )
            conn.commit()


def list_plans(limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """List plans, optionally filtered by status."""
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM plans WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM plans ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_plan(r) for r in rows]


def delete_plan(plan_id: str) -> bool:
    """Delete a plan and all its steps/records (cascades)."""
    with _lock:
        with _connect() as conn:
            cur = conn.execute("DELETE FROM plans WHERE plan_id = ?", (plan_id,))
            conn.commit()
            return cur.rowcount > 0


# ─────────────────────────────────────────────
# PLAN STEP CRUD
# ─────────────────────────────────────────────

def save_plan_step(
    step_id: str,
    plan_id: str,
    step_index: int,
    goal: str,
    tool_hint: str,
    args: Dict[str, Any],
    status: str = "pending",
    result: str = "",
    evaluation: str = "",
) -> None:
    """Insert or replace a plan step."""
    now = datetime.datetime.now().isoformat()
    args_json = json.dumps(args) if args else "{}"
    with _lock:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO plan_steps (plan_id, step_id, step_index, goal, tool_hint, args_json, status, result, evaluation, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id, step_id) DO UPDATE SET
                    step_index = excluded.step_index,
                    goal = excluded.goal,
                    tool_hint = excluded.tool_hint,
                    args_json = excluded.args_json,
                    status = excluded.status,
                    result = excluded.result,
                    evaluation = excluded.evaluation,
                    updated_at = excluded.updated_at
                """,
                (plan_id, step_id, step_index, goal, tool_hint, args_json, status, result, evaluation, now, now),
            )
            conn.commit()


def load_plan_step(plan_id: str, step_id: str) -> Optional[Dict[str, Any]]:
    """Load a single plan step by composite key."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM plan_steps WHERE plan_id = ? AND step_id = ?", (plan_id, step_id)).fetchone()
        return _row_to_plan_step(row) if row else None


def load_plan_step_by_id_only(step_id: str) -> Optional[Dict[str, Any]]:
    """Load a single plan step by step_id only (returns first match if duplicates exist across plans)."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM plan_steps WHERE step_id = ? LIMIT 1", (step_id,)).fetchone()
        return _row_to_plan_step(row) if row else None


def load_plan_steps(plan_id: str) -> List[Dict[str, Any]]:
    """Load all steps for a plan, ordered by step_index."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY step_index ASC",
            (plan_id,),
        ).fetchall()
        return [_row_to_plan_step(r) for r in rows]


def update_step_status(plan_id: str, step_id: str, status: str, result: Optional[str] = None, evaluation: Optional[str] = None) -> None:
    """Update step status and optionally result/evaluation."""
    now = datetime.datetime.now().isoformat()
    with _lock:
        with _connect() as conn:
            if result is not None and evaluation is not None:
                conn.execute(
                    "UPDATE plan_steps SET status = ?, result = ?, evaluation = ?, updated_at = ? WHERE plan_id = ? AND step_id = ?",
                    (status, result, evaluation, now, plan_id, step_id),
                )
            elif result is not None:
                conn.execute(
                    "UPDATE plan_steps SET status = ?, result = ?, updated_at = ? WHERE plan_id = ? AND step_id = ?",
                    (status, result, now, plan_id, step_id),
                )
            elif evaluation is not None:
                conn.execute(
                    "UPDATE plan_steps SET status = ?, evaluation = ?, updated_at = ? WHERE plan_id = ? AND step_id = ?",
                    (status, evaluation, now, plan_id, step_id),
                )
            else:
                conn.execute(
                    "UPDATE plan_steps SET status = ?, updated_at = ? WHERE plan_id = ? AND step_id = ?",
                    (status, now, plan_id, step_id),
                )
            conn.commit()


def delete_plan_steps(plan_id: str) -> int:
    """Delete all steps for a plan. Returns count deleted."""
    with _lock:
        with _connect() as conn:
            cur = conn.execute("DELETE FROM plan_steps WHERE plan_id = ?", (plan_id,))
            conn.commit()
            return cur.rowcount


# ─────────────────────────────────────────────
# EXECUTION RECORD CRUD
# ─────────────────────────────────────────────

def save_execution_record(
    execution_id: str,
    plan_id: str,
    step_id: str,
    step_index: int,
    tool: str,
    args: Dict[str, Any],
    result: str = "",
    evaluation: str = "",
    success: bool = False,
    error: Optional[str] = None,
    attempt: int = 1,
    duration_ms: int = 0,
) -> None:
    """Insert an execution record."""
    timestamp = datetime.datetime.now().isoformat()
    args_json = json.dumps(args) if args else "{}"
    with _lock:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO execution_records
                (execution_id, plan_id, step_id, step_index, tool, args_json, result, evaluation, success, error, timestamp, attempt, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (execution_id, plan_id, step_id, step_index, tool, args_json, result, evaluation, int(success), error, timestamp, attempt, duration_ms),
            )
            conn.commit()


def load_execution_record(execution_id: str) -> Optional[Dict[str, Any]]:
    """Load a single execution record by ID."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM execution_records WHERE execution_id = ?", (execution_id,)).fetchone()
        return _row_to_execution_record(row) if row else None


def load_execution_records_for_plan(plan_id: str) -> List[Dict[str, Any]]:
    """Load all execution records for a plan, ordered by timestamp."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM execution_records WHERE plan_id = ? ORDER BY timestamp ASC",
            (plan_id,),
        ).fetchall()
        return [_row_to_execution_record(r) for r in rows]


def load_execution_records_for_step(plan_id: str, step_id: str) -> List[Dict[str, Any]]:
    """Load all execution records for a step, ordered by attempt."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM execution_records WHERE plan_id = ? AND step_id = ? ORDER BY attempt ASC",
            (plan_id, step_id),
        ).fetchall()
        return [_row_to_execution_record(r) for r in rows]


def load_latest_execution_for_step(plan_id: str, step_id: str) -> Optional[Dict[str, Any]]:
    """Load the most recent execution record for a step."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM execution_records WHERE plan_id = ? AND step_id = ? ORDER BY attempt DESC LIMIT 1",
            (plan_id, step_id),
        ).fetchone()
        return _row_to_execution_record(row) if row else None


def update_execution_record(
    execution_id: str,
    result: Optional[str] = None,
    evaluation: Optional[str] = None,
    success: Optional[bool] = None,
    error: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> bool:
    """Update an execution record. Returns True if found and updated."""
    with _lock:
        with _connect() as conn:
            updates = []
            params = []
            if result is not None:
                updates.append("result = ?")
                params.append(result)
            if evaluation is not None:
                updates.append("evaluation = ?")
                params.append(evaluation)
            if success is not None:
                updates.append("success = ?")
                params.append(int(success))
            if error is not None:
                updates.append("error = ?")
                params.append(error)
            if duration_ms is not None:
                updates.append("duration_ms = ?")
                params.append(duration_ms)
            if not updates:
                return False
            params.append(execution_id)
            cur = conn.execute(
                f"UPDATE execution_records SET {', '.join(updates)} WHERE execution_id = ?",
                params,
            )
            conn.commit()
            return cur.rowcount > 0


# ─────────────────────────────────────────────
# PLAN RECONSTRUCTION
# ─────────────────────────────────────────────

def load_full_plan(plan_id: str) -> Optional[Dict[str, Any]]:
    """Load a plan with all its steps and execution records."""
    plan = load_plan(plan_id)
    if not plan:
        return None
    plan["steps"] = load_plan_steps(plan_id)
    plan["execution_records"] = load_execution_records_for_plan(plan_id)
    return plan


def get_plan_current_step(plan_id: str) -> Optional[int]:
    """Get the current step index for a plan."""
    plan = load_plan(plan_id)
    return plan["current_step"] if plan else None