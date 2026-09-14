"""Tests for plans persistence layer (Phase 2.2.1)."""

import os
import sqlite3
import tempfile
import uuid
from unittest.mock import patch

import pytest

import plans


def _create_temp_plans_db() -> str:
    """Create a temporary plans database for isolated testing."""
    tmp = tempfile.mktemp(suffix=".db")
    return tmp


@pytest.fixture
def temp_db():
    """Provide a temporary database path and clean up after."""
    db_path = _create_temp_plans_db()
    # Patch the DB_PATH in the plans module
    original_db_path = plans.DB_PATH
    plans.DB_PATH = db_path
    plans.init_db()
    yield db_path
    # Cleanup
    plans.DB_PATH = original_db_path
    try:
        os.unlink(db_path)
    except OSError:
        pass


def test_init_db_creates_schema(temp_db):
    """Test that init_db creates all tables and indexes."""
    with plans._connect() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "plans" in table_names
        assert "plan_steps" in table_names
        assert "execution_records" in table_names

        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {r[0] for r in indexes}
        assert "idx_plan_steps_plan_id" in index_names
        assert "idx_execution_records_plan_id" in index_names
        assert "idx_execution_records_step" in index_names


def test_plan_roundtrip(temp_db):
    """Test saving and loading a plan."""
    plan_id = uuid.uuid4().hex[:8]
    goal = "Test goal for roundtrip"

    plans.save_plan(
        plan_id=plan_id,
        original_goal=goal,
        status="running",
        current_step=0,
        final_answer="",
        active_task_id="task_123",
    )

    loaded = plans.load_plan(plan_id)
    assert loaded is not None
    assert loaded["plan_id"] == plan_id
    assert loaded["original_goal"] == goal
    assert loaded["status"] == "running"
    assert loaded["current_step"] == 0
    assert loaded["final_answer"] == ""
    assert loaded["active_task_id"] == "task_123"
    assert loaded["created_at"] is not None
    assert loaded["updated_at"] is not None


def test_plan_status_update(temp_db):
    """Test updating plan status, current_step, and final_answer."""
    plan_id = uuid.uuid4().hex[:8]
    plans.save_plan(plan_id, "Test goal", status="running", current_step=0)

    # Update status and current_step
    plans.update_plan_status(plan_id, status="running", current_step=1)
    loaded = plans.load_plan(plan_id)
    assert loaded["status"] == "running"
    assert loaded["current_step"] == 1

    # Update final_answer
    plans.update_plan_status(plan_id, status="completed", final_answer="Done!")
    loaded = plans.load_plan(plan_id)
    assert loaded["status"] == "completed"
    assert loaded["final_answer"] == "Done!"


def test_plan_steps_roundtrip(temp_db):
    """Test saving and loading multiple plan steps."""
    plan_id = "test_plan_1"
    plans.save_plan(plan_id, "Test goal", status="running")

    # Save multiple steps
    step1_id = "step_1"
    plans.save_plan_step(
        step_id=step1_id,
        plan_id=plan_id,
        step_index=0,
        goal="First step",
        tool_hint="web_search",
        args={"query": "test"},
        status="pending",
    )

    step2_id = "step_2"
    plans.save_plan_step(
        step_id=step2_id,
        plan_id=plan_id,
        step_index=1,
        goal="Second step",
        tool_hint="read_file",
        args={"path": "test.py"},
        status="pending",
    )

    # Load all steps for plan
    steps = plans.load_plan_steps(plan_id)
    assert len(steps) == 2
    assert steps[0]["step_id"] == step1_id
    assert steps[0]["step_index"] == 0
    assert steps[0]["goal"] == "First step"
    assert steps[0]["tool_hint"] == "web_search"
    assert steps[0]["args"] == {"query": "test"}
    assert steps[0]["status"] == "pending"

    assert steps[1]["step_id"] == step2_id
    assert steps[1]["step_index"] == 1
    assert steps[1]["goal"] == "Second step"
    assert steps[1]["tool_hint"] == "read_file"
    assert steps[1]["args"] == {"path": "test.py"}


def test_step_status_update(temp_db):
    """Test updating step status, result, and evaluation."""
    plan_id = "test_plan_1"
    step_id = "step_1"
    plans.save_plan(plan_id, "Test")
    plans.save_plan_step(step_id, plan_id, 0, "Test step", "web_search", {"q": "test"})

    # Update status and result
    plans.update_step_status(plan_id, step_id, "running")
    step = plans.load_plan_step(plan_id, step_id)
    assert step["status"] == "running"

    plans.update_step_status(plan_id, step_id, "completed", result="Found it", evaluation="success")
    step = plans.load_plan_step(plan_id, step_id)
    assert step["status"] == "completed"
    assert step["result"] == "Found it"
    assert step["evaluation"] == "success"


def test_multiple_steps_one_plan(temp_db):
    """Test multiple steps belonging to one plan."""
    plan_id = "test_plan_multi"
    plans.save_plan(plan_id, "Multi-step plan")

    step_ids = []
    for i in range(3):
        step_id = f"step_{i}"
        step_ids.append(step_id)
        plans.save_plan_step(
            step_id=step_id,
            plan_id=plan_id,
            step_index=i,
            goal=f"Step {i}",
            tool_hint="web_search",
            args={"query": f"query {i}"},
        )

    steps = plans.load_plan_steps(plan_id)
    assert len(steps) == 3
    for i, step in enumerate(steps):
        assert step["step_index"] == i
        assert step["step_id"] == step_ids[i]


def test_execution_record_roundtrip(temp_db):
    """Test saving and loading execution records."""
    plan_id = "test_plan_1"
    step_id = "step_1"
    plans.save_plan(plan_id, "Test")
    plans.save_plan_step(step_id, plan_id, 0, "Test", "web_search", {})

    exec_id = "exec_1"
    plans.save_execution_record(
        execution_id=exec_id,
        plan_id=plan_id,
        step_id=step_id,
        step_index=0,
        tool="web_search",
        args={"query": "test"},
        result="Found results",
        evaluation="success",
        success=True,
        attempt=1,
        duration_ms=150,
    )

    loaded = plans.load_execution_record(exec_id)
    assert loaded is not None
    assert loaded["execution_id"] == exec_id
    assert loaded["plan_id"] == plan_id
    assert loaded["step_id"] == step_id
    assert loaded["step_index"] == 0
    assert loaded["tool"] == "web_search"
    assert loaded["args"] == {"query": "test"}
    assert loaded["result"] == "Found results"
    assert loaded["evaluation"] == "success"
    assert loaded["success"] is True
    assert loaded["attempt"] == 1
    assert loaded["duration_ms"] == 150


def test_execution_records_for_plan(temp_db):
    """Test loading all execution records for a plan."""
    plan_id = "test_plan_1"
    step_id = "step_1"
    plans.save_plan(plan_id, "Test")
    plans.save_plan_step(step_id, plan_id, 0, "Test", "web_search", {})

    # Save multiple execution records
    for i in range(3):
        exec_id = f"exec_{i}"
        plans.save_execution_record(
            execution_id=exec_id,
            plan_id=plan_id,
            step_id=step_id,
            step_index=0,
            tool="web_search",
            args={"query": f"q{i}"},
            result=f"Result {i}",
            evaluation="success",
            success=True,
            attempt=i + 1,
        )

    records = plans.load_execution_records_for_plan(plan_id)
    assert len(records) == 3
    for i, r in enumerate(records):
        assert r["attempt"] == i + 1


def test_execution_records_for_step(temp_db):
    """Test loading execution records for a specific step."""
    plan_id = "test_plan_1"
    step_id = "step_1"
    plans.save_plan(plan_id, "Test")
    plans.save_plan_step(step_id, plan_id, 0, "Test", "web_search", {})

    for i in range(2):
        plans.save_execution_record(
            execution_id=f"exec_{i}",
            plan_id=plan_id,
            step_id=step_id,
            step_index=0,
            tool="web_search",
            args={"query": f"q{i}"},
            result=f"Result {i}",
            evaluation="success",
            success=True,
            attempt=i + 1,
        )

    records = plans.load_execution_records_for_step(plan_id, step_id)
    assert len(records) == 2
    assert all(r["step_id"] == step_id for r in records)


def test_latest_execution_for_step(temp_db):
    """Test loading the latest execution for a step."""
    plan_id = "test_plan_1"
    step_id = "step_1"
    plans.save_plan(plan_id, "Test")
    plans.save_plan_step(step_id, plan_id, 0, "Test", "web_search", {})

    for i in range(3):
        plans.save_execution_record(
            execution_id=f"exec_{i}",
            plan_id=plan_id,
            step_id=step_id,
            step_index=0,
            tool="web_search",
            args={"query": f"q{i}"},
            result=f"Result {i}",
            evaluation="success",
            success=True,
            attempt=i + 1,
        )

    latest = plans.load_latest_execution_for_step(plan_id, step_id)
    assert latest is not None
    assert latest["attempt"] == 3


def test_update_execution_record(temp_db):
    """Test updating an execution record."""
    plan_id = "test_plan_1"
    step_id = "step_1"
    plans.save_plan(plan_id, "Test")
    plans.save_plan_step(step_id, plan_id, 0, "Test", "web_search", {})

    exec_id = "exec_1"
    plans.save_execution_record(
        execution_id=exec_id,
        plan_id=plan_id,
        step_id=step_id,
        step_index=0,
        tool="web_search",
        args={},
        result="",
        evaluation="",
        success=False,
    )

    updated = plans.update_execution_record(
        exec_id,
        result="Found it",
        evaluation="success",
        success=True,
        duration_ms=200,
    )
    assert updated is True

    loaded = plans.load_execution_record(exec_id)
    assert loaded["result"] == "Found it"
    assert loaded["evaluation"] == "success"
    assert loaded["success"] is True
    assert loaded["duration_ms"] == 200


def test_update_execution_record_not_found(temp_db):
    """Test updating non-existent execution record returns False."""
    result = plans.update_execution_record("nonexistent", result="test")
    assert result is False


def test_load_full_plan(temp_db):
    """Test loading a plan with all steps and execution records."""
    plan_id = "test_plan_full"
    plans.save_plan(plan_id, "Full plan test", status="running", current_step=1)

    step1_id = "step_1"
    plans.save_plan_step(step_id=step1_id, plan_id=plan_id, step_index=0,
                         goal="Step 1", tool_hint="web_search", args={"q": "1"})
    step2_id = "step_2"
    plans.save_plan_step(step_id=step2_id, plan_id=plan_id, step_index=1,
                         goal="Step 2", tool_hint="read_file", args={"path": "x.py"})

    plans.save_execution_record(
        execution_id="exec_1", plan_id=plan_id, step_id=step1_id,
        step_index=0, tool="web_search", args={"q": "1"}, result="OK",
        evaluation="success", success=True, attempt=1
    )

    full = plans.load_full_plan(plan_id)
    assert full is not None
    assert full["plan_id"] == plan_id
    assert len(full["steps"]) == 2
    assert len(full["execution_records"]) == 1


def test_persistence_across_connections(temp_db):
    """Test that data persists after closing and reopening connection."""
    plan_id = "test_persist"
    plans.save_plan(plan_id, "Persistence test")
    step_id = "step_1"
    plans.save_plan_step(step_id, plan_id, 0, "Step", "web_search", {"q": "test"})

    # Simulate closing and reopening by creating new connection
    # (The module-level connection is per-call, so this happens naturally)
    loaded = plans.load_plan(plan_id)
    assert loaded["plan_id"] == plan_id

    steps = plans.load_plan_steps(plan_id)
    assert len(steps) == 1
    assert steps[0]["step_id"] == step_id


def test_delete_plan_cascades(temp_db):
    """Test that deleting a plan cascades to steps and execution records."""
    plan_id = "test_delete"
    step_id = "step_1"
    exec_id = "exec_1"

    plans.save_plan(plan_id, "To delete")
    plans.save_plan_step(step_id, plan_id, 0, "Step", "web_search", {})
    plans.save_execution_record(
        execution_id=exec_id, plan_id=plan_id, step_id=step_id,
        step_index=0, tool="web_search", args={}
    )

    deleted = plans.delete_plan(plan_id)
    assert deleted is True

    # Plan should be gone
    assert plans.load_plan(plan_id) is None
    # Steps should be gone (cascade)
    assert plans.load_plan_step(plan_id, step_id) is None
    # Execution records should be gone (cascade)
    assert plans.load_execution_record(exec_id) is None


def test_delete_plan_not_found(temp_db):
    """Test deleting non-existent plan returns False."""
    result = plans.delete_plan("nonexistent")
    assert result is False


def test_foreign_keys_enforced(temp_db):
    """Test that foreign key constraints are enforced."""
    # Try to insert step with non-existent plan_id
    with plans._connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO plan_steps (plan_id, step_id, step_index, goal, tool_hint, args_json) VALUES (?, ?, ?, ?, ?, ?)",
                ("nonexistent_plan", "step1", 0, "Test", "web_search", "{}"),
            )


def test_list_plans(temp_db):
    """Test listing plans with optional status filter."""
    for i in range(3):
        pid = f"plan_{i}"
        plans.save_plan(pid, f"Plan {i}", status="running" if i < 2 else "completed")

    all_plans = plans.list_plans()
    assert len(all_plans) == 3

    running = plans.list_plans(status="running")
    assert len(running) == 2

    completed = plans.list_plans(status="completed")
    assert len(completed) == 1


def test_set_plan_active_task(temp_db):
    """Test linking a plan to an ActiveTask."""
    plan_id = "plan_test"
    plans.save_plan(plan_id, "Test")
    plans.set_plan_active_task(plan_id, "task_abc")

    loaded = plans.load_plan(plan_id)
    assert loaded["active_task_id"] == "task_abc"


def test_load_plan_step_not_found(temp_db):
    """Test loading non-existent step returns None."""
    assert plans.load_plan_step("nonexistent_plan", "nonexistent") is None


def test_load_execution_record_not_found(temp_db):
    """Test loading non-existent execution record returns None."""
    assert plans.load_execution_record("nonexistent") is None


def test_load_plan_not_found(temp_db):
    """Test loading non-existent plan returns None."""
    assert plans.load_plan("nonexistent") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])