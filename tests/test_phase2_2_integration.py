"""Integration tests for Phase 2.2.2: Plan persistence integration with agent lifecycle."""

import os
import tempfile
import uuid
from unittest.mock import MagicMock

import pytest

import agent
import plans


def _create_temp_plans_db() -> str:
    """Create a temporary plans database for isolated testing."""
    return tempfile.mktemp(suffix=".db")


@pytest.fixture
def temp_db():
    """Provide a temporary database path and clean up after."""
    db_path = _create_temp_plans_db()
    # Patch the DB_PATH in the plans module
    original_db_path = plans.DB_PATH
    plans.DB_PATH = db_path
    plans.init_db()
    # Also patch agent's plans module reference
    original_agent_plans_db = agent.plans.DB_PATH
    agent.plans.DB_PATH = db_path
    agent.plans.init_db()
    yield db_path
    # Cleanup
    plans.DB_PATH = original_db_path
    agent.plans.DB_PATH = original_agent_plans_db
    try:
        os.unlink(db_path)
    except OSError:
        pass


def _mock_llm_fn(prompt: str) -> str:
    """Mock LLM function that returns a valid plan."""
    return '''[
  {"step_id": "1", "goal": "Search for information", "tool_hint": "web_search", "args": {"query": "test query"}},
  {"step_id": "2", "goal": "Read the file", "tool_hint": "read_file", "args": {"path": "test.py"}}
]'''


def _mock_llm_fn_2(prompt: str) -> str:
    """Mock LLM function that returns a different valid plan."""
    return '''[
  {"step_id": "1", "goal": "Search for different info", "tool_hint": "web_search", "args": {"query": "different query"}},
  {"step_id": "2", "goal": "Write the file", "tool_hint": "write_file", "args": {"path": "out.txt", "content": "data"}}
]'''


def _mock_llm_fn_failure(prompt: str) -> str:
    """Mock LLM function that returns invalid response."""
    return "I cannot create a plan for this."


def test_plan_creation_persists(temp_db):
    """Test that _create_plan persists the Plan and its steps."""
    plan = agent._create_plan("Test goal", _mock_llm_fn)
    
    assert plan is not None
    assert plan.plan_id is not None
    assert plan.original_goal == "Test goal"
    assert len(plan.steps) == 2
    
    # Verify Plan persisted
    loaded_plan = plans.load_plan(plan.plan_id)
    assert loaded_plan is not None
    assert loaded_plan["plan_id"] == plan.plan_id
    assert loaded_plan["original_goal"] == "Test goal"
    assert loaded_plan["status"] == "running"
    assert loaded_plan["current_step"] == 0
    
    # Verify steps persisted
    steps = plans.load_plan_steps(plan.plan_id)
    assert len(steps) == 2
    assert steps[0]["step_index"] == 0
    assert steps[1]["step_index"] == 1


def test_plan_steps_persist_with_correct_data(temp_db):
    """Test that PlanSteps persist with all their data."""
    plan = agent._create_plan("Test goal", _mock_llm_fn)
    
    steps = plans.load_plan_steps(plan.plan_id)
    assert len(steps) == 2
    
    step1 = steps[0]
    assert step1["step_index"] == 0
    assert step1["goal"] == "Search for information"
    assert step1["tool_hint"] == "web_search"
    assert step1["args"] == {"query": "test query"}
    assert step1["status"] == "pending"
    assert step1["result"] == ""
    assert step1["evaluation"] == ""
    
    step2 = steps[1]
    assert step2["step_index"] == 1
    assert step2["goal"] == "Read the file"
    assert step2["tool_hint"] == "read_file"
    assert step2["args"] == {"path": "test.py"}


def test_registration_no_duplicate_persistent_records(temp_db):
    """Test that _register_plan doesn't create duplicate persistent records."""
    plan = agent._create_plan("Test goal", _mock_llm_fn)
    plan_id = plan.plan_id
    
    # Simulate what _register_plan does - add to _plan_store
    agent._register_plan(plan)
    
    # Verify only one persistent record exists
    with plans._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM plans WHERE plan_id = ?", (plan_id,)).fetchone()[0]
        assert count == 1
    
    step_count = plans._connect().execute(
        "SELECT COUNT(*) FROM plan_steps WHERE plan_id = ?", (plan_id,)
    ).fetchone()[0]
    assert step_count == 2


def test_retrieval_works_from_persistence(temp_db):
    """Test that a persisted Plan can be retrieved when appropriate."""
    plan = agent._create_plan("Retrieval test", _mock_llm_fn)
    plan_id = plan.plan_id
    
    # Clear in-memory store to simulate restart
    agent._plan_store.clear()
    
    # get_plan should return None (in-memory fast path)
    # But persistence layer should still have the data
    loaded = plans.load_plan(plan_id)
    assert loaded is not None
    assert loaded["plan_id"] == plan_id
    assert loaded["original_goal"] == "Retrieval test"
    
    # Verify steps also retrievable
    steps = plans.load_plan_steps(plan_id)
    assert len(steps) == 2


def test_state_update_persists(temp_db):
    """Test that PlanStep state updates persist."""
    plan = agent._create_plan("State update test", _mock_llm_fn)
    plan_id = plan.plan_id
    
    # Get the first step
    steps = plans.load_plan_steps(plan_id)
    step_id = steps[0]["step_id"]
    
    # Update step status
    agent.plans.update_step_status(plan_id, step_id, "completed", result="Done", evaluation="success")
    
    # Verify persisted
    loaded = plans.load_plan_step(plan_id, step_id)
    assert loaded["status"] == "completed"
    assert loaded["result"] == "Done"
    assert loaded["evaluation"] == "success"


def test_persistence_failure_does_not_corrupt_memory(temp_db):
    """Test that persistence failure doesn't corrupt in-memory Plan state."""
    # Break the DB path temporarily to simulate persistence failure
    original_db = agent.plans.DB_PATH
    agent.plans.DB_PATH = "/invalid/path/that/does/not/exist/plans.db"
    
    try:
        plan = agent._create_plan("Test goal", _mock_llm_fn)
        # Plan should still be created in memory
        assert plan is not None
        assert plan.original_goal == "Test goal"
        assert len(plan.steps) == 2
        assert plan.plan_id in agent._plan_store
    finally:
        agent.plans.DB_PATH = original_db
        agent.plans.init_db()


def test_invalid_plan_not_persisted(temp_db):
    """Test that invalid plans (no steps) are not persisted."""
    plan = agent._create_plan("Test goal", _mock_llm_fn_failure)
    assert plan is None
    
    # Nothing should be persisted
    with plans._connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        assert count == 0


def test_replan_also_persists(temp_db):
    """Test that _replan (which calls _create_plan) also persists."""
    # Create initial plan
    plan1 = agent._create_plan("Initial goal", _mock_llm_fn)
    
    # Replan with different mock to avoid step_id collision
    plan2 = agent._replan("New goal", _mock_llm_fn_2, [])
    
    # Both plans should be persisted
    assert plans.load_plan(plan1.plan_id) is not None
    assert plans.load_plan(plan2.plan_id) is not None
    assert len(plans.load_plan_steps(plan1.plan_id)) == 2
    assert len(plans.load_plan_steps(plan2.plan_id)) == 2
    assert plan1.plan_id != plan2.plan_id


def test_existing_lifecycle_preserved(temp_db):
    """Test that existing agent lifecycle is preserved."""
    # Test that _create_plan returns a proper Plan object
    plan = agent._create_plan("Test goal", _mock_llm_fn)
    assert isinstance(plan, agent.Plan)
    assert plan.plan_id is not None
    assert plan.original_goal == "Test goal"
    assert len(plan.steps) == 2
    assert plan.status == "running"
    assert plan.current_step == 0
    
    # Test _register_plan still works
    assert plan.plan_id in agent._plan_store
    
    # Test get_plan works
    loaded = agent.get_plan(plan.plan_id)
    assert loaded is plan  # Same object from in-memory store


if __name__ == "__main__":
    pytest.main([__file__, "-v"])