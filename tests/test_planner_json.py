"""Phase 4 tests: robust planner JSON parsing + schema hardening (agent.py).

Covers _extract_json_array / _extract_json_object / _step_from_dict /
_create_plan when the planner model returns fences, prose-wrapped arrays,
plain completion text, or malformed step shapes.
"""

import json

from agent import (
    _create_plan,
    _extract_json_array,
    _extract_json_object,
    _step_from_dict,
    _plan_store,
)
from event_bus import get_recent


class TestExtractJsonArray:
    def test_strict_array(self):
        raw = '[{"step_id": "1", "goal": "g", "tool_hint": "web_search", "args": {"query": "q"}}]'
        data = _extract_json_array(raw)
        assert isinstance(data, list)
        assert data[0]["tool_hint"] == "web_search"

    def test_fenced_array(self):
        raw = 'Here is the plan:\n```json\n[{"step_id": "1", "goal": "g", "token": "keep"}]\n```\nRegards'
        data = _extract_json_array(raw)
        assert isinstance(data, list)
        assert data[0]["token"] == "keep"

    def test_array_embedded_in_prose(self):
        raw = 'Plan:\n[{"step_id": "1", "goal": "find", "tool_hint": "web_search", "args": {}}]\nI will start now.'
        data = _extract_json_array(raw)
        assert len(data) == 1
        assert data[0]["goal"] == "find"

    def test_prose_only_returns_none(self):
        assert _extract_json_array("This task is simple. I will just open Safari and search.") is None

    def test_empty_returns_none(self):
        assert _extract_json_array("") is None
        assert _extract_json_array(None) is None

    def test_brackets_inside_string_values(self):
        """A '[' inside a string value must not truncate the scan."""
        raw = '[{"args": {"query": "[python] env"}}]'
        data = _extract_json_array(raw)
        assert data[0]["args"]["query"] == "[python] env"

    def test_object_not_array_returns_none(self):
        assert _extract_json_array('{"tool": "open_app", "args": {}}') is None


class TestExtractJsonObject:
    def test_prose_with_object(self):
        raw = 'Try this instead: {"tool": "open_app", "args": {"app_name": "Safari"}, "reason": "x"}. Go!'
        data = _extract_json_object(raw)
        assert data is not None
        assert data["tool"] == "open_app"

    def test_prose_only_returns_none(self):
        assert _extract_json_object("Just open Safari manually.") is None

    def test_fenced_object(self):
        raw = '```json\n{"tool": "web_search", "args": {"query": "q"}}\n```'
        data = _extract_json_object(raw)
        assert data["tool"] == "web_search"


class TestStepFromDict:
    def test_missing_fields_fill_defaults(self):
        s = _step_from_dict({"goal": "g"})
        assert s is not None
        assert s.args == {}
        assert s.tool_hint == ""
        assert s.status == "pending"

    def test_extra_fields_dropped(self):
        s = _step_from_dict({"step_id": "custom_step_2", "explanation": "irrelevant", "args": {"query": "q"}})
        assert s.step_id == "custom_step_2"
        assert s.args == {"query": "q"}

    def test_non_dict_arg_coerced(self):
        s = _step_from_dict({"args": "open Safari"})
        assert s.args == {}

    def test_non_dict_step_rejected(self):
        assert _step_from_dict("not a dict") is None
        assert _step_from_dict(42) is None


class TestCreatePlan:
    def _reset_store(self):
        _plan_store.clear()

    def test_plain_completion_reply_no_crash(self):
        """A prose-only reply yields None (graceful), not an exception."""
        self._reset_store()
        called = {"n": 0}

        def ask(prompt):
            called["n"] += 1
            return "This task is straightforward — I will handle it directly."

        plan = _create_plan("organize downloads", ask)
        assert plan is None
        assert called["n"] == 1

    def test_fenced_json_reply_parses(self):
        self._reset_store()
        steps = [
            {"step_id": "1", "goal": "search docs", "tool_hint": "web_search", "args": {"query": "python"}},
            {"step_id": "2", "goal": "open results", "tool_hint": "open_app", "args": {"app_name": "Safari"}},
        ]

        def ask(prompt):
            return f"Okay! Here is the plan:\n```json\n{json.dumps(steps)}\n```\nLet me know if you want changes."

        plan = _create_plan("research python", ask)
        assert plan is not None
        assert len(plan.steps) == 2
        assert plan.steps[1].args == {"app_name": "Safari"}

    def test_malformed_entry_dropped_others_kept(self):
        self._reset_store()
        raw = (
            '[{"step_id": "1", "goal": "search", "tool_hint": "web_search", "args": {"query": "q"}},'
            " 42,"
            ' {"step_id": "2", "goal": "open", "tool_hint": "open_app", "args": {"app_name": "Safari"}}]'
        )
        plan = _create_plan("research", lambda p: raw)
        assert plan is not None
        assert len(plan.steps) == 2
        assert plan.steps[0].goal == "search"

    def test_bad_list_of_non_dicts_returns_none(self):
        self._reset_store()
        plan = _create_plan("research", lambda p: "[1, 2, 3]")
        assert plan is None


class TestPlannerLoopPlainCompletion:
    """run_planner_loop with a plain-completion planner reply."""

    def test_loop_returns_canned_message_without_crashing(self):
        captured = {}
        before = len(get_recent("subagent_completed"))

        from agent import run_planner_loop

        def ask(prompt):
            if "planner" in prompt.lower():
                return "No planning needed — I'll just do it directly."
            return "SUCCESS done"

        def execute_tool(name, args):
            captured["tool"] = name
            return "ok"

        out = run_planner_loop("research python", execute_tool, ask)
        assert "couldn't create a plan" in out
        assert "tool" not in captured  # no tool executed
        assert len(get_recent("subagent_completed")) == before  # no completion event on plan failure
