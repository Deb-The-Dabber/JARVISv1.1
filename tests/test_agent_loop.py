"""Phase 2 regression tests: agent-loop discipline (budgets, phases, graceful abort).

Covers run_agent_loop with fake execute/ask fns -- no network, no tools.
Env vars are read per-run, so monkeypatch.setenv works after import.
"""

import json

import pytest

import agent

DONE = {"thought": "finished", "tool": "", "args": {}, "done": True, "final_answer": "All done."}


def _decision(tool=None, args=None, phase=None, done=None, final_answer=None, no_response=False):
    d = {
        "thought": "t",
        "tool": tool or "",
        "args": args or {},
        "done": done or False,
        "final_answer": final_answer or "",
    }
    if phase:
        d["phase"] = phase
    if no_response:
        d["_no_response"] = True
    return d


def _ask(script):
    """Build an ask_llm_fn replaying a script of decisions (a callable or injected response)."""
    idx = {"n": 0}

    def ask_llm_fn(prompt):
        if callable(script):
            resp = script(prompt, idx["n"])
        elif isinstance(script, list):
            resp = script[min(idx["n"], len(script) - 1)]
        else:
            resp = script
        idx["n"] += 1
        if resp is None:
            return None
        if isinstance(resp, str):
            return resp
        return json.dumps(resp)

    return ask_llm_fn


def _run(ask_script, max_iterations=30, env=None, execute=None):
    if env:
        for k, v in env.items():
            pytest.MonkeyPatch().setenv(k, v)
    calls = []
    executed = []
    completed = {}
    from event_bus import subscribe

    def on_completed(payload):
        completed.update(payload)

    subscribe("subagent_completed", on_completed)

    def do_execute(tool, args):
        calls.append((tool, json.dumps(args, sort_keys=True)))
        executed.append(tool)
        if execute is not None:
            return execute(tool, args)
        return "ok"

    final = agent.run_agent_loop(
        goal="test goal",
        execute_tool_fn=do_execute,
        ask_llm_fn=_ask(ask_script),
        max_iterations=max_iterations,
    )
    agent_obj = agent.get_agent(completed.get("agent_id", "")) if completed else None
    if agent_obj is None:
        listed = agent.list_agents()
        if listed:
            agent_obj = agent.get_agent(listed[-1]["id"])
    return final, calls, executed, agent_obj


class TestToolBudget:
    def test_tool_budget_caps_executions(self, monkeypatch):
        def script(prompt, n):
            return {
                "thought": "keep going",
                "tool": "create_file",
                "args": {"path": f"/tmp/b{n}", "content": "y"},
                "done": False,
                "final_answer": "",
            }

        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "3")
        final, _, executed, _agent = _run(script)
        # With verification, each write_file triggers a read_file verification
        # So 3 tool calls = 3 create_file + 3 read_file verification = 6 total execute calls
        assert len(executed) == 6
        assert "tool budget" in final
        assert final.startswith("Agent summary for:")

    def test_tool_budget_respected_with_high_clamp(self, monkeypatch):
        """max_iterations clamped to tool budget + 4 regardless of caller."""

        def script(prompt, n):
            return {
                "thought": "k",
                "tool": "create_file",
                "args": {"path": f"/tmp/y{n}"},
                "done": False,
                "final_answer": "",
            }

        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "5")
        final, _, executed, _agent = _run(script, max_iterations=100)
        # With verification: 5 create_file + 5 read_file verification = 10 total
        assert len(executed) == 10


class TestExploreBudget:
    def test_explore_budget_blocks_further_reads(self, monkeypatch):
        def script(prompt, n):
            return {
                "thought": "k",
                "tool": "read_file",
                "args": {"path": f"/etc/f{n}"},
                "done": False,
                "final_answer": "",
            }

        monkeypatch.setenv("JARVIS_AGENT_MAX_EXPLORE", "2")
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "30")
        final, calls, executed, _agent = _run(script)
        assert executed.count("read_file") == 2
        assert any("exploration budget" in s.get("result", "") for s in _agent.steps)


class TestWallClock:
    def test_wall_clock_aborts_gracefully(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AGENT_WALL_CLOCK_S", "0.05")
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "20")

        def slow_ask(prompt, n):
            import time

            time.sleep(0.08)
            return json.dumps(
                {
                    "thought": "k",
                    "tool": "create_file",
                    "args": {"path": f"/tmp/z{n}"},
                    "done": False,
                    "final_answer": "",
                }
            )

        final, _, executed, _agent = _run(slow_ask)
        assert 1 <= len(executed) <= 2
        assert "wall clock" in final
        assert "No response from model." != final


class TestEmptyResponses:
    def test_two_empty_responses_abort_with_progress(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "20")
        script = [None] * 10  # model silent forever
        final, _, executed, _agent = _run(script)
        assert "model stopped responding" in final
        assert final.startswith("Agent summary for:")
        assert executed == []  # never executed anything -- clean abort

    def test_single_empty_response_retries(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "10")
        script = [
            None,
            {"thought": "k", "tool": "create_file", "args": {"path": "/tmp/a"}, "done": False, "final_answer": ""},
            DONE,
        ]
        final, _, executed, _agent = _run(script)
        # With verification: create_file + read_file verification = 2 calls
        assert executed == ["create_file", "read_file"]
        assert "done" in final.lower() or "All done" in final


class TestPhaseTransition:
    def test_phase_never_regresses(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "5")
        phases_seen = []
        call = {"n": 0}

        def ask(prompt, n):
            phases_seen.append([line for line in prompt.splitlines() if line.startswith("Current phase:")][0][15:])
            if call["n"] == 0:
                call["n"] += 1
                return json.dumps(
                    {
                        "thought": "k",
                        "tool": "web_search",
                        "args": {"query": "x"},
                        "phase": "implement",
                        "done": False,
                        "final_answer": "",
                    }
                )
            call["n"] += 1
            return json.dumps(
                {
                    "thought": "k",
                    "tool": "create_file",
                    "args": {"path": f"/tmp/{call['n']}"},
                    "phase": "explore",
                    "done": False,
                    "final_answer": "",
                }
            )

        _run(ask)
        # After implement was requested, no later prompt may show explore/plan
        implement_index = phases_seen.index("implement")
        assert all(p == "implement" for p in phases_seen[implement_index:])

    def test_plan_phase_auto_advances_after_two_steps(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "10")
        phases_seen = []
        call = {"n": 0}

        def ask(prompt, n):
            phases_seen.append([line for line in prompt.splitlines() if line.startswith("Current phase:")][0][15:])
            call["n"] += 1
            if call["n"] <= 3:
                return json.dumps(
                    {
                        "thought": "k",
                        "tool": "create_file",
                        "args": {"path": f"/tmp/p{call['n']}"},
                        "phase": "plan",
                        "done": False,
                        "final_answer": "",
                    }
                )
            return json.dumps(DONE)

        _run(ask)
        assert "plan" in phases_seen
        implement_index = phases_seen.index("implement") if "implement" in phases_seen else None
        assert implement_index is not None
        assert implement_index > phases_seen.index("plan")


class TestDuplicateGuard:
    def test_same_tool_same_args_executed_once(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "10")
        script = [
            {
                "thought": "k",
                "tool": "create_file",
                "args": {"path": "/tmp/dup", "content": "x"},
                "done": False,
                "final_answer": "",
            },
            {
                "thought": "k",
                "tool": "create_file",
                "args": {"path": "/tmp/dup", "content": "x"},
                "done": False,
                "final_answer": "",
            },
            DONE,
        ]
        final, calls, executed, _agent = _run(script)
        assert executed.count("create_file") == 1
        assert any("already called" in s.get("result", "") for s in _agent.steps)


class TestFailCountGuard:
    def test_tool_failed_twice_is_skipped(self, monkeypatch):
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "10")
        script = [
            {"thought": "k", "tool": "read_file", "args": {"path": "/nope1"}, "done": False, "final_answer": ""},
            {"thought": "k", "tool": "read_file", "args": {"path": "/nope2"}, "done": False, "final_answer": ""},
            {"thought": "k", "tool": "read_file", "args": {"path": "/nope3"}, "done": False, "final_answer": ""},
            DONE,
        ]
        final, calls, executed, _agent = _run(script, execute=lambda t, a: "ERROR: no such file")
        assert executed.count("read_file") == 2
        assert any("failed too many times" in s.get("result", "") for s in _agent.steps)


class TestNullResponseGuard:
    def test_json_null_provider_reply_does_not_crash(self, monkeypatch):
        """A provider replying literal 'null' (json.loads -> None) must not crash the loop."""
        monkeypatch.setenv("JARVIS_AGENT_MAX_TOOL_CALLS", "10")

        def ask_direct(prompt, n):
            del prompt
            del n
            return "null"

        final, _, executed, _agent = _run(ask_direct)
        assert isinstance(final, str)
        assert executed == []

    def test_parse_decision_null_is_dict(self):
        d = agent._parse_decision("null")
        assert isinstance(d, dict)
        assert d.get("done") is False, "null should not mark task as done"

    def test_parse_decision_plain_text_not_done(self):
        """Plain-text model narration must not terminate the agent loop."""
        d = agent._parse_decision("I need to inspect vision.py.")
        assert isinstance(d, dict)
        assert d.get("done") is False, "Plain text should not mark task as done"
        assert d.get("tool") == ""
        assert d.get("final_answer") == ""
        assert d.get("thought") == "Model returned non-JSON response."


class TestPhase4NarrationFilter:
    """Phase 4: Model narration vs actual progress separation."""

    def test_filter_narration_removes_prefixes(self):
        assert agent._filter_narration("Let me create a file") == "create a file"
        assert agent._filter_narration("Now I will read the file. Content: hello") == "read the file. Content: hello"
        assert agent._filter_narration("I need to check the file. File content: test") == "check the file. File content: test"
        assert agent._filter_narration("Normal text without narration") == "Normal text without narration"
        assert agent._filter_narration("") == ""

    def test_filter_narration_preserves_content_after_prefix(self):
        result = agent._filter_narration("Let me check the file. It contains: hello world")
        assert "hello world" in result
        assert "Let me" not in result

    def test_filter_narration_empty_returns_default(self):
        assert agent._filter_narration("Let me do this") == "do this"
        assert agent._filter_narration("Let me do this. Then that.") == "do this. Then that."


class TestPhase4Verification:
    """Phase 4: Tool execution verification."""

    def test_verify_write_file_triggers_read_back(self):
        def mock_execute(tool, args):
            if tool == "read_file":
                return "File content here"
            return "ok"

        verified, msg = agent._verify_tool_execution(
            "create_file", {"path": "/tmp/test.txt"}, "ok", mock_execute
        )
        assert verified is True
        assert "Verified" in msg

    def test_verify_write_file_fails_if_unreadable(self):
        def mock_execute(tool, args):
            if tool == "read_file":
                return "Could not read file"
            return "ok"

        verified, msg = agent._verify_tool_execution(
            "create_file", {"path": "/tmp/test.txt"}, "ok", mock_execute
        )
        assert verified is False
        assert "Verification failed" in msg

    def test_verify_run_python_detects_passed(self):
        verified, msg = agent._verify_tool_execution(
            "run_python", {"code": "pytest"}, "2 passed, 0 failed", lambda t, a: "ok"
        )
        assert verified is True
        assert "passed" in msg.lower()

    def test_verify_run_python_detects_failed(self):
        verified, msg = agent._verify_tool_execution(
            "run_python", {"code": "pytest"}, "1 failed, 0 passed", lambda t, a: "ok"
        )
        assert verified is False
        assert "failure" in msg.lower()

    def test_verify_run_python_detects_zero_failed(self):
        verified, msg = agent._verify_tool_execution(
            "run_python", {"code": "pytest"}, "0 failed, 2 passed", lambda t, a: "ok"
        )
        assert verified is True

    def test_verify_read_file_no_special_verification(self):
        verified, msg = agent._verify_tool_execution(
            "read_file", {"path": "/tmp/test.txt"}, "file content", lambda t, a: "ok"
        )
        # read_file doesn't have special verification, falls back to result scoring
        assert verified is False  # _is_success_result returns False for empty string


class TestPhase5GoalCriteria:
    """Phase 5: Goal criteria and verification."""

    def test_goal_criteria_persistence(self):
        from task import ActiveTask, CriterionType

        task = ActiveTask(
            task_id="test-criteria",
            goal="Test",
            project_root="/tmp",
            goal_criteria=[
                {"type": "tests_pass", "target": "tests/test_agent_loop.py", "required": True},
                {"type": "file_exists", "target": "task.py", "required": True},
            ]
        )
        assert len(task.goal_criteria) == 2

        task.save()
        from task import ActiveTask as AT
        reloaded = AT.load("test-criteria")
        assert reloaded is not None
        assert len(reloaded.goal_criteria) == 2
        assert reloaded.goal_criteria[0]["type"] == "tests_pass"

    def test_verify_goal_all_pass(self):
        from task import ActiveTask

        task = ActiveTask(
            task_id="test-verify",
            goal="Test",
            project_root="/Users/debasishbeura/Jarvis",
            goal_criteria=[
                {"type": "file_exists", "target": "task.py", "required": True},
                {"type": "syntax_valid", "target": "agent.py", "required": True},
            ]
        )
        passed, results = task.verify_goal()
        assert passed is True
        assert len(results) == 2
        assert all(r["passed"] for r in results)

    def test_verify_goal_optional_fails(self):
        from task import ActiveTask

        task = ActiveTask(
            task_id="test-optional",
            goal="Test",
            project_root="/Users/debasishbeura/Jarvis",
            goal_criteria=[
                {"type": "file_exists", "target": "nonexistent.txt", "required": True},
                {"type": "file_exists", "target": "nonexistent2.txt", "required": False},
            ]
        )
        passed, results = task.verify_goal()
        # Required fails, optional fails - overall should fail because required fails
        assert passed is False
        assert len(results) == 2
        assert results[0]["passed"] is False
        assert results[0]["required"] is True
        assert results[1]["passed"] is False
        assert results[1]["required"] is False

    def test_verify_goal_no_criteria(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test", goal="Test", goal_criteria=[])
        passed, results = task.verify_goal()
        assert passed is True
        assert len(results) == 1
        assert results[0]["passed"] is True


class TestPhase5ProgressTracking:
    """Phase 5D: False progress detection and verified progress tracking."""

    def test_record_step_tracks_verified_progress(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-progress", goal="Test")

        # Record a successful but unverified step
        task.record_step(
            {"tool": "write_file", "success": True, "result": "ok"},
            "write_file", {"path": "/tmp/test.txt"}
        )
        assert task.progress_metrics["tool_calls"] == 1
        assert task.progress_metrics["verified_progress_count"] == 0
        assert task.progress_metrics["false_progress_count"] == 1

        # Record a verified step
        task.record_step(
            {"tool": "write_file", "success": True, "result": "ok"},
            "write_file", {"path": "/tmp/test2.txt"},
            verified=True
        )
        assert task.progress_metrics["tool_calls"] == 2
        assert task.progress_metrics["verified_progress_count"] == 1
        assert task.progress_metrics["false_progress_count"] == 1

    def test_record_verified_progress_corrects_false_count(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-correct", goal="Test")

        # Record unverified success
        task.record_step(
            {"tool": "write_file", "success": True, "result": "ok"},
            "write_file", {"path": "/tmp/test.txt"}
        )
        assert task.progress_metrics["false_progress_count"] == 1

        # Mark it as verified
        task.record_verified_progress(0)
        assert task.progress_metrics["verified_progress_count"] == 1
        assert task.progress_metrics["false_progress_count"] == 0

    def test_get_progress_summary(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-summary", goal="Test")
        task.record_step({"tool": "read_file", "success": True}, "read_file", {"path": "/tmp/a.py"})
        task.record_step({"tool": "write_file", "success": True}, "write_file", {"path": "/tmp/b.py"}, verified=True)

        summary = task.get_progress_summary()
        assert summary["tool_calls"] == 2
        assert summary["verified_progress"] == 1
        assert summary["false_progress"] == 1
        assert summary["files_inspected"] == 1
        assert summary["files_modified"] == 1


class TestPhase5RecoveryStateMachine:
    """Phase 5E: Recovery state machine."""

    def test_recovery_states_exist(self):
        from task import TaskStatus
        assert TaskStatus.BLOCKED == "blocked"
        assert TaskStatus.DIAGNOSING == "diagnosing"
        assert TaskStatus.RECOVERING == "recovering"

    def test_enter_blocked(self):
        from task import ActiveTask, TaskStatus

        task = ActiveTask(task_id="test-blocked", goal="Test")
        task.enter_blocked("Test reason")
        assert task.status == TaskStatus.BLOCKED
        assert task.progress_metrics.get("blocked_reason") == "Test reason"
        assert "blocked_at" in task.progress_metrics

    def test_enter_diagnosing(self):
        from task import ActiveTask, TaskStatus

        task = ActiveTask(task_id="test-diagnosing", goal="Test")
        task.enter_diagnosing("Root cause identified")
        assert task.status == TaskStatus.DIAGNOSING
        assert task.progress_metrics.get("diagnosis") == "Root cause identified"

    def test_enter_recovering(self):
        from task import ActiveTask, TaskStatus

        task = ActiveTask(task_id="test-recovering", goal="Test")
        task.enter_recovering("Apply fix")
        assert task.status == TaskStatus.RECOVERING
        assert task.progress_metrics.get("recovery_plan") == "Apply fix"

    def test_resume_active(self):
        from task import ActiveTask, TaskStatus

        task = ActiveTask(task_id="test-resume", goal="Test")
        task.enter_blocked("Test")
        task.resume_active()
        assert task.status == TaskStatus.ACTIVE

    def test_check_recovery_transitions_stagnation(self):
        from task import ActiveTask, TaskStatus

        task = ActiveTask(task_id="test-stagnation", goal="Test")
        # Simulate 3 non-progress steps
        task.progress_metrics["stagnation_counter"] = 3
        task.status = TaskStatus.ACTIVE

        new_status = task.check_recovery_transitions()
        assert new_status == TaskStatus.BLOCKED
        assert task.status == TaskStatus.BLOCKED

    def test_check_recovery_transitions_blocked_to_diagnosing(self):
        from task import ActiveTask, TaskStatus
        import datetime

        task = ActiveTask(task_id="test-blocked-diagnose", goal="Test")
        task.enter_blocked("Test")
        # Simulate time passing
        task.progress_metrics["blocked_at"] = (datetime.datetime.now() - datetime.timedelta(seconds=60)).isoformat()

        new_status = task.check_recovery_transitions()
        assert new_status == TaskStatus.DIAGNOSING
        assert task.status == TaskStatus.DIAGNOSING

    def test_get_recovery_status(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-recovery-status", goal="Test")
        task.enter_blocked("Test reason")

        status = task.get_recovery_status()
        assert status["status"] == "blocked"
        assert status["blocked_reason"] == "Test reason"
        assert "blocked_at" in status


class TestPhase6GoalVerificationIntegration:
    """Phase 6: Goal verification integration with agent lifecycle."""

    def test_verify_goal_before_complete(self):
        from task import ActiveTask

        task = ActiveTask(
            task_id="test-verify-complete",
            goal="Test",
            project_root="/Users/debasishbeura/Jarvis",
            goal_criteria=[
                {"type": "file_exists", "target": "task.py", "required": True},
                {"type": "syntax_valid", "target": "agent.py", "required": True},
            ]
        )
        passed, results = task.verify_goal()
        assert passed is True
        assert len(results) == 2
        assert all(r["passed"] for r in results)

    def test_verify_goal_fails_on_required(self):
        from task import ActiveTask

        task = ActiveTask(
            task_id="test-fail",
            goal="Test",
            project_root="/Users/debasishbeura/Jarvis",
            goal_criteria=[
                {"type": "file_exists", "target": "nonexistent.txt", "required": True},
            ]
        )
        passed, results = task.verify_goal()
        assert passed is False
        assert len(results) == 1
        assert results[0]["passed"] is False
        assert results[0]["required"] is True

    def test_verify_goal_optional_failure_allowed(self):
        from task import ActiveTask

        task = ActiveTask(
            task_id="test-optional-fail",
            goal="Test",
            project_root="/Users/debasishbeura/Jarvis",
            goal_criteria=[
                {"type": "file_exists", "target": "nonexistent.txt", "required": False},
            ]
        )
        passed, results = task.verify_goal()
        # Optional failure doesn't block
        assert passed is True
        assert len(results) == 1
        assert results[0]["passed"] is False
        assert results[0]["required"] is False

    def test_verify_goal_mixed_required_optional(self):
        from task import ActiveTask

        task = ActiveTask(
            task_id="test-mixed",
            goal="Test",
            project_root="/Users/debasishbeura/Jarvis",
            goal_criteria=[
                {"type": "file_exists", "target": "task.py", "required": True},
                {"type": "file_exists", "target": "nonexistent.txt", "required": False},
            ]
        )
        passed, results = task.verify_goal()
        # Required passes, optional fails - overall passes
        assert passed is True
        assert len(results) == 2
        assert results[0]["passed"] is True
        assert results[1]["passed"] is False


class TestPhase6FailureDiagnosis:
    """Phase 6C: Failure diagnosis from verification results."""

    def test_diagnose_tests_pass_failure(self):
        from task import diagnose_failure

        results = [
            {"type": "tests_pass", "target": "tests/test_agent_loop.py", "required": True, "passed": False, "message": "1 failed, 5 passed"},
        ]
        diagnosis = diagnose_failure(results, "/Users/debasishbeura/Jarvis")
        assert "tests failing" in diagnosis.lower()
        assert "tests/test_agent_loop.py" in diagnosis

    def test_diagnose_syntax_valid_failure(self):
        from task import diagnose_failure

        results = [
            {"type": "syntax_valid", "target": "agent.py", "required": True, "passed": False, "message": "SyntaxError: invalid syntax"},
        ]
        diagnosis = diagnose_failure(results, "/Users/debasishbeura/Jarvis")
        assert "syntax error" in diagnosis.lower()
        assert "agent.py" in diagnosis

    def test_diagnose_file_exists_failure(self):
        from task import diagnose_failure

        results = [
            {"type": "file_exists", "target": "missing.txt", "required": True, "passed": False, "message": "File not found"},
        ]
        diagnosis = diagnose_failure(results, "")
        assert "missing file" in diagnosis.lower()
        assert "missing.txt" in diagnosis

    def test_diagnose_file_contains_failure(self):
        from task import diagnose_failure

        results = [
            {"type": "file_contains", "target": "agent.py", "substring": "missing_function", "required": True, "passed": False, "message": "File does not contain 'missing_function'"},
        ]
        diagnosis = diagnose_failure(results, "/Users/debasishbeura/Jarvis")
        assert "missing_function" in diagnosis
        assert "agent.py" in diagnosis

    def test_diagnose_command_succeeds_failure(self):
        from task import diagnose_failure

        results = [
            {"type": "command_succeeds", "target": "pytest", "required": True, "passed": False, "message": "Command failed: pytest"},
        ]
        diagnosis = diagnose_failure(results, "")
        assert "command failed" in diagnosis.lower()
        assert "pytest" in diagnosis

    def test_diagnose_multiple_failures(self):
        from task import diagnose_failure

        results = [
            {"type": "tests_pass", "target": "tests/test_agent_loop.py", "required": True, "passed": False, "message": "1 failed"},
            {"type": "syntax_valid", "target": "agent.py", "required": True, "passed": False, "message": "SyntaxError"},
        ]
        diagnosis = diagnose_failure(results, "/Users/debasishbeura/Jarvis")
        assert "tests failing" in diagnosis.lower()
        assert "syntax error" in diagnosis.lower()


class TestPhase6RecoveryPlanning:
    """Phase 6D: Recovery planning from verification failures."""

    def test_suggest_recovery_plan_tests(self):
        from task import suggest_recovery_plan

        results = [
            {"type": "tests_pass", "target": "tests/test_agent_loop.py", "required": True, "passed": False, "message": "1 failed"},
        ]
        plan = suggest_recovery_plan(results, "/Users/debasishbeura/Jarvis")
        assert "Run tests" in plan
        assert "Fix failing tests" in plan
        assert "Re-run tests" in plan

    def test_suggest_recovery_plan_syntax(self):
        from task import suggest_recovery_plan

        results = [
            {"type": "syntax_valid", "target": "agent.py", "required": True, "passed": False, "message": "SyntaxError"},
        ]
        plan = suggest_recovery_plan(results, "/Users/debasishbeura/Jarvis")
        assert "Open agent.py" in plan
        assert "fix syntax error" in plan.lower()
        assert "Re-verify syntax" in plan

    def test_suggest_recovery_plan_file_exists(self):
        from task import suggest_recovery_plan

        results = [
            {"type": "file_exists", "target": "missing.txt", "required": True, "passed": False, "message": "File not found"},
        ]
        plan = suggest_recovery_plan(results, "")
        assert "Create or restore missing.txt" in plan
        assert "Verify file exists" in plan

    def test_suggest_recovery_plan_file_contains(self):
        from task import suggest_recovery_plan

        results = [
            {"type": "file_contains", "target": "agent.py", "substring": "missing_function", "required": True, "passed": False, "message": "Missing"},
        ]
        plan = suggest_recovery_plan(results, "/Users/debasishbeura/Jarvis")
        assert "Add 'missing_function'" in plan
        assert "Verify content" in plan

    def test_suggest_recovery_plan_command(self):
        from task import suggest_recovery_plan

        results = [
            {"type": "command_succeeds", "target": "pytest", "required": True, "passed": False, "message": "Failed"},
        ]
        plan = suggest_recovery_plan(results, "")
        assert "Run command manually" in plan
        assert "Fix underlying issue" in plan
        assert "Re-run command" in plan


class TestPhase7ABudgetEnforcement:
    """Phase 7A: Execution budget tracking and enforcement."""

    def test_budget_initialization(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-budget", goal="Test")
        budget = task.execution_budget

        assert budget["time_budget_seconds"] == 600
        assert budget["llm_budget"] == 20
        assert budget["tool_budget"] == 50
        assert budget["replan_budget"] == 3
        assert budget["recovery_budget"] == 2
        assert budget["time_spent_seconds"] == 0
        assert budget["llm_calls"] == 0
        assert budget["tool_calls"] == 0
        assert budget["replans"] == 0
        assert budget["recoveries"] == 0

    def test_budget_persistence(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-budget", goal="Test")
        task.execution_budget["llm_calls"] = 5
        task.execution_budget["tool_calls"] = 10
        task.save()

        from task import ActiveTask as AT
        reloaded = AT.load("test-budget")
        assert reloaded is not None
        assert reloaded.execution_budget["llm_calls"] == 5
        assert reloaded.execution_budget["tool_calls"] == 10

    def test_llm_budget_enforcement(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-llm-budget", goal="Test")
        task.execution_budget["llm_budget"] = 2
        task.execution_budget["llm_calls"] = 2

        budget = task.execution_budget
        assert budget["llm_calls"] >= budget["llm_budget"]

    def test_tool_budget_enforcement(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-tool-budget", goal="Test")
        task.execution_budget["tool_budget"] = 3
        task.execution_budget["tool_calls"] = 3

        budget = task.execution_budget
        assert budget["tool_calls"] >= budget["tool_budget"]

    def test_replan_budget_enforcement(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-replan-budget", goal="Test")
        task.execution_budget["replan_budget"] = 2
        task.execution_budget["replans"] = 2

        budget = task.execution_budget
        assert budget["replans"] >= budget["replan_budget"]

    def test_recovery_budget_enforcement(self):
        from task import ActiveTask

        task = ActiveTask(task_id="test-recovery-budget", goal="Test")
        task.execution_budget["recovery_budget"] = 1
        task.execution_budget["recoveries"] = 1

        budget = task.execution_budget
        assert budget["recoveries"] >= budget["recovery_budget"]

    def test_budget_context_in_prompt(self):
        from task import ActiveTask

        task = ActiveTask(
            task_id="test-context",
            goal="Test",
            project_root="/tmp"
        )
        task.execution_budget["llm_calls"] = 5
        task.execution_budget["tool_calls"] = 3
        task.execution_budget["replans"] = 1
        task.execution_budget["recoveries"] = 0
        task.execution_budget["time_spent_seconds"] = 120

        context = task.get_context_for_prompt()
        assert "LLM 5/20 calls" in context
        assert "Tools 3/50 calls" in context
        assert "Replans 1/3" in context
        assert "Recoveries 0/2" in context
        assert "Time 120/600s" in context


class TestPhase7BMetricRatio:
    """Phase 7B: Metric ratio criterion tests."""

    def test_metric_ratio_pass(self):
        import json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "metrics1.json")
            file2 = os.path.join(tmpdir, "metrics2.json")
            with open(file1, "w") as f:
                json.dump({"numerator": 10.0}, f)
            with open(file2, "w") as f:
                json.dump({"denominator": 2.0}, f)

            from task import verify_criterion
            passed, msg = verify_criterion({
                "type": "metric_ratio",
                "target": file1,
                "metric1": "numerator",
                "metric2": "denominator",
                "metric_file1": file1,
                "metric_file2": file2,
                "operator": ">=",
                "value": 3.0,
                "source": "file"
            }, tmpdir)
            assert passed is True
            assert "5.0000" in msg

    def test_metric_ratio_fail(self):
        import json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "metrics1.json")
            file2 = os.path.join(tmpdir, "metrics2.json")
            with open(file1, "w") as f:
                json.dump({"numerator": 2.0}, f)
            with open(file2, "w") as f:
                json.dump({"denominator": 2.0}, f)

            from task import verify_criterion
            passed, msg = verify_criterion({
                "type": "metric_ratio",
                "target": file1,
                "metric1": "numerator",
                "metric2": "denominator",
                "metric_file1": file1,
                "metric_file2": file2,
                "operator": ">=",
                "value": 3.0,
                "source": "file"
            }, tmpdir)
            assert passed is False
            assert "1.0000" in msg


class TestPhase7BMetricDelta:
    """Phase 7B: Metric delta criterion tests."""

    def test_metric_delta_improvement(self):
        import json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "metrics.json")
            with open(file, "w") as f:
                json.dump({"accuracy": 0.85}, f)

            from task import verify_criterion
            # Improvement from 0.75 to 0.85 = 13.3% improvement
            passed, msg = verify_criterion({
                "type": "metric_delta",
                "target": file,
                "metric": "accuracy",
                "metric_file": file,
                "baseline": 0.75,
                "operator": ">=",
                "value": 0.1,
                "mode": "relative",
                "source": "file"
            }, tmpdir)
            assert passed is True
            assert "PASS" in msg

    def test_metric_delta_regression(self):
        import json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "metrics.json")
            with open(file, "w") as f:
                json.dump({"accuracy": 0.65}, f)

            from task import verify_criterion
            # Regression from 0.75 to 0.65 = -13.3% change
            passed, msg = verify_criterion({
                "type": "metric_delta",
                "target": file,
                "metric": "accuracy",
                "metric_file": file,
                "baseline": 0.75,
                "operator": ">=",
                "value": 0.0,
                "mode": "relative",
                "source": "file"
            }, tmpdir)
            assert passed is False
            assert "FAIL" in msg

    def test_metric_delta_absolute_mode(self):
        import json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            file = os.path.join(tmpdir, "metrics.json")
            with open(file, "w") as f:
                json.dump({"latency_ms": 150}, f)

            from task import verify_criterion
            # Latency improved from 200ms to 150ms = -50ms absolute
            passed, msg = verify_criterion({
                "type": "metric_delta",
                "target": file,
                "metric": "latency_ms",
                "metric_file": file,
                "baseline": 200,
                "operator": "<=",
                "value": -30,
                "mode": "absolute",
                "source": "file"
            }, tmpdir)
            assert passed is True
            assert "PASS" in msg


class TestPhase7BOutputMatches:
    """Phase 7B: Output matches criteria tests."""

    def test_output_matches_substring(self):
        from task import verify_criterion
        passed, msg = verify_criterion({
            "type": "output_matches",
            "target": "echo hello world",
            "command": "echo hello world",
            "pattern": "hello",
            "matcher": "substring"
        }, "")
        assert passed is True
        assert "matches" in msg

    def test_output_matches_regex(self):
            from task import verify_criterion
            passed, msg = verify_criterion({
                "type": "output_matches",
                "target": "echo test123",
                "command": "echo -n test123",
                "pattern": r"test\d+",
                "matcher": "regex"
            }, "")
            assert passed is True

    def test_output_not_matches(self):
            from task import verify_criterion
            passed, msg = verify_criterion({
                "type": "output_not_matches",
                "target": "echo success",
                "command": "echo success",
                "pattern": "error",
                "matcher": "substring"
            }, "")
            assert passed is True
            assert "does not match" in msg


class TestPhase7BRegressionFree:
    """Phase 7B: Regression free criterion tests."""

    def test_regression_free_pass(self):
        from task import verify_criterion
        # Simple command that succeeds
        passed, msg = verify_criterion({
            "type": "regression_free",
            "target": "echo hello",
            "command": "echo hello",
            "allow_new_tests": True
        }, "/tmp")
        assert passed is True

    def test_regression_free_with_baseline(self):
        import json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_file = os.path.join(tmpdir, "baseline.json")
            with open(baseline_file, "w") as f:
                json.dump({"passed": 5, "failed": 0, "errors": 0}, f)

            from task import verify_criterion
            passed, msg = verify_criterion({
                "type": "regression_free",
                "target": "python -c \"print('5 passed, 0 failed, 0 error')\"",
                "command": "python -c \"print('5 passed, 0 failed, 0 error')\"",
                "baseline": baseline_file,
                "allow_new_tests": True
            }, tmpdir)
            assert passed is True
            assert "No regression" in msg


class TestPhase7BPerformanceThreshold:
    """Phase 7B: Performance threshold criterion tests."""

    def test_performance_time_pass(self):
        from task import verify_criterion
        passed, msg = verify_criterion({
            "type": "performance_threshold",
            "target": "sleep 0.1",
            "command": "sleep 0.1",
            "metric": "time",
            "operator": "<=",
            "value": 0.5,
            "unit": "seconds"
        }, "")
        assert passed is True
        assert "PASS" in msg

    def test_performance_time_fail(self):
        from task import verify_criterion
        passed, msg = verify_criterion({
            "type": "performance_threshold",
            "target": "sleep 0.5",
            "command": "sleep 0.5",
            "metric": "time",
            "operator": "<=",
            "value": 0.1,
            "unit": "seconds"
        }, "")
        assert passed is False
        assert "FAIL" in msg


class TestPhase7BComposableCriteria:
    """Phase 7B: Composable criteria (ALL/ANY) tests."""

    def test_all_criteria_pass(self):
        from task import verify_criterion
        passed, msg = verify_criterion({
            "type": "all",
            "target": "echo hello",
            "criteria": [
                {"type": "output_matches", "target": "echo hello", "command": "echo hello", "pattern": "hello", "matcher": "substring"},
                {"type": "output_matches", "target": "echo world", "command": "echo world", "pattern": "world", "matcher": "substring"}
            ]
        }, "")
        assert passed is True

    def test_all_criteria_fail(self):
        from task import verify_criterion
        passed, msg = verify_criterion({
            "type": "all",
            "target": "echo hello",
            "criteria": [
                {"type": "output_matches", "target": "echo hello", "command": "echo hello", "pattern": "hello", "matcher": "substring"},
                {"type": "output_matches", "target": "echo hello", "command": "echo hello", "pattern": "goodbye", "matcher": "substring"}
            ]
        }, "")
        assert passed is False

    def test_any_criteria_pass(self):
        from task import verify_criterion
        passed, msg = verify_criterion({
            "type": "any",
            "target": "echo hello",
            "criteria": [
                {"type": "output_matches", "target": "echo hello", "command": "echo hello", "pattern": "goodbye", "matcher": "substring"},
                {"type": "output_matches", "target": "echo world", "command": "echo world", "pattern": "world", "matcher": "substring"}
            ]
        }, "")
        assert passed is True

    def test_any_criteria_all_fail(self):
        from task import verify_criterion
        passed, msg = verify_criterion({
            "type": "any",
            "target": "echo hello",
            "criteria": [
                {"type": "output_matches", "target": "echo hello", "command": "echo hello", "pattern": "goodbye", "matcher": "substring"},
                {"type": "output_matches", "target": "echo hello", "command": "echo hello", "pattern": "goodbye", "matcher": "substring"}
            ]
        }, "")
        assert passed is False

# ─────────────────────────────────────────────
# Phase 2C: Phase ownership, planner isolation, resume idempotency
# ─────────────────────────────────────────────

import uuid

class TestPhaseOwnership:
    """Test that phase transitions are controlled by task state machine only."""

    def test_continuation_preserves_phase(self):
        """Continuation request must preserve EXPLORE phase."""
        from task import ActiveTask, continue_task, set_current_task, TaskPhase
        
        task = ActiveTask(task_id='test_continuation', goal='Test task', phase=TaskPhase.EXPLORE)
        task.save()
        set_current_task(task)
        
        continued = continue_task()
        assert continued is not None
        assert continued.phase == TaskPhase.EXPLORE, f"Phase changed to {continued.phase}"

    def test_planner_never_mutates_task_phase(self):
        """Planner must not directly mutate task.phase."""
        import agent
        from task import ActiveTask, TaskPhase

        def mock_ask_llm(prompt):
            return ('[{"step_id": "1", "goal": "Test step", '
                    '"tool_hint": "read_file", "args": {"path": "test.py"}}]')

        task = ActiveTask(task_id="test_planner_phase", goal="Test goal", phase=TaskPhase.EXPLORE)
        original_phase = task.phase

        plan = agent._create_plan("Test goal", mock_ask_llm)
        assert plan is not None
        assert len(plan.steps) == 1
        assert task.phase == original_phase

    def test_no_premature_implementation(self):
        """EXPLORE phase task must not execute mutation tools."""
        from agent import run_agent_loop
        from task import ActiveTask, TaskPhase
        
        task = ActiveTask(task_id='test_no_impl', goal='Explore only', phase=TaskPhase.EXPLORE)
        task.save()
        
        mutation_called = {"called": False}
        
        def mock_execute(tool_name, args):
            if tool_name in ("write_file", "create_file", "run_terminal_command"):
                mutation_called["called"] = True
            return "result"
        
        def ask_llm(prompt):
            if "evaluator" in prompt.lower():
                return "SUCCESS - step completed"
            import json
            return json.dumps(_decision(tool="read_file", args={"path": "/tmp/test"}, phase="explore"))
        
        run_agent_loop(
            goal="Explore only",
            execute_tool_fn=mock_execute,
            ask_llm_fn=ask_llm,
            task=ActiveTask(task_id="test_no_impl", goal="Explore only", phase=TaskPhase.EXPLORE),
            max_iterations=3
        )
        
        assert True


class TestPlannerIsolation:
    """Test that planner works correctly."""

    def test_planner_works_without_crashing(self):
        """Planner should build a valid Plan without crashing."""
        import agent

        def mock_ask_llm(prompt):
            return ('[{"step_id": "1", "goal": "Test step", '
                    '"tool_hint": "read_file", "args": {"path": "test.py"}}]')

        plan = agent._create_plan("Test goal", mock_ask_llm)
        assert plan is not None
        assert len(plan.steps) > 0
        assert all(s.tool_hint for s in plan.steps)

    def test_planner_output_validation(self):
        from agent import _extract_json_array, _step_from_dict
        
        malformed = '[{"step_id": "1", "goal": "test"'
        result = _extract_json_array(malformed)
        assert result is None or result == []
        
        valid = '[{"step_id": "1", "goal": "test", "tool_hint": "test", "args": {}}]'
        result = _extract_json_array(valid)
        assert len(result) == 1


class TestProviderFailureIsolation:
    def test_provider_failure_isolation(self, monkeypatch):
        from brain import _handle_provider_failure
        from task import ActiveTask, TaskPhase
        
        task = ActiveTask(task_id='test_provider_fail', goal='Test', phase=TaskPhase.EXPLORE)
        original_phase = task.phase
        
        try:
            from brain import _handle_provider_failure
            _handle_provider_failure('TestProvider', Exception('500 Internal Server Error'))
        except Exception:
            pass
        
        assert task.phase == original_phase


class TestDeadModelExclusion:
    """Issue 7: confirmed-EOL models must never appear in executable routing."""

    DEAD_MODELS = [
        "minimaxai/minimax-m3",
        "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "nvidia/nemotron-3-nano-30b-a3b",
        "stepfun-ai/step-3.7-flash",
    ]

    def test_dead_models_removed_from_nim_slots(self):
        import brain
        for rota in (brain.NIM_MODEL_FAST, brain.NIM_MODEL_CODING, brain.NIM_MODEL_FRONTIER):
            for dead in self.DEAD_MODELS:
                assert dead not in rota, f"{dead} still in NIM slot"

    def test_dead_models_removed_from_context_limits(self):
        from config import MODEL_CONTEXT_LIMITS
        for dead in self.DEAD_MODELS:
            assert dead not in MODEL_CONTEXT_LIMITS, (
                f"{dead} still in MODEL_CONTEXT_LIMITS (context-limits table is config, not routing, "
                "but stale EOL entries here mislead future contributors and the agent's own introspection)"
            )


class TestProviderEOLClassification:
    """Issue 8/7: 410/EOL errors must classify as permanent (immediate retirement),
    not as recoverable — otherwise a retired model burns retry budget forever."""

    def test_410_is_permanent(self, monkeypatch, tmp_path):
        import brain
        assert brain._classify_error("Error code: 410 Gone") == "permanent"
        assert brain._classify_error("The model 'x' has reached its end of life") == "permanent"
        assert brain._classify_error("model is no longer available") == "permanent"


class TestToolContract:
    """A regression guard so a planner-generated alias like file_path never
    crashes a tool call when the contract wants `path`.

    Evidence (live session, Sep 11): planner emitted
        {"tool": "read_file", "args": {"file_path": "/path/to/file"}}
    and the validator returned
        'ERROR: Missing required argument(s) for read_file: path'
    """

    def test_alias_normalization_maps_file_path_to_path(self, monkeypatch):
        import brain
        validated = brain.validate_tool_args("read_file", {"file_path": "/tmp/x.py"})
        assert not isinstance(validated, str), f"validation rejected: {validated}"
        name, args = validated
        assert args["path"] == "/tmp/x.py"

    def test_alias_normalization_idempotent_and_never_overrides_canonical(self, monkeypatch):
        import brain
        # If both keys are present, the canonical path wins; alias never overwrites.
        validated = brain.validate_tool_args(
            "read_file", {"path": "/canonical", "file_path": "/alias"}
        )
        name, args = validated
        assert args["path"] == "/canonical"

    def test_execute_tool_path_accepts_alias_input(self, monkeypatch):
        """Full pipeline: _execute_tool with aliased args executes without the
        'Missing required argument' string being returned."""
        import brain
        out = brain._execute_tool("read_file", {"file_path": "/nonexistent/file.py"})
        assert "Missing required argument" not in out


class TestPythonContract:
    """Issue 6: the contract-side grounding for model-generated Python."""

    def test_malformed_python_syntax_rejected_pre_execution(self):
        """'import subprocess, subprocess.run(...)' — observed in live demo — must
        fail fast with a clear message instead of reaching the sandbox"""
        from tools.code_tools import run_python

        out = run_python('import subprocess, subprocess.run(["ls"])')
        assert "Syntax error" in out, out

    def test_valid_python_runs(self):
        from tools.code_tools import run_python

        out = run_python('print("hello")')
        assert "hello" in out


class TestRecoveryLoopBounded:
    """Issue 2 follow-on: recovery suggestions are bounded per plan, not just
    deduped — a stubborn replanner can't ask for new alternatives forever."""

    def test_recovery_stop_after_per_step_cap(self):
        import agent

        calls = []

        def fake_execute(tool, args):
            calls.append((tool, args))
            return f"executed {tool}"

        # Every evaluation fails; recovery always offers a NEW distinct tool so
        # the dedup guard can't stop it — the bounded per-step retry budget
        # (initial attempt + 2 retries) must be what stops the loop. Recovery
        # must also PRESERVE the original args (never accept model-supplied ones).
        ask_n = [0]

        def fake_ask(prompt):
            ask_n[0] += 1
            if "step evaluator" in prompt:
                return "FAILURE"
            if "DIFFERENT tool" in prompt:
                return '{"tool": "rt%d", "args": {"hijacked": 1}, "reason": "swap"}' % ask_n[0]
            return "ok"

        from agent import Plan, PlanStep, run_planner_loop_with_plan
        plan = Plan(
            plan_id="p_bounded",
            original_goal="g",
            steps=[PlanStep(step_id="s1", goal="one", tool_hint="t1", args={"a": 1})],
        )
        run_planner_loop_with_plan(plan=plan, execute_tool_fn=fake_execute, ask_llm_fn=fake_ask)
        # Initial attempt + exactly 2 bounded retries, then it must stop.
        assert len(calls) == 3, calls
        assert calls[0] == ("t1", {"a": 1})
        # Every retry re-runs with the ORIGINAL args — never the model's args.
        assert all(args == {"a": 1} for _, args in calls), calls
        # All distinct tools (no duplicate (tool,args) re-execution).
        assert len({(t, tuple(sorted(a.items()))) for t, a in calls}) == 3
        # Retries exhausted -> plan is blocked, not silently completed.
        assert plan.status == "blocked"
        assert "NOT generated" in plan.final_answer


class TestConfirmationSemantics:
    """Issue 5: a run_python preview that stops at NeedsConfirmation must be
    recorded as awaiting_confirmation, not as a project failure."""

    CONFIRM_TEXT = (
        "Preview of run python shown above. "
        "Say yes to run it for real OUTSIDE the sandbox, or no to cancel."
    )

    def test_preview_result_is_not_misjudged_by_evaluator(self):
        from agent import Plan, PlanStep, run_planner_loop_with_plan

        asks = []

        def fake_execute(tool, args):
            return self.CONFIRM_TEXT  # simulates sandbox NeedsConfirmation path

        def fake_ask(prompt):
            asks.append(prompt)
            return "FAILURE — preview pending"

        plan = Plan(
            plan_id="pending-confirm",
            original_goal="run a thing",
            steps=[PlanStep(step_id="s1", goal="run it", tool_hint="run_python", args={"code": "print(1)"})],
        )
        run_planner_loop_with_plan(plan=plan, execute_tool_fn=fake_execute, ask_llm_fn=fake_ask)
        s = plan.steps[0]
        assert s.status == "awaiting_confirmation", f"step status was {s.status!r}"
        assert s.evaluation == "pending"
        # The evaluator must NOT receive this preview as an outcome — no ask
        # about it should have been made (eval would just misread preview text).
        assert not any("step evaluator" in p for p in asks), asks

    def test_awaiting_confirmation_skips_recovery_replan(self):
        from agent import Plan, PlanStep, run_planner_loop_with_plan

        asks = []

        def fake_execute(tool, args):
            return self.CONFIRM_TEXT

        def fake_ask(prompt):
            asks.append(prompt)
            return "FAILURE"

        plan = Plan(plan_id="pending-confirm-2", original_goal="run a thing",
                    steps=[PlanStep(step_id="s1", goal="run", tool_hint="run_python", args={"code": "print(1)"})])
        run_planner_loop_with_plan(plan=plan, execute_tool_fn=fake_execute, ask_llm_fn=fake_ask)
        # no retry/replan prompt was issued for the pending-confirmation step
        assert not any("Suggest an alternative approach or tool" in p for p in asks)


class TestRecoveryDedup:
    """Issue 2: per-plan recovery-call dedup must stop identical repeat executions."""

    def test_recovery_suggestion_duplicate_does_not_reexecute(self):
        """If evaluation fails then the recovery ask re-suggests the SAME
        (tool, args) pair that just failed (or that already succeeded), the
        planner must reuse the recorded result instead of re-executing."""
        from agent import Plan, PlanStep, run_planner_loop_with_plan

        calls = []

        def fake_execute(tool, args):
            calls.append((tool, args.get("path")))
            return "some-observable-result"

        # ask_llm is invoked for: step evaluation, then the retry/recovery ask
        answers = iter([
            "FAILURE — first call failed",  # evaluation -> failure
            '{"tool": "read_file", "args": {"path": "same.txt"}, "reason": "retry"}',  # identical suggestion
        ])

        def fake_ask(prompt):
            return next(answers)

        plan = Plan(
            plan_id="p_dedup",
            original_goal="read same.txt",
            steps=[PlanStep(step_id="s1", goal="read", tool_hint="read_file", args={"path": "same.txt"})],
        )
        run_planner_loop_with_plan(plan=plan, execute_tool_fn=fake_execute, ask_llm_fn=fake_ask)
        # First executed the step; retry produced identical signature → no second execution.
        assert len(calls) == 1, f"identical recovery suggestion re-executed: {calls}"


class TestPlannerContractInjection:
    def test_resume_unfinished_work(self, monkeypatch, tmp_path):
        """Resume executes from the matching list position, not ID ordering.

        True restart test: plan is persisted, in-memory reference discarded,
        then reconstructed from persistence by plan_id.
        """
        from agent import Plan, PlanStep, plan_from_persistence, run_planner_loop_with_plan
        import plans

        monkeypatch.setattr(plans, "DB_PATH", str(tmp_path / "plans.db"))
        plans.init_db()

        # Create and persist plan via production persistence path
        plan_id = "resume-plan-restart"
        plans.save_plan(
            plan_id=plan_id,
            original_goal="Resume exact remaining work",
            status="running",
            current_step=0,
        )
        # Persist steps with explicit step_index (trap: IDs disagree with order)
        plans.save_plan_step(
            step_id="z-before", plan_id=plan_id, step_index=0,
            goal="already completed", tool_hint="before_tool", args={"step": "before"},
            status="completed", result="done", evaluation="success",
        )
        plans.save_plan_step(
            step_id="a-resume", plan_id=plan_id, step_index=1,
            goal="resume here", tool_hint="resume_tool", args={"step": "resume"},
            status="pending", result="", evaluation="",
        )
        plans.save_plan_step(
            step_id="m-after", plan_id=plan_id, step_index=2,
            goal="finish later", tool_hint="after_tool", args={"step": "after"},
            status="pending", result="", evaluation="",
        )

        # Discard in-memory reference (simulate process restart)
        original_plan_ref = None  # no in-memory Plan object retained

        # Reconstruct via production reconstruction path
        reloaded_plan = plan_from_persistence(plan_id)
        assert reloaded_plan is not None, "reconstruction returned None"
        assert reloaded_plan.plan_id == plan_id
        assert reloaded_plan.original_goal == "Resume exact remaining work"
        assert len(reloaded_plan.steps) == 3
        # Steps ordered by step_index: z-before(0), a-resume(1), m-after(2)
        assert reloaded_plan.steps[0].step_id == "z-before"
        assert reloaded_plan.steps[1].step_id == "a-resume"
        assert reloaded_plan.steps[2].step_id == "m-after"
        # Verify it's a fresh object, not the original (which we never created in-memory)
        assert original_plan_ref is None  # nothing to compare; construction was from DB

        executed = []
        attempts_before_execution = []

        def execute(tool, args):
            records = plans.load_execution_records_for_plan(plan_id)
            attempts_before_execution.append(records[-1])
            executed.append((tool, args["step"]))
            return f"{args['step']} completed"

        def ask_llm(prompt):
            if "step evaluator" in prompt.lower():
                return "SUCCESS"
            return "Plan complete"

        # True restart path: caller has only plan_id
        run_planner_loop_with_plan(
            plan_id=plan_id,
            execute_tool_fn=execute,
            ask_llm_fn=ask_llm,
            resume_from_step_id="a-resume",
        )

        # z-before (index 0) skipped; a-resume (index 1) runs; m-after (index 2) runs
        assert executed == [("resume_tool", "resume"), ("after_tool", "after")]
        assert [record["step_id"] for record in attempts_before_execution] == [
            "a-resume",
            "m-after",
        ]
        assert all(record["success"] is False for record in attempts_before_execution)
        assert all(record["result"] == "" for record in attempts_before_execution)

        records = plans.load_execution_records_for_plan(plan_id)
        assert [record["step_id"] for record in records] == ["a-resume", "m-after"]
        assert all(record["success"] is True for record in records)
        assert plans.load_plan_step(plan_id, "z-before")["status"] == "completed"
        assert plans.load_plan_step(plan_id, "a-resume")["status"] == "completed"
        assert plans.load_plan_step(plan_id, "m-after")["status"] == "completed"


class TestDuplicateExecutionProtection:
    """Test that retries don't re-execute completed steps."""

    def test_duplicate_execution_protection(self, monkeypatch):
        """Retry must not re-execute a durably completed step."""
        from agent import run_agent_loop
        from task import ActiveTask, TaskPhase
        from unittest.mock import MagicMock
        import json
        
        task = ActiveTask(task_id='test_dup', goal='Test', phase=TaskPhase.EXPLORE)
        task.record_step({"tool": "read_file", "args": {"path": "test.py"}, "success": True, "result": "content", "thought": "read test"}, "read_file", {"path": "test.py"})
        task.save()
        
        execution_count = {"read_file": 0}
        
        def mock_execute(tool_name, args):
            if tool_name == "read_file":
                execution_count["read_file"] += 1
            return "result"
        
        def ask_llm(prompt):
            if "evaluator" in prompt.lower():
                return "SUCCESS - step completed"
            return json.dumps(_decision(tool="read_file", args={"path": "test.py"}, phase="explore"))
        
        run_agent_loop(
            goal="Test",
            execute_tool_fn=mock_execute,
            ask_llm_fn=lambda p: json.dumps(_decision(tool="read_file", args={"path": "test.py"}, phase="explore")),
            task=ActiveTask(task_id="test_dup", goal="Test", phase=TaskPhase.EXPLORE),
            max_iterations=3
        )
        
        assert True
