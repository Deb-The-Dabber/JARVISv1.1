"""Phase 2.2.3C: plan-resume / continuation / bounded-retry / synthesis gate.

Covers the 8-bug research-continuation cluster:

  #1  imperative continuation commands ("run the searches", "do it", ...) must
      resolve to the CURRENT session's task — without hijacking data-plane
      requests ("run a marathon plan", "write a poem").
  #2/#5  resume runs persisted steps with skip-completed; failed steps re-run
      with their EXACT original query/args; tool replacement never rewrites args.
  #3  control instructions are routed to the executor (plan resume), never
      turned into a web_search query.
  #4  a continuation goes to the plan executor, not LLM prose.
  #6  bounded retry (initial + 2 additional attempts), permanent failures not
      pointlessly retried, synthesis gate: failed required step -> plan BLOCKED
      with an explicit missing-evidence report, artifact NOT generated, and a
      required step is never silently downgraded to optional.
  #7  confirmation broadening: "write the file"/"apply it" confirm a staged
      pending op, and confirming does NOT re-run the original request.
  #8  pending ops carry session ownership; cross-session applies are discarded.

The cross-session / stale-task protections are asserted throughout.
"""

import pytest

from agent import Plan, PlanStep, _normalize_tool_name, plan_from_persistence, run_planner_loop_with_plan


@pytest.fixture(autouse=True)
def _init_plans_schema(_isolate_persistent_state):
    """Create the plans schema AFTER conftest isolates the DB path."""
    import plans

    plans.init_db()
    yield


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

def _mk_plan(plan_id, steps):
    return Plan(
        plan_id=plan_id,
        original_goal="test goal",
        steps=[PlanStep(**s) for s in steps],
    )


def _persist_plan(plan_id, steps, status="running"):
    """Persist a plan + steps directly (as a process restart would see it)."""
    import plans

    plans.save_plan(plan_id, "test goal", status=status)
    for i, s in enumerate(steps):
        plans.save_plan_step(
            step_id=s["step_id"],
            plan_id=plan_id,
            step_index=i,
            goal=s["goal"],
            tool_hint=s["tool_hint"],
            args=s.get("args", {}),
            status=s.get("status", "pending"),
            result=s.get("result", ""),
            evaluation=s.get("evaluation", ""),
            required=s.get("required", True),
        )
    return plan_id


class _Eval:
    """Mock ask that judges steps by content and offers recovery tools."""

    def __init__(self, recovery_tools=("alt_search",), fail_term="timeout"):
        self.recovery_tools = recovery_tools
        self.fail_term = fail_term
        self.recovery_calls = 0
        self.eval_calls = 0
        self.summary_calls = 0
        self.calls = []

    def ask(self, prompt):
        self.calls.append(prompt)
        if "step evaluator" in prompt:
            self.eval_calls += 1
            return "FAILURE" if self.fail_term in prompt else "SUCCESS"
        if "DIFFERENT tool" in prompt or "alternative" in prompt:
            self.recovery_calls += 1
            if self.recovery_calls <= len(self.recovery_tools):
                t = self.recovery_tools[self.recovery_calls - 1]
                return '{"tool": "%s", "args": {"hijacked": 1}, "reason": "swap"}' % t
            return '{"tool": "", "reason": "none left"}'
        if "Summarize what was accomplished" in prompt:
            self.summary_calls += 1
            return "COMPLETED SUMMARY"
        return "ok"


def _recording_execute(results=None):
    """Mock execute that records (tool, args) and returns canned results."""
    calls = []

    def execute(tool, args):
        calls.append((tool, dict(args or {})))
        if results is None:
            return "executed " + tool
        return results.get(tool, "executed " + tool)

    return execute, calls


# ─────────────────────────────────────────────
# #1 continuation resolution
# ─────────────────────────────────────────────

class TestContinuationResolution:
    def test_imperative_continuation_resolves_to_current_task(self):
        import task

        t = task.create_task(goal="research hawaii volcano hazard zones", session_id="s1")
        r = task.resolve_task("s1", "run the searches")
        assert r["action"] == "resume"
        assert r["task"] and r["task"].task_id == t.task_id

    def test_do_it_and_retry_those_resolve_to_current_task(self):
        import task

        t = task.create_task(goal="investigate dr mamlapalli data", session_id="s2")
        for msg in ("do it", "retry those", "keep going", "write the file", "continue"):
            r = task.resolve_task("s2", msg)
            assert r["action"] == "resume", f"{msg!r} -> {r}"
            assert r["task"].task_id == t.task_id

    def test_data_plane_request_never_hijacks_task(self):
        import task

        task.create_task(goal="research hawaii volcano hazard zones", session_id="s3")
        for msg in ("run a marathon training plan", "write a poem about rain",
                    "retry the download", "do the laundry"):
            r = task.resolve_task("s3", msg)
            # No continuation binding -> fresh request; the phrase must NOT
            # hijack the current research task.
            assert r["action"] == "new", f"{msg!r} -> {r}"
            assert r["task"] is None

    def test_continuation_with_no_eligible_task_does_not_fabricate(self):
        import task

        r = task.resolve_task("empty-session", "run the searches")
        assert r["action"] == "none"
        assert r["task"] is None

    def test_cross_session_task_never_resumed(self):
        import task

        # Task belongs to session A; continuation arrives in session B.
        task.create_task(goal="hawaii research", session_id="owner-session")
        r = task.resolve_task("other-session", "run the searches")
        assert r["action"] in ("none", "new")
        assert r["task"] is None or r["task"].session_id == "other-session"


# ─────────────────────────────────────────────
# detect_control_command (#3 gate + no hijack)
# ─────────────────────────────────────────────

class TestControlCommandDetection:
    def test_recognizes_directives(self):
        import task

        assert any(d["type"] == "resume_plan" for d in task.detect_control_command("run the searches"))
        assert any(d["type"] == "resume_plan" for d in task.detect_control_command("continue"))
        assert any(d["type"] == "resume_plan" for d in task.detect_control_command("do it"))
        assert any(d["type"] == "retry_failed" for d in task.detect_control_command("retry those"))
        assert any(d["type"] == "retry_failed" for d in task.detect_control_command("retry the failed tool calls"))
        assert any(d["type"] == "apply_pending" for d in task.detect_control_command("write the file"))

    def test_apply_pending_sorts_first(self):
        import task

        d = task.detect_control_command("yes write the file and retry those")
        assert d[0]["type"] == "apply_pending"

    def test_replace_tool_detected(self):
        import task

        d = task.detect_control_command("retry the failed searches using normal web_search")
        assert any(dd.get("replace_tool") == "web_search" for dd in d)

    def test_data_plane_requests_are_not_control_commands(self):
        import task

        for msg in ("run a marathon training plan", "write a poem", "retry the download",
                    "apply for a job", "write an essay about history"):
            assert task.detect_control_command(msg) == [], f"{msg!r} produced directives"

    def test_no_message_yields_nothing(self):
        import task

        assert task.detect_control_command("") == []
        assert task.detect_control_command("   ") == []


# ─────────────────────────────────────────────
# #2/#5 resume semantics
# ─────────────────────────────────────────────

class TestResumeSemantics:
    def test_resume_skips_completed_steps(self):
        _persist_plan("rp-skip", [
            {"step_id": "c1", "goal": "done", "tool_hint": "web_search", "args": {"query": "a"},
             "status": "completed", "result": "ok", "evaluation": "success"},
            {"step_id": "c2", "goal": "pending", "tool_hint": "web_search", "args": {"query": "b"}},
        ])
        execute, calls = _recording_execute({"web_search": "found"})
        evalx = _Eval()
        run_planner_loop_with_plan(plan_id="rp-skip", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        # Only the pending step ran; the completed one was skipped.
        assert calls == [("web_search", {"query": "b"})], calls

    def test_resume_reruns_failed_step_with_exact_args(self):
        _persist_plan("rp-fail", [
            {"step_id": "f1", "goal": "search exact phrase", "tool_hint": "web_search",
             "args": {"query": "exact phrase to find"}, "status": "failed",
             "result": "boom", "evaluation": "failure"},
        ])
        execute, calls = _recording_execute({"web_search": "found results"})
        evalx = _Eval()
        run_planner_loop_with_plan(plan_id="rp-fail", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        assert calls and calls[0] == ("web_search", {"query": "exact phrase to find"}), calls

    def test_recovery_preserves_original_args(self):
        _persist_plan("rp-args", [
            {"step_id": "a1", "goal": "gather evidence", "tool_hint": "web_search",
             "args": {"query": "keep this exact phrase"}, "status": "failed",
             "result": "timeout", "evaluation": "failure"},
        ])
        execute, calls = _recording_execute({"web_search": "timeout", "alt_search": "found"})
        evalx = _Eval(recovery_tools=("alt_search",))
        run_planner_loop_with_plan(plan_id="rp-args", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        alt_calls = [a for t, a in calls if t == "alt_search"]
        assert alt_calls, calls
        # The recovery tool must run with the ORIGINAL query — never the
        # model-supplied {"hijacked": 1}.
        assert alt_calls[0] == {"query": "keep this exact phrase"}, alt_calls

    def test_plan_linked_to_task_and_loadable(self):
        import plans
        from task import ActiveTask, add_plan_to_task, create_task

        t = create_task(goal="link task", session_id="link-session")
        _persist_plan("rp-link", [{"step_id": "l1", "goal": "g", "tool_hint": "web_search", "args": {"q": "1"}}])
        add_plan_to_task(t.task_id, "rp-link")
        reloaded = ActiveTask.load(t.task_id)
        assert "rp-link" in reloaded.plan_ids
        # Both directions of the linkage are discoverable.
        found = plans.load_plan_for_task(t.task_id, exclude_status=("completed",))
        assert found and found["plan_id"] == "rp-link"

    def test_plan_from_persistence_restores_required(self):
        _persist_plan("rp-req", [
            {"step_id": "r1", "goal": "g1", "tool_hint": "web_search", "args": {"q": "1"}, "required": True},
            {"step_id": "r2", "goal": "g2", "tool_hint": "web_search", "args": {"q": "2"}, "required": False},
        ])
        p = plan_from_persistence("rp-req")
        assert [s.required for s in p.steps] == [True, False]

    def test_stale_plan_never_executed_under_another_task(self):
        """V5 ownership invariant: a plan owned by task A is rejected under task B."""
        from task import create_task

        _persist_plan("rp-owner", [{"step_id": "o1", "goal": "g", "tool_hint": "web_search", "args": {"q": "1"}}])
        p = plan_from_persistence("rp-owner")
        p.owner_task_id = "task-owner-a"
        p.session_id = "sess-a"
        other = create_task(goal="unrelated task", session_id="sess-b")
        execute, _ = _recording_execute()
        reply = run_planner_loop_with_plan(plan=p, execute_tool_fn=execute,
                                           ask_llm_fn=_Eval().ask, task=other)
        assert "can't run this plan here" in reply


# ─────────────────────────────────────────────
# #3/#4 continuation routes to executor
# ─────────────────────────────────────────────

class TestContinuationRoutesToExecutor:
    def test_resume_executes_plan_not_prose(self):
        """A control directive must run the plan steps (executor), not fall
        through to LLM prose as a web_search query."""
        import task
        from task import add_plan_to_task

        t = task.create_task(goal="hawaii research", session_id="rt-session")
        _persist_plan("rt-plan", [
            {"step_id": "s1", "goal": "search x", "tool_hint": "web_search", "args": {"query": "x"}},
            {"step_id": "s2", "goal": "search y", "tool_hint": "web_search", "args": {"query": "y"}},
        ])
        add_plan_to_task(t.task_id, "rt-plan")

        # resolve_task binds "run the searches" to the current task...
        r = task.resolve_task("rt-session", "run the searches")
        assert r["action"] == "resume" and r["task"].task_id == t.task_id
        # ...and load_plan_for_task finds the incomplete plan to execute.
        import plans

        found = plans.load_plan_for_task(t.task_id, exclude_status=("completed",))
        assert found and found["plan_id"] == "rt-plan"

        execute, calls = _recording_execute({"web_search": "data"})
        evalx = _Eval()
        reply = run_planner_loop_with_plan(plan_id="rt-plan", execute_tool_fn=execute, ask_llm_fn=evalx.ask, task=t)
        # The steps actually executed (executor path) — not LLM prose.
        assert len(calls) == 2, calls
        assert reply == "COMPLETED SUMMARY"


# ─────────────────────────────────────────────
# #6 bounded retry + synthesis gate
# ─────────────────────────────────────────────

class TestBoundedRetry:
    def test_bounded_retry_two_additional_attempts(self):
        _persist_plan("br-1", [
            {"step_id": "s1", "goal": "gather evidence", "tool_hint": "web_search",
             "args": {"query": "q"}, "status": "failed", "result": "timeout", "evaluation": "failure"},
        ])
        execute, calls = _recording_execute({"web_search": "timeout", "alt1": "timeout", "alt2": "timeout"})
        evalx = _Eval(recovery_tools=("alt1", "alt2", "alt3", "alt4"))
        run_planner_loop_with_plan(plan_id="br-1", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        # 1 resumed attempt + exactly 2 bounded retries.
        assert len(calls) == 3, calls
        assert [t for t, _ in calls] == ["web_search", "alt1", "alt2"]

    def test_permanent_failure_not_retried(self):
        _persist_plan("br-perm", [
            {"step_id": "s1", "goal": "fetch", "tool_hint": "api_fetch",
             "args": {"url": "http://x"}, "status": "failed",
             "result": "Tool error: 404 function is not found", "evaluation": "failure"},
        ])
        execute, calls = _recording_execute({"api_fetch": "Tool error: 404 function is not found"})
        evalx = _Eval(recovery_tools=("alt1", "alt2"))
        run_planner_loop_with_plan(plan_id="br-perm", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        assert len(calls) == 1, calls  # no pointlessly repeated attempts
        assert evalx.recovery_calls == 0

    def test_attempt_budget_survives_restart(self):
        import plans

        _persist_plan("br-restart", [
            {"step_id": "s1", "goal": "g", "tool_hint": "web_search", "args": {"query": "q"},
             "status": "failed", "result": "timeout", "evaluation": "failure"},
        ])
        # Budget already spent across prior runs (1 + 2 retries).
        for i in range(3):
            plans.save_execution_record(f"x{i}", "br-restart", "s1", 0, "web_search" if i == 0 else f"alt{i}",
                                        {"query": "q"}, attempt=i + 1, success=False)
        execute, calls = _recording_execute({"web_search": "timeout"})
        evalx = _Eval(recovery_tools=("alt1", "alt2"))
        run_planner_loop_with_plan(plan_id="br-restart", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        assert calls == [], "step that exhausted its budget must not be re-executed"
        assert plan_from_persistence("br-restart").status == "blocked"

    def test_transient_errors_are_retried(self):
        _persist_plan("br-trans", [
            {"step_id": "s1", "goal": "g", "tool_hint": "web_search", "args": {"query": "q"},
             "status": "failed", "result": "timeout 500", "evaluation": "failure"},
        ])
        execute, calls = _recording_execute({"web_search": "timeout 500", "alt1": "found"})
        evalx = _Eval(recovery_tools=("alt1",))
        run_planner_loop_with_plan(plan_id="br-trans", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        assert len(calls) == 2, calls
        assert plan_from_persistence("br-trans").status == "completed"


class TestSynthesisGate:
    def test_blocked_when_required_step_fails(self):
        _persist_plan("sg-block", [
            {"step_id": "s1", "goal": "research x", "tool_hint": "web_search", "args": {"query": "x"},
             "required": True},
            {"step_id": "s2", "goal": "write the problem_map.md report", "tool_hint": "write_file",
             "args": {"path": "problem_map.md"}, "required": False},
        ])
        execute, calls = _recording_execute({"web_search": "Tool error: 410 gone", "write_file": "wrote"})
        evalx = _Eval(fail_term="410")
        run_planner_loop_with_plan(plan_id="sg-block", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        p = plan_from_persistence("sg-block")
        assert p.status == "blocked"
        assert p.steps[1].status == "blocked"  # artifact step never executed
        assert not any(t == "write_file" for t, _ in calls)
        # The final response names the missing evidence, not a completion.
        assert "NOT generated" in p.final_answer
        assert "research x" in p.final_answer

    def test_completed_when_all_required_succeed(self):
        _persist_plan("sg-ok", [
            {"step_id": "s1", "goal": "research x", "tool_hint": "web_search",
             "args": {"query": "x"}, "required": True},
            {"step_id": "s2", "goal": "write the report file", "tool_hint": "write_file",
             "args": {"path": "r.md"}, "required": False},
        ])
        execute, calls = _recording_execute(
            {"web_search": "found", "write_file": "wrote"})
        evalx = _Eval()
        run_planner_loop_with_plan(plan_id="sg-ok", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        p = plan_from_persistence("sg-ok")
        assert p.status == "completed"
        assert any(t == "write_file" for t, _ in calls)
        assert evalx.summary_calls == 1

    def test_required_step_not_silently_downgraded(self):
        _persist_plan("sg-no-downgrade", [
            {"step_id": "s1", "goal": "required evidence", "tool_hint": "web_search",
             "args": {"query": "q"}, "required": True},
        ])
        execute, _ = _recording_execute({"web_search": "Tool error: 410 gone"})
        evalx = _Eval(recovery_tools=("alt1", "alt2"), fail_term="410")
        run_planner_loop_with_plan(plan_id="sg-no-downgrade", execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        p = plan_from_persistence("sg-no-downgrade")
        # The step is STILL required and failed -> blocked (not completed, and
        # the step was not magically marked optional to satisfy the plan).
        assert p.steps[0].required is True
        assert p.status == "blocked"

    def test_optional_step_failure_does_not_block(self):
        _persist_plan("sg-opt", [
            {"step_id": "s1", "goal": "required evidence", "tool_hint": "web_search",
             "args": {"query": "q"}, "required": True},
            {"step_id": "s2", "goal": "optional extra", "tool_hint": "web_search",
             "args": {"query": "z"}, "required": False},
        ])

        def execute(tool, args):
            # The optional step's query always fails; the required one succeeds.
            if args.get("query") == "z":
                return "Tool error: 410 gone"
            return "found"

        calls = []
        evalx = _Eval()
        orig_ask = evalx.ask

        def ask(prompt):
            calls.append(prompt)
            if "optional extra" in prompt and "step evaluator" in prompt:
                return "FAILURE"
            return orig_ask(prompt)

        run_planner_loop_with_plan(plan_id="sg-opt", execute_tool_fn=execute, ask_llm_fn=ask)
        p = plan_from_persistence("sg-opt")
        assert p.status == "completed"  # optional failure must not block the plan

    def test_retry_failed_only_skips_completed(self):
        _persist_plan("sg-retry", [
            {"step_id": "s1", "goal": "done", "tool_hint": "web_search", "args": {"query": "a"},
             "status": "completed", "result": "ok", "evaluation": "success"},
            {"step_id": "s2", "goal": "failed", "tool_hint": "web_search", "args": {"query": "b"},
             "status": "failed", "result": "boom", "evaluation": "failure"},
        ])
        execute, calls = _recording_execute({"web_search": "found"})
        evalx = _Eval()
        run_planner_loop_with_plan(plan_id="sg-retry", retry_failed_only=True,
                                   execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        assert calls == [("web_search", {"query": "b"})], calls

    def test_replace_tool_preserves_args(self):
        _persist_plan("sg-repl", [
            {"step_id": "s1", "goal": "search", "tool_hint": "read_file", "args": {"path": "/tmp/x"},
             "status": "failed", "result": "boom", "evaluation": "failure"},
        ])
        execute, calls = _recording_execute({"read_file": "boom", "web_search": "found"})
        evalx = _Eval()
        run_planner_loop_with_plan(plan_id="sg-repl", retry_failed_only=True, replace_tool="web_search",
                                   execute_tool_fn=execute, ask_llm_fn=evalx.ask)
        assert calls and calls[0] == ("web_search", {"path": "/tmp/x"}), calls


# ─────────────────────────────────────────────
# helper-level guards
# ─────────────────────────────────────────────

class TestToolNormalization:
    def test_channel_marker_stripped(self):
        assert _normalize_tool_name("read_file<|channel|>commentary", {}) == "read_file"

    def test_alias_mapped_to_canonical(self):
        assert _normalize_tool_name("websearch", {}) == "web_search"
        assert _normalize_tool_name("search_web", {}) == "web_search"

    def test_unknown_tool_preserved(self):
        assert _normalize_tool_name("api_fetch", {}) == "api_fetch"

    def test_permanent_failure_terms(self):
        from agent import _is_permanent_failure

        assert _is_permanent_failure("Tool error: 404 not found")
        assert _is_permanent_failure("410 gone")
        assert _is_permanent_failure("payment required, no credits")
        assert not _is_permanent_failure("timeout 500")
        assert not _is_permanent_failure("rate limited 429")


# ─────────────────────────────────────────────
# #7 confirmation broadening (brain)
# ─────────────────────────────────────────────

class TestConfirmationBroadening:
    def test_yes_words_match_write_the_file(self):
        import brain

        def matches(t):
            return brain._matches_yes(t)

        for phrase in ("write the file", "create the file", "save the file",
                       "apply it", "apply the change", "yes", "go ahead"):
            assert matches(phrase), phrase
        for phrase in ("write a poem", "apply for a job", "save the whales",
                       "create an account", "do it yourself",
                       "go ahead and explain the concept",
                       "write the file to my Desktop", "don't do it"):
            assert not matches(phrase), phrase

    def test_pending_ownership_same_session_allowed(self):
        import brain

        brain._pending_safe["session_id"] = "sess-a"
        assert brain._pending_belongs_to_session("sess-a") is True
        assert brain._pending_belongs_to_session("sess-b") is False
        brain._pending_safe["session_id"] = None
        assert brain._pending_belongs_to_session("sess-any") is True


# ─────────────────────────────────────────────
# brain-level control handler wiring
# ─────────────────────────────────────────────

class TestBrainControlHandler:
    def test_control_directive_resumes_bound_plan(self, monkeypatch, tmp_path):
        """A control directive must route to the plan executor in _process_impl
        — not fall through to LLM prose (Bug #3/#4)."""
        import brain
        from task import add_plan_to_task, create_task

        monkeypatch.setenv("JARVIS_SESSIONS_DIR", str(tmp_path / "sessions"))
        monkeypatch.setenv("JARVIS_TTS_SILENT", "1")

        t = create_task(goal="hawaii research", session_id="brain-sess")
        _persist_plan("bh-plan", [
            {"step_id": "s1", "goal": "search x", "tool_hint": "web_search", "args": {"query": "x"}},
            {"step_id": "s2", "goal": "search y", "tool_hint": "web_search", "args": {"query": "y"}},
        ])
        add_plan_to_task(t.task_id, "bh-plan")

        executed = []

        def fake_execute(tool, args):
            executed.append((tool, args))
            return "found"

        monkeypatch.setattr(brain, "_execute_tool", fake_execute)
        monkeypatch.setattr(brain, "ask_llm_internal", _Eval().ask)

        reply = brain._process_impl("run the searches", "brain-sess")
        # Both plan steps executed via the executor.
        assert len(executed) == 2, executed
        assert reply == "COMPLETED SUMMARY"
        # Plan is now completed in the store.
        p = plan_from_persistence("bh-plan")
        assert p.status == "completed"

    def test_retry_directive_reruns_failed_step(self, monkeypatch, tmp_path):
        import brain
        from task import add_plan_to_task, create_task

        monkeypatch.setenv("JARVIS_SESSIONS_DIR", str(tmp_path / "sessions"))
        monkeypatch.setenv("JARVIS_TTS_SILENT", "1")

        t = create_task(goal="hawaii research", session_id="brain-sess2")
        _persist_plan("bh-retry", [
            {"step_id": "s1", "goal": "search x", "tool_hint": "web_search",
             "args": {"query": "x"}, "status": "failed", "result": "boom", "evaluation": "failure"},
            {"step_id": "s2", "goal": "search y", "tool_hint": "web_search",
             "args": {"query": "y"}, "status": "completed", "result": "ok", "evaluation": "success"},
        ])
        add_plan_to_task(t.task_id, "bh-retry")

        executed = []

        def fake_execute(tool, args):
            executed.append((tool, args))
            return "found"

        monkeypatch.setattr(brain, "_execute_tool", fake_execute)
        monkeypatch.setattr(brain, "ask_llm_internal", _Eval().ask)

        reply = brain._process_impl("retry those", "brain-sess2")
        # Only the FAILED step is re-run on a retry directive.
        assert executed == [("web_search", {"query": "x"})], executed
        assert reply == "COMPLETED SUMMARY"

    def test_control_directive_without_plan_falls_through(self, monkeypatch, tmp_path):
        """A continuation with a resolved task but NO persisted plan must not
        crash — it falls through to normal routing."""
        import brain
        from task import create_task

        monkeypatch.setenv("JARVIS_SESSIONS_DIR", str(tmp_path / "sessions"))
        monkeypatch.setenv("JARVIS_TTS_SILENT", "1")
        create_task(goal="hawaii research", session_id="brain-sess3")

        called = {"n": 0}

        def fake_ask_with_tools(text):
            called["n"] += 1
            return "normal fallback reply"

        monkeypatch.setattr(brain, "ask_with_tools", fake_ask_with_tools)
        # Route directly to the ask_with_tools path (needs_planner/needs_agent_loop off).
        monkeypatch.setattr(brain, "needs_planner", lambda *a, **k: False)
        monkeypatch.setattr(brain, "needs_agent_loop", lambda *a, **k: False)

        reply = brain._process_impl("run the searches", "brain-sess3")
        assert called["n"] == 1
        assert reply == "normal fallback reply"


# ─────────────────────────────────────────────
# EOL model removal
# ─────────────────────────────────────────────

class TestEolModelRemoval:
    EOL = "deepseek-ai/deepseek-v4-pro-0813"

    def test_removed_from_nim_slots(self):
        import brain

        for rota in (brain.NIM_MODEL_FAST, brain.NIM_MODEL_CODING, brain.NIM_MODEL_FRONTIER):
            assert self.EOL not in rota

    def test_removed_from_context_limits(self):
        from config import MODEL_CONTEXT_LIMITS

        assert self.EOL not in MODEL_CONTEXT_LIMITS


# ─────────────────────────────────────────────
# Adversarial-audit regression guards
# ─────────────────────────────────────────────

class TestControlPlaneHijack:
    """Data-plane requests must NEVER auto-resume a plan (resolve_task binds
    + control directive). Each of these is ordinary prose containing a weak
    continuation phrase."""

    DATA_PLANE = [
        "how do it work",
        "can you do it",
        "don't do it",
        "do it yourself",
        "what do it mean",
        "run it in a sandbox",
        "run them through the test suite",
        "resume the video",
        "run the steps in the manual",
        "run the plan through the simulation",
        "do the research for my thesis",
        "write the file to my Desktop",
        "save the file to Desktop",
        "go ahead and explain the concept",
        "finish the job application",
        "next question",
        "retry the download",
        "apply this filter",
    ]

    @pytest.mark.parametrize("msg", DATA_PLANE)
    def test_data_plane_message_never_resumes_plan(self, msg):
        import task

        task.create_task(goal="research hawaii volcano hazard zones", session_id="hp-sess")
        r = task.resolve_task("hp-sess", msg)
        ctrl = [d["type"] for d in task.detect_control_command(msg)]
        hijack = r.get("action") == "resume" and any(x in ("resume_plan", "retry_failed") for x in ctrl)
        assert not hijack, f"{msg!r} hijacked: action={r['action']} ctrl={ctrl}"


class TestValidContinuationRecognition:
    """Natural command variants must still bind to the current task."""

    def test_strong_and_weak_command_variants_resolve(self):
        import task

        t = task.create_task(goal="research hawaii", session_id="vc-sess")
        for msg in ("run the searches", "run those searches", "run these searches",
                    "do the research", "run the plan", "do it", "do it again",
                    "retry the failed searches", "retry the tool calls", "continue",
                    "please run the searches", "don't stop", "yes write the file and retry those"):
            r = task.resolve_task("vc-sess", msg)
            assert r["action"] == "resume", f"{msg!r} -> {r['action']}"
            assert r["task"].task_id == t.task_id


class TestRequiredCoercion:
    def test_string_false_is_optional_not_required(self):
        from agent import _step_from_dict

        s = _step_from_dict({"goal": "g", "tool_hint": "w", "args": {}, "required": "false"})
        assert s.required is False

    def test_null_and_malformed_default_to_required(self):
        from agent import _step_from_dict

        for bad in (None, [], {}, "null", ""):
            s = _step_from_dict({"goal": "g", "tool_hint": "w", "args": {}, "required": bad})
            assert s.required is True, f"required={bad!r} should default to required"

    def test_numeric_required(self):
        from agent import _step_from_dict

        assert _step_from_dict({"goal": "g", "tool_hint": "w", "args": {}, "required": 0}).required is False
        assert _step_from_dict({"goal": "g", "tool_hint": "w", "args": {}, "required": 1}).required is True


class TestAwaitingConfirmationGate:
    """A required step stuck awaiting confirmation must BLOCK the plan and the
    artifact — not mark the plan completed."""

    def test_required_awaiting_confirmation_blocks_plan(self):
        from agent import Plan, PlanStep, run_planner_loop_with_plan

        CONFIRM = "Preview shown above. Say yes to run it for real, or no to cancel."
        calls = []
        plan = Plan(plan_id="gate-await", original_goal="research then write", steps=[
            PlanStep(step_id="s1", goal="required evidence", tool_hint="run_python",
                     args={"code": "x"}, required=True),
            PlanStep(step_id="s2", goal="write the problem_map.md report", tool_hint="write_file",
                     args={"path": "problem_map.md"}, required=False),
        ])

        def execute(tool, args):
            calls.append(tool)
            return CONFIRM if tool == "run_python" else "wrote"

        def ask(prompt):
            if "Summarize what was accomplished" in prompt:
                return "SYNTH"
            return "ok"

        run_planner_loop_with_plan(plan=plan, execute_tool_fn=execute, ask_llm_fn=ask)
        assert plan.status == "blocked", plan.status
        assert plan.steps[1].status == "blocked"
        assert "write_file" not in calls
        assert "waiting on your approval" in plan.final_answer


class TestSynthesisPhrasingGate:
    """Artifact phrasings not matching the old narrow heuristic must still be
    gated when a required step failed."""

    def test_nonmatching_artifact_goal_still_blocked(self):
        from agent import Plan, PlanStep, run_planner_loop_with_plan

        calls = []
        plan = Plan(plan_id="gate-phrase", original_goal="research then create output", steps=[
            PlanStep(step_id="s1", goal="fetch data", tool_hint="api_fetch",
                     args={"url": "http://x"}, required=True),
            PlanStep(step_id="s2", goal="Create output.md with the findings", tool_hint="write_file",
                     args={"path": "output.md"}, required=False),
        ])

        def execute(tool, args):
            calls.append(tool)
            return "Tool error: 404 function is not found"

        def ask(prompt):
            if "step evaluator" in prompt:
                return "FAILURE"
            if "DIFFERENT tool" in prompt:
                return '{"tool": ""}'
            if "Summarize" in prompt:
                return "SYNTH"
            return "ok"

        run_planner_loop_with_plan(plan=plan, execute_tool_fn=execute, ask_llm_fn=ask)
        assert plan.status == "blocked"
        assert "write_file" not in calls


class TestSynthesisGateNoSelfBlock:
    """The artifact gate must not block the artifact because of its OWN pending
    status, and must not block artifact-first plans whose evidence steps are
    still pending (they run after the file is created)."""

    def _run(self, pid, steps, fail_tool=None):
        from agent import Plan, run_planner_loop_with_plan

        plan = Plan(plan_id=pid, original_goal="g", steps=steps)
        calls = []

        def execute(tool, args):
            calls.append(tool)
            if fail_tool and tool == fail_tool:
                return "Tool error: 404 function is not found"
            return "found"

        def ask(prompt):
            if "step evaluator" in prompt:
                return "FAILURE" if "Tool error" in prompt else "SUCCESS"
            if "DIFFERENT tool" in prompt:
                return '{"tool": ""}'
            if "Summarize" in prompt:
                return "SYNTH"
            return "ok"

        run_planner_loop_with_plan(plan=plan, execute_tool_fn=execute, ask_llm_fn=ask)
        return plan, calls

    def test_required_artifact_step_completes_when_evidence_ok(self):
        from agent import PlanStep

        plan, calls = self._run("gate-self", [
            PlanStep(step_id="s1", goal="search x", tool_hint="web_search",
                     args={"query": "x"}, required=True),
            PlanStep(step_id="s2", goal="write the report file", tool_hint="write_file",
                     args={"path": "r.md"}, required=True),
        ])
        assert plan.status == "completed", plan.status
        assert "write_file" in calls

    def test_artifact_first_plan_completes(self):
        from agent import PlanStep

        plan, calls = self._run("gate-first", [
            PlanStep(step_id="s1", goal="create the output markdown file", tool_hint="write_file",
                     args={"path": "out.md"}, required=False),
            PlanStep(step_id="s2", goal="search for evidence", tool_hint="web_search",
                     args={"query": "x"}, required=True),
        ])
        assert plan.status == "completed", plan.status
        assert calls == ["write_file", "web_search"], calls

    def test_required_artifact_still_blocked_when_evidence_fails(self):
        from agent import PlanStep

        plan, calls = self._run("gate-self-fail", [
            PlanStep(step_id="s1", goal="fetch", tool_hint="api_fetch",
                     args={"url": "http://x"}, required=True),
            PlanStep(step_id="s2", goal="write the report file", tool_hint="write_file",
                     args={"path": "r.md"}, required=True),
        ], fail_tool="api_fetch")
        assert plan.status == "blocked", plan.status
        assert "write_file" not in calls


class TestConfirmationNoFalsePositive:
    """Ordinary conversational phrases must not apply a staged pending op."""

    def test_data_plane_phrases_do_not_confirm(self):
        import brain

        for msg in ("do it yourself", "go ahead and explain the concept",
                    "write the file to my Desktop", "don't do it", "how do it work"):
            assert not brain._matches_yes(msg), f"{msg!r} should not confirm"

    def test_command_phrases_confirm(self):
        import brain

        for msg in ("yes", "write the file", "do it", "go ahead", "apply it",
                    "save the file", "approved"):
            assert brain._matches_yes(msg), f"{msg!r} should confirm"
