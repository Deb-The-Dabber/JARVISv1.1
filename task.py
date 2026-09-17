"""Active Task System (Phase 2) - Persistent task state for autonomous agent loops.
Phase 5: Goal criteria and verification support.
"""

import datetime
import hashlib
import json
import os
import re
import subprocess
import threading
from typing import Optional, Dict, List, Any


class TaskPhase:
    EXPLORE = "explore"
    PLAN = "plan"
    IMPLEMENT = "implement"
    TEST = "test"
    VERIFY = "verify"
    COMPLETE = "complete"


class TaskEventType:
    """Phase 5F: Structured event types for Thought/Action/Reality separation."""
    MODEL_THOUGHT = "model_thought"      # What the model says it will do
    TOOL_ACTION = "tool_action"          # What JARVIS actually executes
    VERIFIED_REALITY = "verified_reality" # What actually happened (verified)
    STATE_CHANGE = "state_change"        # Task/phase/status changes
    PROGRESS_UPDATE = "progress_update"  # Progress metric updates
    VERIFICATION_RESULT = "verification_result" # Verification outcomes


class TaskStatus:
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    # Phase 5E: Recovery states
    BLOCKED = "blocked"
    DIAGNOSING = "diagnosing"
    RECOVERING = "recovering"


# V5 rewrite: the task system is a session-scoped registry, not a global
# singleton. Each task persists as its own file under ~/.jarvis/tasks/; a
# per-session "current" pointer records which task the conversation is on.
# The legacy ~/.jarvis/active_task.json is migrated ONCE into an archive and
# is never auto-resumed (it was the source of stale-task hijack).
# All paths are env-overridable so tests (including subprocess servers that
# inherit os.environ) can never touch the real store.
_LEGACY_ACTIVE_TASK_FILE = os.getenv(
    "JARVIS_LEGACY_ACTIVE_TASK_FILE", os.path.expanduser("~/.jarvis/active_task.json")
)
_TASKS_DIR = os.getenv("JARVIS_TASKS_DIR", os.path.expanduser("~/.jarvis/tasks"))
_CURRENT_FILE = os.path.join(_TASKS_DIR, "current.json")
_LEGACY_ARCHIVE_DIR = os.path.join(_TASKS_DIR, "legacy")
_active_task_lock = threading.RLock()
_migration_done = False


def _task_file(task_id: str) -> str:
    return os.path.join(_TASKS_DIR, f"task_{task_id}.json")


def _load_current_map() -> Dict[str, str]:
    try:
        if os.path.exists(_CURRENT_FILE):
            with open(_CURRENT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_current_map(data: Dict[str, str]) -> None:
    try:
        os.makedirs(_TASKS_DIR, exist_ok=True)
        with open(_CURRENT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _migrate_legacy() -> None:
    """Archive the legacy global active_task.json once.

    The legacy file represented a single global 'active task' that was not
    session-scoped and was resumed on any continuation-ish message (the root
    cause of stale-task hijack). We preserve it by moving it into the archive
    dir; it is never loaded as the current task.
    """
    global _migration_done
    with _active_task_lock:
        if _migration_done:
            return
        _migration_done = True
        try:
            if not os.path.exists(_LEGACY_ACTIVE_TASK_FILE):
                return
            os.makedirs(_LEGACY_ARCHIVE_DIR, exist_ok=True)
            dest = os.path.join(_LEGACY_ARCHIVE_DIR, "active_task.json")
            if not os.path.exists(dest):
                os.replace(_LEGACY_ACTIVE_TASK_FILE, dest)
        except Exception:
            pass


# ─────────────────────────────────────────────
# PHASE 5: GOAL CRITERIA & VERIFICATION
# ─────────────────────────────────────────────

class CriterionType:
    TESTS_PASS = "tests_pass"
    SYNTAX_VALID = "syntax_valid"
    FILE_EXISTS = "file_exists"
    FILE_CONTAINS = "file_contains"
    FILE_NOT_CONTAINS = "file_not_contains"
    COMMAND_SUCCEEDS = "command_succeeds"
    CUSTOM = "custom"
    # Phase 7: Behavioral/Metric criteria
    METRIC_THRESHOLD = "metric_threshold"
    METRIC_RATIO = "metric_ratio"
    METRIC_DELTA = "metric_delta"
    OUTPUT_MATCHES = "output_matches"
    OUTPUT_NOT_MATCHES = "output_not_matches"
    BEHAVIOR_MATCHES = "behavior_matches"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    REGRESSION_FREE = "regression_free"
    ALL = "all"
    ANY = "any"


def verify_criterion(criterion: dict, project_root: str = "") -> tuple[bool, str]:
    """
    Verify a single goal criterion.
    Returns (passed: bool, message: str).
    """
    ctype = criterion.get("type", "")
    target = criterion.get("target", "")
    required = criterion.get("required", True)
    
    if not target:
        return False, f"Criterion missing target"
    
    # Handle each criterion type with its own error handling
    if ctype == CriterionType.TESTS_PASS:
        try:
            cmd = ["python", "-m", "pytest", target, "-v", "--tb=short"]
            result = subprocess.run(
                cmd, 
                cwd=project_root or os.getcwd(),
                capture_output=True, 
                text=True,
                timeout=120
            )
            passed = result.returncode == 0
            output = result.stdout[-2000:] if result.stdout else ""
            if result.stderr:
                output += "\n" + result.stderr[-1000:]
            return passed, f"pytest {'passed' if passed else 'failed'}: {output}"
        except subprocess.TimeoutExpired:
            return False, "pytest timed out"
        except Exception as e:
            return False, f"pytest error: {e}"
    
    elif ctype == CriterionType.SYNTAX_VALID:
        if not os.path.exists(target):
            return False, f"File not found: {target}"
        if not target.endswith(".py"):
            return False, f"Syntax check only for .py files: {target}"
        try:
            with open(target, "r") as f:
                compile(f.read(), target, "exec")
            return True, f"Syntax valid: {target}"
        except SyntaxError as e:
            return False, f"Syntax error in {target}: {e}"
        except Exception as e:
            return False, f"Syntax check error: {e}"
    
    elif ctype == CriterionType.FILE_EXISTS:
        path = os.path.join(project_root or os.getcwd(), target)
        exists = os.path.exists(path)
        return exists, f"File {'exists' if exists else 'not found'}: {target}"
    
    elif ctype == CriterionType.FILE_CONTAINS:
        path = os.path.join(project_root or os.getcwd(), target)
        if not os.path.exists(path):
            return False, f"File not found: {target}"
        substring = criterion.get("substring", "")
        if not substring:
            return False, f"FILE_CONTAINS criterion missing 'substring' field"
        try:
            with open(path, "r") as f:
                content = f.read()
            contains = substring in content
            return contains, f"File {'contains' if contains else 'does not contain'} '{substring}': {target}"
        except Exception as e:
            return False, f"File read error: {e}"
    
    elif ctype == CriterionType.FILE_NOT_CONTAINS:
        path = os.path.join(project_root or os.getcwd(), target)
        if not os.path.exists(path):
            return False, f"File not found: {target}"
        substring = criterion.get("substring", "")
        if not substring:
            return False, f"FILE_NOT_CONTAINS criterion missing 'substring' field"
        try:
            with open(path, "r") as f:
                content = f.read()
            not_contains = substring not in content
            return not_contains, f"File {'does not contain' if not_contains else 'contains'} '{substring}': {target}"
        except Exception as e:
            return False, f"File read error: {e}"
    
    elif ctype == CriterionType.COMMAND_SUCCEEDS:
        try:
            cmd = criterion.get("command", target)
            result = subprocess.run(
                cmd, 
                shell=True,
                cwd=project_root or os.getcwd(),
                capture_output=True, 
                text=True,
                timeout=120
            )
            passed = result.returncode == 0
            output = result.stdout[-1000:] if result.stdout else ""
            if result.stderr:
                output += "\n" + result.stderr[-500:]
            return passed, f"Command {'succeeded' if passed else 'failed'}: {output}"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Command error: {e}"
    
    elif ctype == CriterionType.CUSTOM:
        verifier_name = criterion.get("verifier", "")
        if not verifier_name:
            return False, "CUSTOM criterion missing 'verifier' function name"
        return False, f"Custom verifier '{verifier_name}' not implemented"
    
    # Phase 7: Behavioral/Metric criteria
    elif ctype == CriterionType.METRIC_THRESHOLD:
        # metric_threshold: compare a measured metric against a threshold
        metric_name = criterion.get("metric", target)
        operator = criterion.get("operator", ">=")
        threshold = criterion.get("value", 0)
        source = criterion.get("source", "file")
        
        try:
            if source == "file":
                metric_file = criterion.get("metric_file", target)
                # Resolve relative to project_root
                if not os.path.isabs(metric_file):
                    metric_file = os.path.join(project_root or os.getcwd(), metric_file)
                if not os.path.exists(metric_file):
                    return False, f"Metric file not found: {metric_file}"
                with open(metric_file, "r") as f:
                    data = json.load(f)
                value = data.get(metric_name)
                if value is None:
                    return False, f"Metric '{metric_name}' not found in {metric_file}"
            elif source == "command":
                cmd = criterion.get("command", target)
                result = subprocess.run(
                    cmd, shell=True, cwd=project_root or os.getcwd(),
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode != 0:
                    return False, f"Command failed: {result.stderr[:200]}"
                try:
                    data = json.loads(result.stdout)
                    value = data.get(metric_name)
                except json.JSONDecodeError:
                    value = float(result.stdout.strip())
            elif source == "module":
                module_name = criterion.get("module", target)
                func_name = criterion.get("function", metric_name)
                module = __import__(module_name, fromlist=[func_name])
                func = getattr(module, func_name)
                value = func()
            else:
                return False, f"Unknown metric source: {source}"
            
            if value is None:
                return False, f"Metric '{metric_name}' not found"
            
            ops = {
                ">=": lambda a, b: a >= b,
                "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                "<": lambda a, b: a < b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
            }
            op_func = ops.get(operator)
            if not op_func:
                return False, f"Unknown operator: {operator}"
            
            passed = op_func(value, threshold)
            return passed, f"Metric '{metric_name}' = {value} {operator} {threshold}: {'PASS' if passed else 'FAIL'}"
        except Exception as e:
            return False, f"Metric threshold verification error: {e}"
    
    elif ctype == CriterionType.METRIC_RATIO:
        metric1_name = criterion.get("metric1", target)
        metric2_name = criterion.get("metric2", "")
        operator = criterion.get("operator", ">=")
        threshold = criterion.get("value", 0)
        source = criterion.get("source", "file")
        
        try:
            if source == "file":
                metric_file1 = criterion.get("metric_file1", target)
                if not os.path.isabs(metric_file1):
                    metric_file1 = os.path.join(project_root or os.getcwd(), metric_file1)
                if not os.path.exists(metric_file1):
                    return False, f"Metric file 1 not found: {metric_file1}"
                with open(metric_file1, "r") as f:
                    data1 = json.load(f)
                value1 = data1.get(metric1_name)
                if value1 is None:
                    return False, f"Metric '{metric1_name}' not found in {metric_file1}"
                
                metric_file2 = criterion.get("metric_file2", "")
                if not metric_file2:
                    return False, "metric_file2 required for metric_ratio"
                if not os.path.isabs(metric_file2):
                    metric_file2 = os.path.join(project_root or os.getcwd(), metric_file2)
                if not os.path.exists(metric_file2):
                    return False, f"Metric file 2 not found: {metric_file2}"
                with open(metric_file2, "r") as f:
                    data2 = json.load(f)
                value2 = data2.get(metric2_name)
                if value2 is None:
                    return False, f"Metric '{metric2_name}' not found in {metric_file2}"
            else:
                return False, f"Only 'file' source supported for metric_ratio currently"
            
            if value2 == 0:
                return False, f"Metric ratio denominator is zero"
            
            ratio = value1 / value2
            
            ops = {
                ">=": lambda a, b: a >= b,
                "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                "<": lambda a, b: a < b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
            }
            op_func = ops.get(operator)
            if not op_func:
                return False, f"Unknown operator: {operator}"
            
            passed = op_func(ratio, threshold)
            return passed, f"Metric ratio '{metric1_name}' / '{metric2_name}' = {ratio:.4f} {operator} {threshold}: {'PASS' if passed else 'FAIL'}"
        except Exception as e:
            return False, f"Metric ratio verification error: {e}"
    
    elif ctype == CriterionType.METRIC_DELTA:
        metric_name = criterion.get("metric", target)
        baseline_value = criterion.get("baseline", 0)
        operator = criterion.get("operator", ">=")
        threshold = criterion.get("value", 0)
        mode = criterion.get("mode", "relative")
        source = criterion.get("source", "file")
        
        try:
            if source == "file":
                metric_file = criterion.get("metric_file", target)
                if not os.path.isabs(metric_file):
                    metric_file = os.path.join(project_root or os.getcwd(), metric_file)
                if not os.path.exists(metric_file):
                    return False, f"Metric file not found: {metric_file}"
                with open(metric_file, "r") as f:
                    data = json.load(f)
                current_value = data.get(metric_name)
                if current_value is None:
                    return False, f"Metric '{metric_name}' not found in {metric_file}"
            elif source == "command":
                cmd = criterion.get("command", target)
                result = subprocess.run(
                    cmd, shell=True, cwd=project_root or os.getcwd(),
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode != 0:
                    return False, f"Command failed: {result.stderr[:200]}"
                try:
                    data = json.loads(result.stdout)
                    current_value = data.get(metric_name)
                except json.JSONDecodeError:
                    current_value = float(result.stdout.strip())
            elif source == "module":
                module_name = criterion.get("module", target)
                func_name = criterion.get("function", metric_name)
                module = __import__(module_name, fromlist=[func_name])
                func = getattr(module, func_name)
                current_value = func()
            else:
                return False, f"Unknown metric source: {source}"
            
            if current_value is None:
                return False, f"Metric '{metric_name}' not found"
            
            if mode == "relative":
                if baseline_value == 0:
                    delta = float('inf') if current_value > 0 else 0
                else:
                    delta = (current_value - baseline_value) / abs(baseline_value)
            else:
                delta = current_value - baseline_value
            
            ops = {
                ">=": lambda a, b: a >= b,
                "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                "<": lambda a, b: a < b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
            }
            op_func = ops.get(operator)
            if not op_func:
                return False, f"Unknown operator: {operator}"
            
            passed = op_func(delta, threshold)
            return passed, f"Metric delta for '{metric_name}': {delta:.4f} {operator} {threshold}: {'PASS' if passed else 'FAIL'}"
        except Exception as e:
            return False, f"Metric delta verification error: {e}"
    
    elif ctype == CriterionType.OUTPUT_MATCHES:
        command = criterion.get("command", target)
        pattern = criterion.get("pattern", "")
        matcher = criterion.get("matcher", "substring")
        
        if not command:
            return False, "OUTPUT_MATCHES criterion missing 'command'"
        if not pattern and matcher != "any":
            return False, "OUTPUT_MATCHES criterion missing 'pattern'"
        
        try:
            result = subprocess.run(
                command, shell=True, cwd=project_root or os.getcwd(),
                capture_output=True, text=True, timeout=120
            )
            output = result.stdout + result.stderr
            
            if matcher == "exact":
                matched = output.strip() == pattern
            elif matcher == "substring":
                matched = pattern in output
            elif matcher == "regex":
                matched = bool(re.search(pattern, output))
            elif matcher == "contains":
                matched = any(p in output for p in pattern) if isinstance(pattern, list) else pattern in output
            elif matcher == "any":
                matched = len(output.strip()) > 0
            else:
                return False, f"Unknown matcher: {matcher}"
            
            return matched, f"Output {'matches' if matched else 'does not match'} pattern '{pattern}' ({matcher})"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Output match error: {e}"
    
    elif ctype == CriterionType.OUTPUT_NOT_MATCHES:
        command = criterion.get("command", target)
        pattern = criterion.get("pattern", "")
        matcher = criterion.get("matcher", "substring")
        
        if not command:
            return False, "OUTPUT_NOT_MATCHES criterion missing 'command'"
        if not pattern and matcher != "any":
            return False, "OUTPUT_NOT_MATCHES criterion missing 'pattern'"
        
        try:
            result = subprocess.run(
                command, shell=True, cwd=project_root or os.getcwd(),
                capture_output=True, text=True, timeout=120
            )
            output = result.stdout + result.stderr
            
            if matcher == "exact":
                matched = output.strip() == pattern
            elif matcher == "substring":
                matched = pattern in output
            elif matcher == "regex":
                matched = bool(re.search(pattern, output))
            elif matcher == "contains":
                matched = any(p in output for p in pattern) if isinstance(pattern, list) else pattern in output
            else:
                return False, f"Unknown matcher: {matcher}"
            
            return not matched, f"Output {'does not match' if not matched else 'matches'} pattern '{pattern}' ({matcher})"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, f"Output match error: {e}"
    
    elif ctype == CriterionType.REGRESSION_FREE:
        command = criterion.get("command", target) or "python -m pytest -v --tb=short"
        baseline_file = criterion.get("baseline", "")
        allow_new_tests = criterion.get("allow_new_tests", True)
        
        try:
            cmd = command
            result = subprocess.run(
                cmd, shell=True, cwd=project_root or os.getcwd(),
                capture_output=True, text=True, timeout=300
            )
            
            passed = result.returncode == 0
            output = result.stdout[-3000:] if result.stdout else ""
            if result.stderr:
                output += "\n" + result.stderr[-2000:]
            
            if baseline_file and os.path.exists(baseline_file):
                with open(baseline_file, "r") as f:
                    baseline = json.load(f)
                
                current_passed = re.search(r"(\d+)\s+passed", output)
                current_failed = re.search(r"(\d+)\s+failed", output)
                current_errors = re.search(r"(\d+)\s+error", output)
                
                curr_passed = int(current_passed.group(1)) if current_passed else 0
                curr_failed = int(current_failed.group(1)) if current_failed else 0
                curr_errors = int(current_errors.group(1)) if current_errors else 0
                
                base_passed = baseline.get("passed", 0)
                base_failed = baseline.get("failed", 0)
                base_errors = baseline.get("errors", 0)
                
                regression = (curr_failed > base_failed) or (curr_errors > base_errors)
                
                if not allow_new_tests and curr_passed > base_passed:
                    return False, f"New tests added (not allowed): {curr_passed} vs {base_passed}"
                
                if regression:
                    return False, f"Regression detected: passed {curr_passed}/{base_passed}, failed {curr_failed}/{base_failed}, errors {curr_errors}/{base_errors}"
                
                return True, f"No regression: passed {curr_passed} (was {base_passed}), failed {curr_failed} (was {base_failed})"
            
            return passed, f"Regression check {'passed' if passed else 'failed'}: {output[:500]}"
        except subprocess.TimeoutExpired:
            return False, "Regression test timed out"
        except Exception as e:
            return False, f"Regression check error: {e}"
    
    elif ctype == CriterionType.PERFORMANCE_THRESHOLD:
        command = criterion.get("command", target)
        metric = criterion.get("metric", "time")
        operator = criterion.get("operator", "<=")
        threshold = criterion.get("value", 0)
        unit = criterion.get("unit", "seconds")
        
        if not command:
            return False, "PERFORMANCE_THRESHOLD criterion missing 'command'"
        
        try:
            import time
            import resource
            
            if metric == "time":
                start = time.perf_counter()
                result = subprocess.run(
                    command, shell=True, cwd=project_root or os.getcwd(),
                    capture_output=True, text=True, timeout=120
                )
                elapsed = time.perf_counter() - start
                value = elapsed
                unit_str = "seconds"
            elif metric == "memory":
                result = subprocess.run(
                    command, shell=True, cwd=project_root or os.getcwd(),
                    capture_output=True, text=True, timeout=120
                )
                usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                value = usage.ru_maxrss / 1024.0
                unit_str = "MB"
            else:
                return False, f"Unknown performance metric: {metric}"
            
            ops = {
                ">=": lambda a, b: a >= b,
                "<=": lambda a, b: a <= b,
                ">": lambda a, b: a > b,
                "<": lambda a, b: a < b,
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
            }
            op_func = ops.get(operator)
            if not op_func:
                return False, f"Unknown operator: {operator}"
            
            passed = op_func(value, threshold)
            return passed, f"Performance '{metric}' = {value:.3f} {unit_str} {operator} {threshold}: {'PASS' if passed else 'FAIL'}"
        except subprocess.TimeoutExpired:
            return False, "Performance test timed out"
        except Exception as e:
            return False, f"Performance check error: {e}"
    
    elif ctype == CriterionType.ALL:
        sub_criteria = criterion.get("criteria", [])
        if not sub_criteria:
            return False, "ALL criterion missing 'criteria' list"
        
        results = []
        all_passed = True
        for sub in sub_criteria:
            passed, message = verify_criterion(sub, project_root)
            results.append({"type": sub.get("type"), "passed": passed, "message": message})
            if not passed:
                all_passed = False
        
        message = f"ALL: {'all passed' if all_passed else 'some failed'}"
        return all_passed, message
    
    elif ctype == CriterionType.ANY:
        sub_criteria = criterion.get("criteria", [])
        if not sub_criteria:
            return False, "ANY criterion missing 'criteria' list"
        
        results = []
        any_passed = False
        for sub in sub_criteria:
            passed, message = verify_criterion(sub, project_root)
            results.append({"type": sub.get("type"), "passed": passed, "message": message})
            if passed:
                any_passed = True
        
        message = f"ANY: {'at least one passed' if any_passed else 'all failed'}"
        return any_passed, message
    
    else:
        return False, f"Unknown criterion type: {ctype}"


def verify_all_criteria(criteria: List[dict], project_root: str = "") -> tuple[bool, List[dict]]:
    """
    Verify all goal criteria.
    Returns (all_passed: bool, results: List[dict]).
    """
    results = []
    all_passed = True
    
    for criterion in criteria:
        passed, message = verify_criterion(criterion, project_root)
        result = {
            "type": criterion.get("type", "unknown"),
            "target": criterion.get("target", ""),
            "required": criterion.get("required", True),
            "passed": passed,
            "message": message,
        }
        results.append(result)
        if result.get("required", True) and not passed:
            all_passed = False
    
    return all_passed, results


class ActiveTask:
    """Persistent task state that survives across conversation turns."""
    
    def __init__(
        self,
        task_id: str,
        goal: str,
        project_root: str = "",
        phase: str = TaskPhase.EXPLORE,
        status: str = TaskStatus.ACTIVE,
        completed_steps: List[Dict] = None,
        next_action: str = "",
        progress_metrics: Dict = None,
        created_at: str = None,
        updated_at: str = None,
        # Phase 5: Goal criteria
        goal_criteria: List[Dict] = None,
        # V5 rewrite: session-scoped identity + temporal + plan ownership
        session_id: str = "default",
        temporal_context: Dict = None,
        plan_ids: List[str] = None,
    ):
        self.task_id = task_id
        self.goal = goal
        self.project_root = project_root
        self.phase = phase
        self.status = status
        self.session_id = session_id
        self.temporal_context = temporal_context or {}
        self.plan_ids = plan_ids or []
        self.completed_steps = completed_steps or []
        self.next_action = next_action
        self.progress_metrics = progress_metrics or {
            "files_inspected": [],
            "files_modified": [],
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tool_calls": 0,
            "last_progress_step": 0,
            "stagnation_counter": 0,
            # Phase 5: verified progress tracking
            "verified_progress_count": 0,
            "false_progress_count": 0,
        }
        # Phase 7A: Execution budget tracking
        self.execution_budget = {
            "time_budget_seconds": 600,      # 10 minutes default
            "time_spent_seconds": 0,
            "llm_budget": 20,                # 20 LLM calls default
            "llm_calls": 0,
            "tool_budget": 50,               # 50 tool calls default
            "tool_calls": 0,
            "replan_budget": 3,              # 3 replans default
            "replans": 0,
            "recovery_budget": 2,            # 2 recoveries default
            "recoveries": 0,
        }
        now = datetime.datetime.now().isoformat()
        self.created_at = created_at or now
        self.updated_at = updated_at or now
        # Cross-turn tool call memory (tool_name:arg_key -> step_index)
        self.tool_history: Dict[str, int] = {}
        # Phase 5: Goal criteria
        self.goal_criteria = goal_criteria or []
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "project_root": self.project_root,
            "phase": self.phase,
            "status": self.status,
            "session_id": self.session_id,
            "temporal_context": self.temporal_context,
            "plan_ids": self.plan_ids,
            "completed_steps": self.completed_steps,
            "next_action": self.next_action,
            "progress_metrics": self.progress_metrics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_history": self.tool_history,
            "goal_criteria": self.goal_criteria,
            "execution_budget": self.execution_budget,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ActiveTask":
        task = cls(
            task_id=data.get("task_id", ""),
            goal=data.get("goal", ""),
            project_root=data.get("project_root", ""),
            phase=data.get("phase", TaskPhase.EXPLORE),
            status=data.get("status", TaskStatus.ACTIVE),
            completed_steps=data.get("completed_steps", []),
            next_action=data.get("next_action", ""),
            progress_metrics=data.get("progress_metrics", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            goal_criteria=data.get("goal_criteria", []),
            session_id=data.get("session_id", "default"),
            temporal_context=data.get("temporal_context", {}),
            plan_ids=data.get("plan_ids", []),
        )
        task.tool_history = data.get("tool_history", {})
        task.execution_budget = data.get("execution_budget", {})
        return task
    
    def save(self) -> None:
        """Persist this task to its per-task file."""
        with _active_task_lock:
            try:
                os.makedirs(_TASKS_DIR, exist_ok=True)
                self.updated_at = datetime.datetime.now().isoformat()
                with open(_task_file(self.task_id), "w", encoding="utf-8") as f:
                    json.dump(self.to_dict(), f, indent=2)
            except Exception as e:
                print(f"[DEBUG] Failed to save task: {e}")
    
    @classmethod
    def load(cls, task_id: str = "") -> Optional["ActiveTask"]:
        """Load a task by id from the registry."""
        with _active_task_lock:
            try:
                if not task_id:
                    return None
                path = _task_file(task_id)
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        return cls.from_dict(data)
            except Exception as e:
                print(f"[DEBUG] Failed to load task: {e}")
        return None
    
    @classmethod
    def clear(cls) -> None:
        """Deprecated global-singleton clear; kept for API compatibility.

        The V5 registry does not have a single 'active task' to clear.
        """
        pass
    
    def record_step(self, step_data: dict, tool_name: str = "", args: dict = None, verified: bool = False) -> None:
        """Record a completed step and update progress metrics.
        
        Args:
            step_data: The step data dict
            tool_name: Name of the tool that was executed
            args: Arguments passed to the tool
            verified: Whether the tool execution was verified as making actual progress
        """
        step_data["timestamp"] = datetime.datetime.now().isoformat()
        step_data["phase"] = self.phase
        # Add step_id if not present (for stable identity)
        if "step_id" not in step_data:
            step_data["step_id"] = hashlib.sha256(
                f"{step_data.get('tool','')}|{step_data.get('thought','')}|{json.dumps(step_data.get('args',{}), sort_keys=True)}".encode()
            ).hexdigest()[:12]
        # Track verified vs tool success
        step_data["verified"] = verified
        self.completed_steps.append(step_data)
        self.updated_at = datetime.datetime.now().isoformat()
        
        # Update progress metrics
        self.progress_metrics["tool_calls"] += 1
        if verified:
            self.progress_metrics["verified_progress_count"] += 1
            self.progress_metrics["last_progress_step"] = len(self.completed_steps)
            self.progress_metrics["stagnation_counter"] = 0
        else:
            # Increment false progress counter for tool success without verification
            if step_data.get("success"):
                self.progress_metrics["false_progress_count"] = self.progress_metrics.get("false_progress_count", 0) + 1
        
        # Track files inspected/modified
        if tool_name == "read_file" and args:
            path = args.get("path", "")
            if path and path not in self.progress_metrics["files_inspected"]:
                self.progress_metrics["files_inspected"].append(path)
        elif tool_name in ("write_file", "create_file", "append_file") and args:
            path = args.get("path", "")
            if path and path not in self.progress_metrics["files_modified"]:
                self.progress_metrics["files_modified"].append(path)
        elif tool_name in ("run_python", "run_terminal_command") and args:
            # Running tests/commands counts as progress
            self.progress_metrics["last_progress_step"] = len(self.completed_steps)
            self.progress_metrics["stagnation_counter"] = 0
        
        # Track tool history for duplicate detection
        if tool_name and args:
            arg_key = str(sorted((k, str(v)) for k, v in (args or {}).items()))
            history_key = f"{tool_name}:{arg_key}"
            self.tool_history[history_key] = len(self.completed_steps) - 1
        
        self.save()
    
    def record_verified_progress(self, step_index: int) -> None:
        """Mark a previously recorded step as verified progress."""
        if 0 <= step_index < len(self.completed_steps):
            step = self.completed_steps[step_index]
            if not step.get("verified", False):
                step["verified"] = True
                self.progress_metrics["verified_progress_count"] += 1
                self.progress_metrics["last_progress_step"] = len(self.completed_steps)
                self.progress_metrics["stagnation_counter"] = 0
                # Decrement false progress if it was previously counted
                if step.get("success"):
                    self.progress_metrics["false_progress_count"] = max(0, self.progress_metrics.get("false_progress_count", 0) - 1)
                self.updated_at = datetime.datetime.now().isoformat()
                self.save()
    
    def get_progress_summary(self) -> dict:
        """Get a summary of progress metrics including verified vs false progress."""
        return {
            "tool_calls": self.progress_metrics.get("tool_calls", 0),
            "verified_progress": self.progress_metrics.get("verified_progress_count", 0),
            "false_progress": self.progress_metrics.get("false_progress_count", 0),
            "files_inspected": len(self.progress_metrics.get("files_inspected", [])),
            "files_modified": len(self.progress_metrics.get("files_modified", [])),
            "tests_run": self.progress_metrics.get("tests_run", 0),
            "tests_passed": self.progress_metrics.get("tests_passed", 0),
            "tests_failed": self.progress_metrics.get("tests_failed", 0),
            "stagnation_counter": self.progress_metrics.get("stagnation_counter", 0),
            "last_progress_step": self.progress_metrics.get("last_progress_step", 0),
        }
    
    # Phase 5E: Recovery State Machine
    
    def enter_blocked(self, reason: str) -> None:
        """Enter BLOCKED state - task cannot proceed due to external dependency or repeated failure."""
        self.status = TaskStatus.BLOCKED
        self.progress_metrics["blocked_reason"] = reason
        self.progress_metrics["blocked_at"] = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now().isoformat()
        self.save()
    
    def enter_diagnosing(self, diagnosis: str = "") -> None:
        """Enter DIAGNOSING state - actively investigating root cause."""
        self.status = TaskStatus.DIAGNOSING
        self.progress_metrics["diagnosis"] = diagnosis
        self.progress_metrics["diagnosing_at"] = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now().isoformat()
        self.save()
    
    def enter_recovering(self, recovery_plan: str = "") -> None:
        """Enter RECOVERING state - implementing fix for identified issue."""
        self.status = TaskStatus.RECOVERING
        self.progress_metrics["recovery_plan"] = recovery_plan
        self.progress_metrics["recovering_at"] = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now().isoformat()
        self.save()
    
    def resume_active(self) -> None:
        """Resume ACTIVE state from recovery."""
        self.status = TaskStatus.ACTIVE
        self.updated_at = datetime.datetime.now().isoformat()
        self.save()
    
    def check_recovery_transitions(self, verify_fn=None) -> str:
        """
        Check if state transitions are needed based on current metrics.
        Returns the new status if changed, otherwise current status.
        """
        # Check for stagnation -> BLOCKED
        if self.status == TaskStatus.ACTIVE and self.check_stagnation():
            self.enter_blocked("Stagnation detected: no verified progress in 3+ steps")
            return TaskStatus.BLOCKED
        
        # BLOCKED -> DIAGNOSING after some time or manual trigger
        if self.status == TaskStatus.BLOCKED:
            blocked_at = self.progress_metrics.get("blocked_at")
            if blocked_at:
                try:
                    blocked_time = datetime.datetime.fromisoformat(blocked_at)
                    if (datetime.datetime.now() - blocked_time).total_seconds() > 30:
                        self.enter_diagnosing("Auto-transition: blocked for 30s, starting diagnosis")
                        return TaskStatus.DIAGNOSING
                except Exception:
                    pass
        
        # DIAGNOSING -> RECOVERING when diagnosis complete
        if self.status == TaskStatus.DIAGNOSING:
            diagnosis = self.progress_metrics.get("diagnosis", "")
            if diagnosis and "root cause" in diagnosis.lower():
                self.enter_recovering("Applying fix based on diagnosis")
                return TaskStatus.RECOVERING
        
        # RECOVERING -> ACTIVE after verification passes
        if self.status == TaskStatus.RECOVERING:
            if self.goal_criteria:
                passed, _ = self.verify_goal()
                if passed:
                    self.resume_active()
                    return TaskStatus.ACTIVE
            elif self.progress_metrics.get("verified_progress_count", 0) > self.progress_metrics.get("last_progress_step", 0):
                self.resume_active()
                return TaskStatus.ACTIVE
        
        return self.status
    
    def get_recovery_status(self) -> dict:
        """Get current recovery state info."""
        return {
            "status": self.status,
            "blocked_reason": self.progress_metrics.get("blocked_reason"),
            "diagnosis": self.progress_metrics.get("diagnosis"),
            "recovery_plan": self.progress_metrics.get("recovery_plan"),
            "blocked_at": self.progress_metrics.get("blocked_at"),
            "diagnosing_at": self.progress_metrics.get("diagnosing_at"),
            "recovering_at": self.progress_metrics.get("recovering_at"),
        }
    
    def was_tool_called(self, tool_name: str, args: dict = None) -> bool:
        """Check if tool was already called with same args in this task."""
        if not tool_name or not args:
            return False
        arg_key = str(sorted((k, str(v)) for k, v in (args or {}).items()))
        history_key = f"{tool_name}:{arg_key}"
        return history_key in self.tool_history
    
    def get_duplicate_step_index(self, tool_name: str, args: dict = None) -> Optional[int]:
        """Get the step index where this tool+args was previously called."""
        if not tool_name or not args:
            return None
        arg_key = str(sorted((k, str(v)) for k, v in (args or {}).items()))
        history_key = f"{tool_name}:{arg_key}"
        return self.tool_history.get(history_key)
    
    def update_progress(self, metrics_update: dict) -> None:
        """Update progress metrics with new values."""
        self.progress_metrics.update(metrics_update)
        self.updated_at = datetime.datetime.now().isoformat()
        self.save()
    
    def check_stagnation(self) -> bool:
        """Check if task has stagnated (3+ steps without progress)."""
        stagnation = self.progress_metrics.get("stagnation_counter", 0)
        return stagnation >= 3
    
    def increment_stagnation(self) -> None:
        """Increment stagnation counter."""
        self.progress_metrics["stagnation_counter"] = self.progress_metrics.get("stagnation_counter", 0) + 1
        self.save()
    
    def reset_stagnation(self) -> None:
        """Reset stagnation counter on meaningful progress."""
        self.progress_metrics["stagnation_counter"] = 0
        self.progress_metrics["last_progress_step"] = len(self.completed_steps)
        self.save()
    
    def verify_goal(self) -> tuple[bool, List[dict]]:
        """
        Phase 5C: Verify all goal criteria for this task.
        Returns (all_passed: bool, results: List[dict]).
        """
        if not self.goal_criteria:
            return True, [{"message": "No goal criteria defined", "passed": True}]
        
        all_passed, results = verify_all_criteria(self.goal_criteria, self.project_root)
        
        # Update progress metrics with verification results
        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)
        self.progress_metrics["verification_results"] = results
        self.progress_metrics["criteria_passed"] = passed_count
        self.progress_metrics["criteria_total"] = len(results)
        self.updated_at = datetime.datetime.now().isoformat()
        self.save()
        
        # All required criteria must pass
        all_required_passed = all(
            result["passed"] for result in results if result.get("required", True)
        )
        return all_required_passed, results
    
    def set_goal_criteria(self, criteria: List[dict]) -> None:
        """Set goal criteria for this task."""
        self.goal_criteria = criteria
        self.updated_at = datetime.datetime.now().isoformat()
        self.save()
    
    def get_criteria_status(self) -> List[dict]:
        """Get current status of all criteria without re-verifying."""
        if not self.goal_criteria:
            return []
        return [
            {
                "type": c.get("type", "unknown"),
                "target": c.get("target", ""),
                "required": c.get("required", True),
            }
            for c in self.goal_criteria
        ]
    
    def get_context_for_prompt(self) -> str:
        """Generate task context string for injection into agent prompt."""
        lines = [
            f"ACTIVE TASK: {self.task_id[:8]}",
            f"Goal: {self.goal}",
            f"Project: {self.project_root or '(none)'}",
            f"Phase: {self.phase.upper()}",
            f"Status: {self.status.upper()}",
            f"Completed steps: {len(self.completed_steps)}",
        ]
        if self.temporal_context.get("note"):
            lines.append(f"Temporal context: {self.temporal_context['note']}")
        if self.goal_criteria:
            lines.append("Goal Criteria:")
            for i, c in enumerate(self.goal_criteria, 1):
                required = "✓" if c.get("required", True) else "○"
                lines.append(f"  {i}. {required} {c.get('type', '?')}: {c.get('target', '?')}")
        if self.completed_steps:
            lines.append("Completed:")
            for i, step in enumerate(self.completed_steps[-5:], 1):
                tool = step.get("tool", "?")
                success = "✓" if step.get("success") else "✗"
                lines.append(f"  {i}. {tool} {success}")
        if self.next_action:
            lines.append(f"Next action: {self.next_action}")
        pm = self.progress_metrics
        lines.append(f"Progress: {len(pm.get('files_inspected', []))} files inspected, "
                     f"{len(pm.get('files_modified', []))} files modified, "
                     f"{pm.get('tests_passed', 0)}/{pm.get('tests_run', 0)} tests passed")
        if pm.get("criteria_passed") is not None:
            lines.append(f"Criteria: {pm.get('criteria_passed', 0)}/{pm.get('criteria_total', 0)} passed")
        # Phase 7A: Budget status
        budget = self.execution_budget
        lines.append(f"Budget: LLM {budget['llm_calls']}/{budget['llm_budget']} calls, "
                     f"Tools {budget['tool_calls']}/{budget['tool_budget']} calls, "
                     f"Replans {budget['replans']}/{budget['replan_budget']}, "
                     f"Recoveries {budget['recoveries']}/{budget['recovery_budget']}, "
                     f"Time {budget['time_spent_seconds']}/{budget['time_budget_seconds']}s")
        return "\n".join(lines)

    def get_compact_planner_context(self) -> str:
        """Generate compact task context for planner (~500 tokens).
        
        Provides only the essential state needed for planning:
        - Goal, phase, completed steps, constraints, budget, next action
        Excludes conversation history, provider prompts, duplicated tool results.
        """
        completed_summary = ", ".join(
            f"{s.get('tool','?')} {'✓' if s.get('success') else '✗'}"
            for s in self.completed_steps[-5:]
        ) if self.completed_steps else "(none)"
        
        failed_tools = [s.get("tool","?") for s in self.completed_steps if not s.get("success")]
        
        criteria_summary = ", ".join(
            f"{c.get('type','?')}:{c.get('target','?')}"
            for c in self.goal_criteria
        ) if self.goal_criteria else "(none)"
        
        budget = self.execution_budget
        
        return (
            f"Task: {self.goal}\n"
            f"Phase: {self.phase}\n"
            f"Completed: {len(self.completed_steps)} steps ({completed_summary})\n"
            f"Failed: {failed_tools if failed_tools else '(none)'}\n"
            f"Constraints: {criteria_summary}\n"
            f"Budget: LLM {budget['llm_calls']}/{budget['llm_budget']}, "
            f"Tools {budget['tool_calls']}/{budget['tool_budget']}\n"
            f"Next: {self.next_action or '(none)'}"
        )

    def _validate_phase_transition(self, requested: str) -> bool:
        """Validate that a phase transition is allowed.
        
        Only allows: EXPLORE -> PLAN -> IMPLEMENT (monotonic forward).
        """
        _PHASE_ORDER = (TaskPhase.EXPLORE, TaskPhase.PLAN, TaskPhase.IMPLEMENT)
        if requested not in _PHASE_ORDER:
            return False
        try:
            cur_idx = _PHASE_ORDER.index(self.phase)
            req_idx = _PHASE_ORDER.index(requested)
        except ValueError:
            return False
        return req_idx >= cur_idx


def get_current_task(session_id: str = "default") -> Optional[ActiveTask]:
    """Return the current task pointer for a session (or None)."""
    _migrate_legacy()
    with _active_task_lock:
        current_id = _load_current_map().get(session_id)
    if not current_id:
        return None
    return ActiveTask.load(current_id)


def set_current_task(task: Optional[ActiveTask], session_id: str | None = None) -> None:
    """Set the current-task pointer for a session."""
    with _active_task_lock:
        data = _load_current_map()
        if task is None:
            data.pop(session_id or "default", None)
        else:
            data[task.session_id] = task.task_id
        _save_current_map(data)


def list_tasks(session_id: str | None = None) -> List[ActiveTask]:
    """List all persisted tasks, optionally filtered by session."""
    _migrate_legacy()
    tasks: List[ActiveTask] = []
    try:
        for name in os.listdir(_TASKS_DIR):
            if not name.startswith("task_") or not name.endswith(".json"):
                continue
            if name == "current.json":
                continue
            path = os.path.join(_TASKS_DIR, name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                task = ActiveTask.from_dict(data)
            except Exception:
                continue
            if session_id is None or task.session_id == session_id:
                tasks.append(task)
    except Exception:
        pass
    return tasks


def get_task(task_id: str) -> Optional[ActiveTask]:
    return ActiveTask.load(task_id)


def save_task(task: ActiveTask) -> None:
    task.save()


def archive_task(task_id: str) -> Optional[ActiveTask]:
    """Mark a task ARCHIVED (kept for reference, never resumable)."""
    task = ActiveTask.load(task_id)
    if task:
        task.status = "archived"
        task.save()
        if get_current_task(task.session_id) and get_current_task(task.session_id).task_id == task_id:
            set_current_task(None, task.session_id)
    return task


def pause_task(task_id: str) -> Optional[ActiveTask]:
    task = ActiveTask.load(task_id)
    if task:
        task.status = TaskStatus.PAUSED
        task.save()
    return task


def get_active_task() -> Optional[ActiveTask]:
    """Compat shim: the current task of the default session (never a global
    singleton across sessions)."""
    return get_current_task("default")


def set_active_task(task: Optional[ActiveTask]) -> None:
    """Compat shim: set the current-task pointer for the task's session."""
    set_current_task(task, (task.session_id if task else "default"))


def create_task(goal: str, project_root: str = "", session_id: str = "default",
                temporal_context: Dict = None, status: str = TaskStatus.ACTIVE) -> ActiveTask:
    """Create a new task and make it the current task for its session.

    The previous current task of the session is PAUSED (kept, but no longer
    the active in-conversation task).
    """
    _migrate_legacy()
    prev = get_current_task(session_id)
    task_id = f"task-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.urandom(4).hex()}"
    task = ActiveTask(
        task_id=task_id,
        goal=goal,
        project_root=project_root,
        session_id=session_id,
        status=status,
        temporal_context=temporal_context or {},
    )
    task.save()
    if prev and prev.task_id != task.task_id and prev.status in (TaskStatus.ACTIVE,):
        prev.status = TaskStatus.PAUSED
        prev.save()
    set_current_task(task)
    return task


def continue_task() -> Optional[ActiveTask]:
    """Compat shim: resume the current task of the default session."""
    task = get_current_task("default")
    if task and task.status in (TaskStatus.ACTIVE, TaskStatus.PAUSED):
        task.status = TaskStatus.ACTIVE
        task.save()
    return task


def complete_task(task_id: str | None = None, session_id: str = "default") -> Optional[ActiveTask]:
    """Mark a task COMPLETED and clear its current pointer if it is current."""
    task = task_id and ActiveTask.load(task_id) or get_current_task(session_id)
    if task:
        task.status = TaskStatus.COMPLETED
        task.save()
        cur = get_current_task(task.session_id)
        if cur and cur.task_id == task.task_id:
            set_current_task(None, task.session_id)
    return task


def get_task_context_for_prompt(session_id: str = "default") -> str:
    """Get formatted task context for injection into agent prompt."""
    task = get_current_task(session_id)
    if task:
        return task.get_context_for_prompt()
    return ""


# ─────────────────────────────────────────────
# CONTINUATION SIGNALS & DETERMINISTIC RESOLUTION
# ─────────────────────────────────────────────

# Continuation phrases are split by ambiguity.
#
# STRONG phrases are unambiguous operations on the CURRENT task's work
# ("run the searches", "retry the failed", "execute the plan"). They may be
# matched with word boundaries anywhere in a message.
#
# WEAK phrases ("do it", "run it", "write the file", "resume", ...) are also
# common in ordinary data-plane speech ("how do it work", "write the file to
# my Desktop", "resume the video"). They count as a continuation ONLY when the
# message is command-like — i.e. it consists almost entirely of recognized
# control vocabulary plus politeness filler — so a bare "do it" continues the
# task but "how do it work" / "do it yourself" / "run it in a sandbox" never do.
_STRONG_CONTINUATION_PHRASES = (
    "continue",
    "continue working",
    "continue the work",
    "keep going",
    "go on",
    "carry on",
    "don't stop",
    "do not stop",
    "proceed",
    "move on",
    "keep working",
    "run the searches",
    "run these searches",
    "run those searches",
    "execute the plan",
    "execute the searches",
    "do these searches",
    "do those searches",
    "retry the failed",
    "retry the failed searches",
    "retry the searches",
    "retry the steps",
    "retry the tool calls",
)

_WEAK_CONTINUATION_PHRASES = (
    "resume",
    "next",
    "go ahead",
    "finish it",
    "finish the job",
    "finish the research",
    "do it",
    "do the searches",
    "do the research",
    "do the work",
    "run it",
    "run them",
    "run the steps",
    "run the plan",
    "run the queries",
    "run the research",
    "retry them",
    "retry those",
    "write the file",
    "create the file",
    "make the file",
    "save the file",
)

# All vocabulary tokens that appear in any recognized continuation phrase —
# used by _is_command_like() to decide whether a message is genuinely a
# command rather than ordinary prose that merely contains a weak phrase.
# _CONTROL_CONFIRM_TOKENS adds the confirmation-vocabulary words ("apply it",
# "send it", "apply the change") so those phrases are also recognized as
# command-like; "apply this filter" still fails because "filter" is not here.
_CONTROL_CONFIRM_TOKENS = frozenset({"apply", "send", "change", "patch", "edit"})
_CONTROL_TOKEN_SET = frozenset(
    w
    for _p in _STRONG_CONTINUATION_PHRASES + _WEAK_CONTINUATION_PHRASES
    for w in _p.split()
) | _CONTROL_CONFIRM_TOKENS

# Politeness / discourse filler that may surround a control command without
# changing its nature ("please run the searches now", "okay do it").
_CONTROL_FILLER = frozenset({
    "please", "pls", "now", "okay", "ok", "yes", "yeah", "yep", "yup", "sure",
    "thanks", "thank", "ty", "again", "alright", "right", "just", "kindly",
    "hey", "hi", "so", "well", "man", "then",
})

_CONTROL_CONNECTORS = frozenset({"and", "or", "but", ",", "&"})

# Negation markers. A weak phrase inside a negated message ("don't do it") is
# never a command — only STRONG phrases like "don't stop" survive negation.
_CONTROL_NEGATION = frozenset({"don't", "dont", "not", "never", "can't", "cannot", "nope", "no"})


def _phrase_re(phrase: str) -> str:
    """Word-boundary regex for a phrase, tolerant of apostrophes/plurals."""
    return r"\b" + re.escape(phrase) + r"\b"


def _is_command_like(text: str) -> bool:
    """True when ``text`` is essentially a control command.

    A message is command-like when every token is either part of the control
    vocabulary, politeness filler, or a connector — and it is not negated.
    This is what keeps a bare "do it" (a continuation) distinct from "how do it
    work" / "do it yourself" / "run it in a sandbox" / "don't do it" (ordinary
    data-plane requests).
    """
    tokens = re.findall(r"[a-z0-9&']+", (text or "").lower())
    if not tokens:
        return False
    for tok in tokens:
        if tok in _CONTROL_NEGATION:
            return False
        if tok in _CONTROL_FILLER or tok in _CONTROL_CONNECTORS:
            continue
        if tok in _CONTROL_TOKEN_SET:
            continue
        return False
    return True


def has_continuation_signal(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "").lower().strip())
    if any(re.search(_phrase_re(p), t) for p in _STRONG_CONTINUATION_PHRASES):
        return True
    if any(re.search(_phrase_re(p), t) for p in _WEAK_CONTINUATION_PHRASES):
        return _is_command_like(t)
    return False

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in",
    "on", "for", "and", "or", "but", "it", "this", "that", "these", "those",
    "i", "you", "we", "they", "he", "she", "with", "from", "at", "by",
    "as", "do", "does", "did", "have", "has", "had", "can", "could", "will",
    "would", "should", "then", "there", "their", "them", "not", "so", "my",
    "your", "our", "about", "please", "make", "get", "find", "search", "also",
    "before", "after", "fix", "fixes", "what", "when", "where", "which", "who",
    "your", "some", "any", "new", "just", "let", "like", "want", "need",
}


def _significant_tokens(text: str) -> set:
    return {
        tok for tok in re.findall(r"[a-z][a-z0-9'-]{1,30}", (text or "").lower())
        if tok not in _STOPWORDS and len(tok) > 1
    }


def _token_overlap(a: set, b: set) -> int:
    return len(a & b)


def is_continuation_request(text: str) -> bool:
    """Keep the original name as a pure signal check (used by tests/tools).

    NOTE: a continuation signal ALONE never selects a task anymore. Use
    resolve_task() for actual task selection.
    """
    return has_continuation_signal(text)


# ── Control-plane command detection ─────────────────────────────────────────
# Control commands operate on the CURRENT task's persisted work (resume the
# plan, retry failed steps, apply a pending write). They must be distinctive
# phrases — a broad matcher on "run"/"retry"/"write" alone would hijack normal
# data-plane requests ("run a marathon plan", "write a poem", "retry the
# download" as a new task). Cross-session/stale-task protection stays in
# resolve_task(); these patterns only gate the control handler in brain.
# The same strong/weak split as the continuation phrases applies: weak patterns
# only count when the message is command-like (_is_command_like).

# type -> (strong patterns, weak patterns)
_CONTROL_PATTERNS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "resume_plan": (
        (
            r"\b(run|execute)\s+(the|these|those)\s+searches\b",
            r"\bexecute\s+(the\s+)?plan\b",
            r"\bdo\s+(these|those)\s+(searches|research|work)\b",
            r"\b(keep going|carry on|continue|proceed|move on|go on|keep working|"
            r"don't stop|do not stop|continue working)\b",
        ),
        (
            r"\bresume\b",
            r"\bnext\b",
            r"\bgo ahead\b",
            r"\bfinish\s+it\b",
            r"\bfinish\s+the\s+(job|research)\b",
            r"\bdo\s+it\b",
            r"\b(do|run)\s+the\s+(searches|research|work)\b",
            r"\brun\s+(it|them)\b",
            r"\brun\s+the\s+(steps|plan|queries|research)\b",
        ),
    ),
    "retry_failed": (
        (
            r"\bre[-]?try\s+the\s+failed\b",
            r"\bre[-]?try\s+the\s+failed\s+searches\b",
            r"\bre[-]?try\s+the\s+searches\b",
            r"\bre[-]?try\s+the\s+steps\b",
            r"\bre[-]?try\s+the\s+tool\s+calls\b",
        ),
        (
            r"\bre[-]?try\s+them\b",
            r"\bre[-]?try\s+those\b",
        ),
    ),
    "apply_pending": (
        (),
        (
            r"\b(write|create|make|save)\s+the\s+file\b",
            r"\bapply\s+(it|the\s+(change|patch|edit|file|write))\b",
            r"\byes[,\s]*(write|create|save|apply|do\s+it)\b",
            r"\b(go ahead and|please)\s+(write|create|save|apply)\b",
            r"\b(confirm|approved)\b",
        ),
    ),
}

_CONTROL_REPLACE_TOOL = [
    r"\b(normal\s+)?web[-_ ]?search\b",
    r"\bweb[-_ ]?search\s+tool\b",
]


def detect_control_command(text: str) -> list[dict]:
    """Detect a control-plane directive for the current task.

    Returns a list of {"type": ..., "replace_tool": ...} in a sensible order
    (apply pending writes first, then resume/retry the plan). Only distinctive
    phrases trigger; a message that is really new research/data-plane content
    will return [] and flow through normal routing. Weak (ambiguous) phrases
    ("do it", "write the file") only fire when the message is command-like, so
    "how do it work" / "write the file to my Desktop" never enter the control
    plane.
    """
    t = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not t:
        return []
    is_command = _is_command_like(t)
    directives: list[dict] = []
    seen: set = set()

    def _push(typ, replace_tool=None):
        key = (typ, replace_tool)
        if key in seen:
            return
        seen.add(key)
        d = {"type": typ}
        if replace_tool:
            d["replace_tool"] = replace_tool
        directives.append(d)

    for typ, (strong, weak) in _CONTROL_PATTERNS.items():
        if any(re.search(p, t) for p in strong):
            _push(typ)
        elif is_command and any(re.search(p, t) for p in weak):
            _push(typ)

    # Optional tool replacement: "use/using normal web_search instead"
    replace_tool = None
    if re.search(r"(?:use|using)\s+(?:normal\s+)?web[-_ ]?search", t) or re.search(r"web[-_ ]?search\s+tool", t):
        replace_tool = "web_search"
    # Reorder: apply pending first, then retry, then resume.
    order = {"apply_pending": 0, "retry_failed": 1, "resume_plan": 2}
    directives.sort(key=lambda d: order.get(d["type"], 9))
    if replace_tool:
        for d in directives:
            if d["type"] in ("retry_failed", "resume_plan"):
                d["replace_tool"] = replace_tool
    return directives


def add_plan_to_task(task_id: str, plan_id: str) -> None:
    """Record that ``plan_id`` is owned by task ``task_id`` (plan linkage).

    Writes BOTH directions so the ownership is discoverable regardless of
    which store is queried first:
      - task registry: task.plan_ids (in-memory + persisted)
      - plans store:   plans.active_task_id (so load_plan_for_task() finds it)
    """
    task = ActiveTask.load(task_id)
    if task:
        if plan_id and plan_id not in task.plan_ids:
            task.plan_ids.append(plan_id)
            task.save()
    if plan_id:
        try:
            import plans

            plans.set_plan_active_task(plan_id, task_id)
        except Exception:
            pass


def _reference_cue(text: str) -> bool:
    """Does the message use a task-reference cue (id, 'task', 'that', 'this')?"""
    t = (text or "").lower()
    if re.search(r"\btask[-_][a-z0-9]+\b|\btask\s+(?:[a-z0-9-]{3,})\b", t):
        return True
    if re.search(r"\b(resume|continue|finish|proceed|work on|pick up|that)\b", t):
        return True
    return False


def _explicit_task_ref(message: str, candidates: List[ActiveTask]) -> Optional[ActiveTask]:
    """Deterministic explicit reference: task id present, or a distinctive
    goal-phrase match alongside a reference cue."""
    m_lower = message.lower()
    for task in candidates:
        if task.task_id and task.task_id.lower() in m_lower:
            return task
        if len(task.task_id) >= 8 and task.task_id[:8].lower() in m_lower:
            return task
    msg_tokens = _significant_tokens(message)
    if _reference_cue(message):
        best = None
        best_score = 0
        for task in candidates:
            goal_tokens = _significant_tokens(task.goal)
            score = _token_overlap(msg_tokens, goal_tokens)
            if score >= 2 and score > best_score:
                best = task
                best_score = score
        if best:
            return best
    return None


def token_overlap(text_a: str, text_b: str) -> int:
    """Public deterministic token-overlap score between two strings."""
    return _token_overlap(_significant_tokens(text_a), _significant_tokens(text_b))


def _best_overlap_task(message: str, candidates: List[ActiveTask]) -> Optional[ActiveTask]:
    """Deterministic semantic overlap (token Jaccard-like count). No LLM."""
    msg_tokens = _significant_tokens(message)
    if not msg_tokens:
        return None
    best = None
    best_score = 0
    for task in candidates:
        goal_tokens = _significant_tokens(task.goal)
        score = _token_overlap(msg_tokens, goal_tokens)
        if score >= 2 and score > best_score:
            best = task
            best_score = score
    return best


def resolve_task(session_id: str, message: str) -> Dict[str, Any]:
    """Deterministically resolve which task (if any) a message refers to.

    Returns {"action": str, "task": Optional[ActiveTask], "reason": str} where
    action is one of:
      "explicit"  — the message explicitly references a task
      "resume"    — continuation of the current in-session task or a strongly
                    overlapping same-session active/paused task
      "new"       — explicit new intent; caller should create a new task
      "none"      — no task involvement (chat / correction / unrelated)

    Precedence (mandated):
      current explicit intent
        > conversation-local (current) task reference
        > strong semantic relationship (same session)
        > persisted task state (same session only)
        > never: cross-session, completed/archived, unrelated stale tasks
    """
    current = get_current_task(session_id)
    same_session = [t for t in list_tasks(session_id) if t.task_id != (current.task_id if current else None)]

    resumable = lambda t: t is not None and t.status in (TaskStatus.ACTIVE, TaskStatus.PAUSED)

    # 1. Explicit reference (id or distinctive goal phrase + cue)
    explicit = _explicit_task_ref(message, [t for t in ([current] if current else []) + same_session])
    if explicit and resumable(explicit):
        return {"action": "explicit", "task": explicit, "reason": "explicit task reference"}

    # 1b. Explicit task-id token that is NOT a same-session task (cross-session
    #     or unknown): never fall through to resume the current task instead.
    _id_ref = re.search(r"\btask[-_][a-z0-9]{3,}\b", message.lower())
    if _id_ref:
        ref_id = _id_ref.group(0)
        known = {t.task_id.lower() for t in ([current] if current else []) + same_session}
        if ref_id not in known:
            return {"action": "none", "task": None, "reason": f"explicit reference to unknown/cross-session task {ref_id}"}

    cont = has_continuation_signal(message)

    # 2. Current in-session task + continuation signal
    if current and cont and resumable(current):
        return {"action": "resume", "task": current, "reason": "current task + continuation signal"}

    # 3. Same-session active/paused task with strong semantic overlap + continuation
    if cont:
        overlap = _best_overlap_task(message, [t for t in same_session if resumable(t)])
        if overlap:
            return {"action": "resume", "task": overlap, "reason": "semantic overlap + continuation signal"}

    # 4. Continuation signal but nothing eligible -> never hijack a stale task
    if cont:
        return {"action": "none", "task": None, "reason": "continuation signal but no eligible in-session task"}

    # 5. Otherwise: fresh intent
    return {"action": "new", "task": None, "reason": "no continuation binding; fresh request"}


# Phase 6C/D: Failure Diagnosis & Recovery Planning

def diagnose_failure(verification_results: List[dict], project_root: str = "") -> str:
    """
    Phase 6C: Analyze verification failures and suggest root cause.
    Returns a diagnostic string.
    """
    failed = [r for r in verification_results if not r.get("passed") and r.get("required", True)]
    if not failed:
        return "No required criteria failed."
    
    diagnoses = []
    for f in failed:
        ctype = f.get("type", "")
        target = f.get("target", "")
        msg = f.get("message", "")
        
        if ctype == "tests_pass":
            diagnoses.append(f"Tests failing in {target}: {msg[:200]}")
        elif ctype == "syntax_valid":
            diagnoses.append(f"Syntax error in {target}: {msg[:200]}")
        elif ctype == "file_exists":
            diagnoses.append(f"Missing file {target}: may need to create or locate it")
        elif ctype == "file_contains":
            diagnoses.append(f"Expected content not found in {target}: missing '{f.get('substring', '')}'")
        elif ctype == "file_not_contains":
            diagnoses.append(f"Forbidden content found in {target}: contains '{f.get('substring', '')}'")
        elif ctype == "command_succeeds":
            diagnoses.append(f"Command failed: {target} - {msg[:200]}")
        else:
            diagnoses.append(f"Criterion {ctype} failed for {target}: {msg[:200]}")
    
    return "; ".join(diagnoses)


def suggest_recovery_plan(verification_results: List[dict], project_root: str = "") -> str:
    """
    Phase 6D: Generate a recovery plan based on verification failures.
    Returns a step-by-step recovery plan string.
    """
    failed = [r for r in verification_results if not r.get("passed") and r.get("required", True)]
    if not failed:
        return "All criteria passing - no recovery needed."
    
    steps = []
    for f in failed:
        ctype = f.get("type", "")
        target = f.get("target", "")
        
        if ctype == "tests_pass":
            steps.append(f"1. Run tests for {target} to see detailed failures")
            steps.append(f"2. Fix failing tests in {target}")
            steps.append(f"3. Re-run tests to verify fixes")
        elif ctype == "syntax_valid":
            steps.append(f"1. Open {target} and fix syntax error")
            steps.append(f"2. Re-verify syntax")
        elif ctype == "file_exists":
            steps.append(f"1. Create or restore {target}")
            steps.append(f"2. Verify file exists and has correct content")
        elif ctype == "file_contains":
            substring = f.get("substring", "")
            steps.append(f"1. Add '{substring}' to {target}")
            steps.append(f"2. Verify content is present")
        elif ctype == "file_not_contains":
            substring = f.get("substring", "")
            steps.append(f"1. Remove '{substring}' from {target}")
            steps.append(f"2. Verify content removed")
        elif ctype == "command_succeeds":
            steps.append(f"1. Run command manually to see error: {target}")
            steps.append(f"2. Fix underlying issue")
            steps.append(f"3. Re-run command to verify")
        else:
            steps.append(f"1. Address {ctype} failure for {target}")
    
    if not steps:
        steps.append("1. Investigate unknown failure type")
        steps.append("2. Apply appropriate fix")
    
    return "\n".join(steps)


def run_recovery_cycle(task: "ActiveTask", execute_tool_fn, ask_llm_fn, max_cycles: int = 3) -> bool:
    """
    Phase 6E: Run a recovery cycle - diagnose, plan, execute, verify.
    Returns True if recovery succeeded (all criteria pass).
    """
    if not task.goal_criteria:
        return True
    
    for cycle in range(max_cycles):
        # Verify current state
        passed, results = task.verify_goal()
        if passed:
            return True
        
        # Diagnose
        diagnosis = diagnose_failure(results)
        task.enter_diagnosing(diagnosis)
        
        # Plan recovery
        plan = suggest_recovery_plan(results)
        task.enter_recovering(plan)
        
        # Execute recovery via agent (this would need the agent loop)
        # For now, we just return the plan
        # The agent loop should pick up the recovery plan
        
        # Try verification again
        passed, results = task.verify_goal()
        if passed:
            task.resume_active()
            return True
    
    return False
