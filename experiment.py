"""
Experiment System (Phase 7C) — Autonomous Experiment Engine
Persistent experiment state with full lifecycle management.
"""

import datetime
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Literal

from task import ActiveTask, get_active_task, continue_task, complete_task, TaskStatus


# ─────────────────────────────────────────────
# EXPERIMENT RESULT MODEL
# ─────────────────────────────────────────────

class ExperimentStatus:
    PROPOSED = "proposed"
    BASELINED = "baselined"
    RUNNING = "running"
    MEASURED = "measured"
    VERIFIED = "verified"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"


@dataclass
class ExperimentResult:
    """
    Complete record of an autonomous experiment.

    This is the canonical record of what JARVIS tried, what happened,
    and what the verifier concluded.
    """
    experiment_id: str
    task_id: str
    hypothesis: str
    status: str = ExperimentStatus.PROPOSED

    # Baseline (before any changes)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_timestamp: str = ""

    # Modifications made
    modifications: List[Dict[str, Any]] = field(default_factory=list)

    # Measurements after modifications
    current_metrics: Dict[str, Any] = field(default_factory=dict)
    measurement_timestamp: str = ""

    # Goal criteria that define success
    goal_criteria: List[Dict[str, Any]] = field(default_factory=list)

    # Verification results
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    all_criteria_passed: bool = False

    # Decision
    decision: str = ""  # ACCEPT, REJECT, INCONCLUSIVE, BLOCKED
    decision_reason: str = ""

    # Recovery tracking
    recovery_cycles: int = 0
    recovery_history: List[Dict[str, Any]] = field(default_factory=list)

    # Budget tracking
    execution_budget: Dict[str, Any] = field(default_factory=lambda: {
        "time_budget_seconds": 600,
        "llm_budget": 20,
        "tool_budget": 50,
        "replan_budget": 3,
        "recovery_budget": 2,
        "time_spent_seconds": 0,
        "llm_calls_used": 0,
        "tool_calls_used": 0,
        "replans_used": 0,
        "recovery_cycles_used": 0,
    })

    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        now = datetime.datetime.now().isoformat()
        if not self.experiment_id:
            self.experiment_id = f"exp-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if not self.baseline_timestamp and self.baseline_metrics:
            self.baseline_timestamp = now

    def to_dict(self) -> dict:
            return {
                "experiment_id": self.experiment_id,
                "task_id": self.task_id,
                "hypothesis": self.hypothesis,
                "status": self.status,
                "baseline_metrics": self.baseline_metrics,
                "baseline_timestamp": self.baseline_timestamp,
                "modifications": self.modifications,
                "current_metrics": self.current_metrics,
                "measurement_timestamp": self.measurement_timestamp,
                "goal_criteria": self.goal_criteria,
                "verification_results": self.verification_results,
                "all_criteria_passed": self.all_criteria_passed,
                "decision": self.decision,
                "decision_reason": self.decision_reason,
                "recovery_cycles": self.recovery_cycles,
                "recovery_history": self.recovery_history,
                "execution_budget": self.execution_budget,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
            }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        exp = cls(
            experiment_id=data.get("experiment_id", ""),
            task_id=data.get("task_id", ""),
            hypothesis=data.get("hypothesis", ""),
            status=data.get("status", ExperimentStatus.PROPOSED),
            baseline_metrics=data.get("baseline_metrics", {}),
            baseline_timestamp=data.get("baseline_timestamp", ""),
            modifications=data.get("modifications", []),
            current_metrics=data.get("current_metrics", {}),
            measurement_timestamp=data.get("measurement_timestamp", ""),
            goal_criteria=data.get("goal_criteria", []),
            verification_results=data.get("verification_results", []),
            all_criteria_passed=data.get("all_criteria_passed", False),
            decision=data.get("decision", ""),
            decision_reason=data.get("decision_reason", ""),
            recovery_cycles=data.get("recovery_cycles", 0),
            recovery_history=data.get("recovery_history", []),
            execution_budget=data.get("execution_budget", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
        )
        return exp

    def save(self, conn: sqlite3.Connection) -> None:
        """Persist experiment to database."""
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments (
                    experiment_id, task_id, hypothesis, status,
                    baseline_metrics, baseline_timestamp,
                    modifications, current_metrics, measurement_timestamp,
                    goal_criteria, verification_results, all_criteria_passed,
                    decision, decision_reason,
                    recovery_cycles, recovery_history,
                    execution_budget,
                    created_at, updated_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.experiment_id,
                self.task_id,
                self.hypothesis,
                self.status,
                json.dumps(self.baseline_metrics),
                self.baseline_timestamp,
                json.dumps(self.modifications),
                json.dumps(self.current_metrics),
                self.measurement_timestamp,
                json.dumps(self.goal_criteria),
                json.dumps(self.verification_results),
                int(self.all_criteria_passed),
                self.decision,
                self.decision_reason,
                self.recovery_cycles,
                json.dumps(self.recovery_history),
                json.dumps(self.execution_budget),
                self.created_at,
                self.updated_at,
                self.started_at,
                self.completed_at,
            ))

    @classmethod
    def load(cls, conn: sqlite3.Connection, experiment_id: str) -> Optional["ExperimentResult"]:
        cursor = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        # Convert row to dict
        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, row))

        # Parse JSON fields
        for key in ["baseline_metrics", "modifications", "current_metrics", 
                    "goal_criteria", "verification_results", "recovery_history",
                    "execution_budget"]:
            if data.get(key):
                data[key] = json.loads(data[key])

        data["all_criteria_passed"] = bool(data.get("all_criteria_passed", 0))

        return cls.from_dict(data)

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget utilization."""
        b = self.execution_budget
        return {
            "time_budget_seconds": b.get("time_budget_seconds", 600),
            "llm_budget": b.get("llm_budget", 20),
            "tool_budget": b.get("tool_budget", 50),
            "replan_budget": b.get("replan_budget", 3),
            "recovery_budget": b.get("recovery_budget", 2),
            "llm_calls_used": b.get("llm_calls_used", 0),
            "tool_calls_used": b.get("tool_calls_used", 0),
            "replans_used": b.get("replans_used", 0),
            "recovery_cycles_used": b.get("recovery_cycles_used", 0),
        }

    def check_budget_exhausted(self) -> Optional[str]:
        """Check if any budget is exhausted. Returns reason if exhausted, None otherwise."""
        b = self.execution_budget
        if b.get("llm_calls_used", 0) >= b.get("llm_budget", 20):
            return f"LLM budget exhausted ({b.get('llm_calls_used', 0)}/{b.get('llm_budget', 20)} calls)"
        if b.get("tool_calls_used", 0) >= b.get("tool_budget", 50):
            return f"Tool budget exhausted ({b.get('tool_calls_used', 0)}/{b.get('tool_budget', 50)} calls)"
        if b.get("replans_used", 0) >= b.get("replan_budget", 3):
            return f"Replan budget exhausted ({b.get('replans_used', 0)}/{b.get('replan_budget', 3)} replans)"
        if b.get("recovery_cycles_used", 0) >= b.get("recovery_budget", 2):
            return f"Recovery budget exhausted ({b.get('recovery_cycles_used', 0)}/{b.get('recovery_budget', 2)} recoveries)"
        return None

    def record_modification(self, tool_name: str, args: dict, result: str, verified: bool = False) -> None:
        """Record a modification step."""
        self.modifications.append({
            "step": len(self.modifications) + 1,
            "tool": tool_name,
            "args": args,
            "result": result[:500] if len(result) > 500 else result,
            "verified": verified,
            "timestamp": datetime.datetime.now().isoformat()
        })
        self.tool_calls_used += 1
        self.updated_at = datetime.datetime.now().isoformat()

    def record_baseline(self, metrics: Dict[str, Any]) -> None:
        """Record baseline metrics."""
        self.baseline_metrics = metrics
        self.baseline_timestamp = datetime.datetime.now().isoformat()
        self.status = ExperimentStatus.BASELINED
        self.updated_at = datetime.datetime.now().isoformat()

    def record_measurement(self, metrics: Dict[str, Any]) -> None:
        """Record current metrics after modifications."""
        self.current_metrics = metrics
        self.measurement_timestamp = datetime.datetime.now().isoformat()
        self.status = ExperimentStatus.MEASURED
        self.updated_at = datetime.datetime.now().isoformat()

    def record_verification(self, results: List[Dict[str, Any]], all_passed: bool) -> None:
        """Record verification results."""
        self.verification_results = results
        self.all_criteria_passed = all_passed
        self.status = ExperimentStatus.VERIFIED
        self.updated_at = datetime.datetime.now().isoformat()

    def make_decision(self, decision: str, reason: str) -> None:
        """Record final experiment decision."""
        valid_decisions = ["ACCEPT", "REJECT", "INCONCLUSIVE", "BLOCKED"]
        if decision not in valid_decisions:
            raise ValueError(f"Invalid decision: {decision}. Must be one of {valid_decisions}")

        self.decision = decision
        self.decision_reason = reason
        self.status = ExperimentStatus(decision.lower())
        self.completed_at = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now().isoformat()


    # ─────────────────────────────────────────────
# EXPERIMENT DATABASE
# ─────────────────────────────────────────────

_EXPERIMENT_DB_FILE = os.path.expanduser("~/.jarvis/experiments.db")
_experiment_db_lock = threading.RLock()


def get_experiment_db() -> sqlite3.Connection:
    """Get (or create) the experiment database connection."""
    os.makedirs(os.path.dirname(_EXPERIMENT_DB_FILE), exist_ok=True)
    conn = sqlite3.connect(_EXPERIMENT_DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init_experiment_schema(conn)
    return conn


def _init_experiment_schema(conn: sqlite3.Connection) -> None:
    """Initialize the experiment database schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            status TEXT NOT NULL,
            baseline_metrics TEXT,
            baseline_timestamp TEXT,
            modifications TEXT,
            current_metrics TEXT,
            measurement_timestamp TEXT,
            goal_criteria TEXT,
            verification_results TEXT,
            all_criteria_passed INTEGER,
            decision TEXT,
            decision_reason TEXT,
            recovery_cycles INTEGER DEFAULT 0,
            recovery_history TEXT,
            llm_calls_used INTEGER DEFAULT 0,
            tool_calls_used INTEGER DEFAULT 0,
            replans_used INTEGER DEFAULT 0,
            recovery_cycles_used INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT
        )
    """)

    # Index for common queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_task_id ON experiments(task_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at)")
    conn.commit()


def get_experiment_conn() -> sqlite3.Connection:
    """Get the experiment database connection."""
    return get_experiment_db()


def create_experiment(task_id: str, hypothesis: str, goal_criteria: List[Dict[str, Any]] = None) -> ExperimentResult:
    """Create a new experiment."""
    exp = ExperimentResult(
        experiment_id=f"exp-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
        task_id=task_id,
        hypothesis=hypothesis,
        goal_criteria=goal_criteria or [],
        status=ExperimentStatus.PROPOSED,
    )

    conn = get_experiment_db()
    exp.save(conn)
    return exp


def get_experiment(experiment_id: str) -> Optional[ExperimentResult]:
    """Load an experiment by ID."""
    conn = get_experiment_db()
    return ExperimentResult.load(conn, experiment_id)


def get_experiments_by_task(task_id: str) -> List[ExperimentResult]:
    """Get all experiments for a task."""
    conn = get_experiment_db()
    cursor = conn.execute(
        "SELECT experiment_id FROM experiments WHERE task_id = ? ORDER BY created_at DESC",
        (task_id,)
    )
    return [ExperimentResult.load(conn, row[0]) for row in cursor.fetchall()]


def get_latest_experiment() -> Optional[ExperimentResult]:
    """Get the most recently created experiment."""
    conn = get_experiment_db()
    cursor = conn.execute(
        "SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT 1"
    )
    row = cursor.fetchone()
    if row:
        return ExperimentResult.load(conn, row[0])
    return None


# ─────────────────────────────────────────────
# BASELINE MANAGEMENT
# ─────────────────────────────────────────────

_BASELINE_DIR = os.path.expanduser("~/.jarvis/baselines")
_baseline_lock = threading.RLock()


def save_baseline(task_id: str, name: str, metrics: Dict[str, Any]) -> str:
    """Save a baseline measurement for a task."""
    os.makedirs(_BASELINE_DIR, exist_ok=True)

    baseline_id = f"{task_id}-{name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    baseline_file = os.path.join(_BASELINE_DIR, f"{baseline_id}.json")

    baseline_data = {
        "baseline_id": baseline_id,
        "task_id": task_id,
        "name": name,
        "metrics": metrics,
        "created_at": datetime.datetime.now().isoformat(),
    }

    with _baseline_lock:
        with open(baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

    return baseline_id


def load_baseline(baseline_id: str) -> Optional[Dict[str, Any]]:
    """Load a baseline by ID."""
    with _baseline_lock:
        baseline_file = os.path.join(_BASELINE_DIR, f"{baseline_id}.json")
        if not os.path.exists(baseline_file):
            return None

        with open(baseline_file, "r") as f:
            return json.load(f)


def list_baselines(task_id: str = None) -> List[Dict[str, Any]]:
    """List all baselines, optionally filtered by task."""
    with _baseline_lock:
        if not os.path.exists(_BASELINE_DIR):
            return []

        baselines = []
        for filename in os.listdir(_BASELINE_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(_BASELINE_DIR, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    if task_id is None or data.get("task_id") == task_id:
                        baselines.append(data)
                except Exception:
                    pass
        return baselines


# ─────────────────────────────────────────────
# EXPERIMENT CONTROLLER
# ─────────────────────────────────────────────

class ExperimentController:
    """
    Orchestrates the full experiment lifecycle:
    PROPOSED → BASELINED → RUNNING → MEASURED → VERIFIED → ACCEPT/REJECT/INCONCLUSIVE/BLOCKED
    """

    def __init__(
        self,
        execute_tool_fn,
        ask_llm_fn,
        speak_fn=None,
        max_iterations: int = 30,
    ):
        self.execute_tool_fn = execute_tool_fn
        self.ask_llm_fn = ask_llm_fn
        self.speak_fn = speak_fn
        self.max_iterations = max_iterations

        # Budget from environment (inherits from agent)
        self.max_tool_calls = int(os.getenv("JARVIS_AGENT_MAX_TOOL_CALLS", "10"))
        self.max_explore = int(os.getenv("JARVIS_AGENT_MAX_EXPLORE", "8"))
        self.wall_clock_s = float(os.getenv("JARVIS_AGENT_WALL_CLOCK_S", "90"))

    def run_experiment(
        self,
        hypothesis: str,
        goal_criteria: List[Dict[str, Any]],
        project_root: str = "",
        initial_task: ActiveTask = None,
    ) -> ExperimentResult:
        """
        Run a complete experiment from hypothesis to decision.

        Returns the ExperimentResult with final decision.
        """
        # Create or resume experiment
        if initial_task:
            task = initial_task
        else:
            from task import create_task
            task = create_task(hypothesis, project_root=os.getcwd())

        # Set goal criteria on task
        task.set_goal_criteria(goal_criteria)

        # Create experiment record
        experiment = create_experiment(
            task_id=task.task_id,
            hypothesis=hypothesis,
            goal_criteria=goal_criteria,
        )

        experiment.started_at = datetime.datetime.now().isoformat()
        experiment.save(get_experiment_db())

        # Link experiment to task
        task.execution_budget = task.execution_budget or {
            "time_budget_seconds": 600,
            "llm_budget": 20,
            "tool_budget": 50,
            "replan_budget": 3,
            "recovery_budget": 2,
            "time_spent_seconds": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "replans": 0,
            "recoveries": 0,
        }
        task.save()

        # Run the experiment loop
        experiment = self._run_experiment_loop(
            experiment=experiment,
            task=task,
            hypothesis=hypothesis,
            goal_criteria=goal_criteria,
            project_root=project_root,
        )

        experiment.completed_at = datetime.datetime.now().isoformat()
        experiment.save(get_experiment_db())

        return experiment

    def _run_experiment_loop(
        self,
        experiment: ExperimentResult,
        task: ActiveTask,
        hypothesis: str,
        goal_criteria: List[Dict[str, Any]],
        project_root: str,
    ) -> ExperimentResult:
        """Main experiment loop."""
        from agent import run_agent_loop, _execute_tool

        experiment.status = ExperimentStatus.BASELINED
        experiment.save(get_experiment_db())

        # Phase 1: Baseline measurement
        baseline_metrics = self._capture_baseline(goal_criteria)
        experiment.record_baseline(baseline_metrics)
        experiment.save(get_experiment_db())

        # Phase 2: Main experiment loop
        max_iterations = 30
        tool_calls = 0
        explore_calls = 0
        empty_decisions = 0
        phase = "explore"
        start_time = time.time()
        deadline = time.time() + 90  # wall clock

        from agent import Agent
        agent = Agent(goal=hypothesis, max_iterations=max_iterations)

        # Pre-populate with task's tool history
        for history_key, step_idx in task.tool_history.items():
            if ":" in history_key:
                tool_name, arg_key = history_key.split(":", 1)
                agent.already_called.add((tool_name, arg_key))

        # Inject task context into goal
        if task.get_context_for_prompt():
            goal_with_context = f"{task.get_context_for_prompt()}\n\nCurrent Goal: {hypothesis}"
        else:
            goal_with_context = hypothesis

        agent.goal = goal_with_context

        # Re-use the agent loop logic but with experiment tracking
        # We'll delegate to run_agent_loop but with experiment tracking
        # For now, we'll run a simplified version

        # Delegate to the existing agent loop but with experiment tracking
        # The run_agent_loop will handle the agent loop
        # We need to wrap it to capture experiment data

        # Use the existing run_agent_loop but with experiment tracking
        # We'll patch the execute_tool_fn to record experiment data
        original_execute = _execute_tool

        def tracked_execute(tool_name: str, args: dict) -> str:
            nonlocal experiment
            result = original_execute(tool_name, args)
            verified = "Verification Failed" not in str(result) and "error" not in str(result).lower()
            experiment.record_modification(tool_name, args, str(result), verified)
            experiment.save(get_experiment_db())
            return result

        # Run the agent loop with experiment tracking
        result = run_agent_loop(
            goal=hypothesis,
            execute_tool_fn=tracked_execute,
            ask_llm_fn=self.ask_llm_fn,
            speak_fn=self.speak_fn,
            max_iterations=self.max_iterations,
            task=experiment,
        )

        # After agent loop, measure and verify
        final_metrics = self._capture_final_metrics()
        experiment.record_measurement(final_metrics)
        experiment.save(get_experiment_db())

        # Verify goal
        passed, verification_results = task.verify_goal()
        experiment.record_verification(verification_results, passed)
        experiment.save(get_experiment_db())

        if passed:
            experiment.make_decision("ACCEPT", "All goal criteria verified successfully")
        else:
            # Try recovery
            recovery_result = self._attempt_recovery(experiment, task)
            if recovery_result == "REJECT":
                experiment.make_decision("REJECT", "Recovery attempts exhausted, goal not achievable")
            elif recovery_result == "INCONCLUSIVE":
                experiment.make_decision("INCONCLUSIVE", "Unable to determine success within budget")
            elif recovery_result == "BLOCKED":
                experiment.make_decision("BLOCKED", "External blocker preventing progress")
            else:
                experiment.make_decision("INCONCLUSIVE", "Recovery incomplete within budget")

        experiment.save(get_experiment_db())
        return experiment

    def _capture_baseline(self, goal_criteria: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Capture baseline metrics before any modifications."""
        # This would run the initial verification to get baseline
        # For now, return empty - actual implementation depends on criteria
        return {}

    def _capture_final_metrics(self) -> Dict[str, Any]:
        """Capture final metrics after modifications."""
        return {}

    def _attempt_recovery(self, experiment: ExperimentResult, task: ActiveTask) -> str:
        """
        Attempt recovery when goal verification fails.
        Returns: "REJECT", "INCONCLUSIVE", "BLOCKED", or "CONTINUE"
        """
        from task import diagnose_failure, suggest_recovery_plan, run_recovery_cycle

        # Check recovery budget
        if experiment.recovery_cycles_used >= 2:
            return "BLOCKED"

        # Get verification results
        if not experiment.verification_results:
            return "INCONCLUSIVE"

        # Diagnose failure
        diagnosis = diagnose_failure(experiment.verification_results)
        experiment.recovery_history.append({
            "type": "diagnosis",
            "diagnosis": diagnosis,
            "timestamp": datetime.datetime.now().isoformat(),
        })

        # Plan recovery
        plan = suggest_recovery_plan(experiment.verification_results)
        experiment.recovery_history.append({
            "type": "recovery_plan",
            "plan": plan,
            "timestamp": datetime.datetime.now().isoformat(),
        })

        experiment.enter_recovering(plan)
        experiment.save(get_experiment_db())

        # Run recovery cycle
        # This would run a focused agent loop to fix the issues
        # For now, return INCONCLUSIVE to let the main loop handle it
        experiment.recovery_cycles_used += 1
        experiment.recovery_cycles += 1
        experiment.save(get_experiment_db())

        return "CONTINUE"


def run_autonomous_experiment(
    hypothesis: str,
    goal_criteria: List[Dict[str, Any]],
    project_root: str = "",
    max_iterations: int = 30,
) -> ExperimentResult:
    """
    High-level function to run an autonomous experiment.
    This is the main entry point for Phase 7C.
    """
    from brain import _execute_tool, ask_llm_internal
    from tts import speak as _speak_status

    controller = ExperimentController(
        execute_tool_fn=_execute_tool,
        ask_llm_fn=ask_llm_internal,
        speak_fn=_speak_status,
        max_iterations=max_iterations,
    )

    return controller.run_experiment(
        hypothesis=hypothesis,
        goal_criteria=goal_criteria,
        project_root=project_root,
    )


# ─────────────────────────────────────────────
# VISION BENCHMARK ADAPTER
# ─────────────────────────────────────────────

def create_vision_experiment(
    goal: str = "Improve motion detection in artificial retina",
    project_root: str = "/Users/debasishbeura/Jarvis/jarvis_vision_experiment",
) -> tuple[ExperimentResult, List[Dict[str, Any]]]:
    """
    Create an experiment configured for the artificial retina vision benchmark.

    Returns the experiment and its goal criteria.
    """
    goal_criteria = [
        {
            "type": "metric_ratio",
            "target": "motion_metrics.json",
            "metric1": "moving_spike_rate",
            "metric2": "static_spike_rate",
            "metric_file1": os.path.join(project_root, "motion_metrics.json"),
            "metric_file2": os.path.join(project_root, "motion_metrics.json"),
            "operator": ">=",
            "value": 3.0,
            "source": "file",
            "required": True,
        },
        {
            "type": "file_exists",
            "target": os.path.join(project_root, "photoreceptor.py"),
            "required": True,
        },
        {
            "type": "file_exists",
            "target": os.path.join(project_root, "tests/test_photoreceptor.py"),
            "required": True,
        },
        {
            "type": "tests_pass",
            "target": "tests/test_photoreceptor.py",
            "required": True,
        },
        {
            "type": "performance_threshold",
            "target": "python -c \"from motion_detector import run_motion_experiment; run_motion_experiment()\"",
            "command": "cd /Users/debasishbeura/Jarvis/jarvis_vision_experiment && python -c \"from motion_detector import run_motion_experiment; run_motion_experiment()\"",
            "metric": "time",
            "operator": "<=",
            "value": 30.0,
            "unit": "seconds",
            "required": False,
        },
    ]

    task = create_task(
        goal="Improve motion detection in artificial retina - achieve discrimination ratio >= 3.0x",
        project_root=project_root,
    )
    task.set_goal_criteria(goal_criteria)

    experiment = create_experiment(
        task_id=task.task_id,
        hypothesis="Improve motion detection in artificial retina by implementing photoreceptor encoding and temporal contrast detection to achieve motion discrimination ratio >= 3.0x while maintaining latency under 30s",
        goal_criteria=goal_criteria,
    )

    return experiment, goal_criteria


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Run the vision experiment
    experiment, criteria = create_vision_experiment()
    print(f"Created experiment: {experiment.experiment_id}")
    print(f"Goal: {experiment.hypothesis}")
    print(f"Criteria: {len(criteria)} criteria defined")

    # Run the experiment
    result = run_autonomous_experiment(
        hypothesis=experiment.hypothesis,
        goal_criteria=criteria,
        project_root="/Users/debasishbeura/Jarvis/jarvis_vision_experiment",
        max_iterations=30,
    )

    print(f"\nExperiment {result.experiment_id} completed!")
    print(f"Decision: {result.decision}")
    print(f"Reason: {result.decision_reason}")
    print(f"Criteria passed: {result.all_criteria_passed}")
    print(f"Recovery cycles: {result.recovery_cycles}")
