"""Universal action sandbox.

Every dangerous action runs in a sandbox FIRST. The outcome is shown to the
user, then the user decides whether to proceed outside the sandbox.

  - File writes (create_file/write_file/append_file)  → unified diff preview
  - Terminal commands & Python execution              → real sandboxed run, output shown
  - Browser/computer/discord/mail/drive/etc.          → intent preview (what WOULD happen)
  - Self-modification (modify_own_tool, writes to protected core files)
                                                      → special review: static analysis,
                                                        dry run, typed confirmation,
                                                        backup + rollback, registry reload

Toggles:
  JARVIS_ACTION_SANDBOX=0      disable everything here
  JARVIS_EVAL_MODE=1           bypass (CI)
  JARVIS_PROTECTED_FILES       extra protected paths (comma separated)
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import file_sandbox

JARVIS_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(tempfile.gettempdir(), "jarvis_sandbox_backups")
BACKUP_TTL_SECONDS = 24 * 60 * 60
SELF_MOD_AUDIT_PATH = os.path.join(os.path.expanduser("~"), ".jarvis", "self_mod_audit.jsonl")

WRITE_TOOLS = {"create_file", "write_file", "append_file"}
EXEC_TOOLS = {"run_terminal_command", "run_python", "run_python_sandboxed", "run_command_sandboxed"}
SELF_MOD_TOOL = "modify_own_tool"

# ─────────────────────────────────────────────
# REPO-FILE DESTRUCTION GUARD
# ─────────────────────────────────────────────
# The sandbox preview for terminal/python commands is a REAL run (OS sandbox,
# but the real filesystem is visible). A command like `rm brain.py` therefore
# executes during the preview itself — before the user ever approves. Any
# destructive operation targeting the Jarvis repo or protected core files is
# rejected BEFORE a preview runs.
_DESTRUCTIVE_BINS = ("rm", "rmdir", "unlink", "shred", "truncate", "mv")
_DESTRUCTIVE_PAT = re.compile(
    r"(^|[\s;|&()])(" + "|".join(_DESTRUCTIVE_BINS) + r")(\s|$)",
    re.IGNORECASE,
)
_PY_DESTRUCTIVE_PAT = re.compile(
    r"(os\.(remove|unlink|rmdir)\s*\(|shutil\.rmtree\s*\(|\.unlink\s*\(|os\.truncate\s*\()",
    re.IGNORECASE,
)


def _protected_paths() -> list[str]:
    seen = {os.path.abspath(JARVIS_ROOT)}
    for rel in PROTECTED_CORE_FILES:
        seen.add(os.path.normpath(os.path.join(JARVIS_ROOT, rel)))
    extra = os.environ.get("JARVIS_PROTECTED_FILES", "")
    for p in extra.split(","):
        p = p.strip()
        if p and os.path.isabs(p):
            seen.add(os.path.normpath(p))
    return sorted(seen)


def _targets_repo(path_str: str, protected: list[str]) -> bool:
    raw = path_str.strip().strip("\"'")
    if not raw or raw.startswith("-") or any(c in raw for c in "*?["):
        return False
    p = os.path.expanduser(raw)
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(JARVIS_ROOT, p))
    p = os.path.normpath(p)
    if os.path.basename(p) in {os.path.basename(b) for b in PROTECTED_CORE_FILES}:
        return True
    for base in protected:
        try:
            if p == base or os.path.commonpath([p, base]) == base:
                return True
        except ValueError:
            continue
    return False


def destructive_preview_block(cmd: str, is_python: bool = False) -> str | None:
    """Return a block reason if `cmd` destructively targets the repo/protected
    files. None means safe to preview."""
    protected = _protected_paths()
    if is_python:
        if not _PY_DESTRUCTIVE_PAT.search(cmd):
            return None
        for m in re.finditer(r"([\"'])(.*?)\1", cmd):
            if _targets_repo(m.group(2), protected):
                return f"Destructive Python call targets a protected file: {m.group(2)}"
        return None
    for segment in re.split(r"[;|&]", cmd):
        if not _DESTRUCTIVE_PAT.search(segment):
            continue
        tokens = segment.split()
        op = tokens[0].lower()
        if op == "mv":
            args = [t for t in tokens[1:]]
        else:
            args = [t for t in tokens[1:] if not t.startswith("-")]
        for arg in args:
            if _targets_repo(arg, protected):
                return f"{op} targets a protected file: {arg}"
    return None

# Core files that get the special self-modification review.
PROTECTED_CORE_FILES = {
    "brain.py": "MAXIMUM",
    "safety.py": "MAXIMUM",
    "action_sandbox.py": "MAXIMUM",
    "file_sandbox.py": "MAXIMUM",
    "terminal.py": "HIGH",
    "server.py": "HIGH",
    "config.py": "HIGH",
    "agent.py": "HIGH",
    "tools/__init__.py": "HIGH",
    "tools/code_tools.py": "MEDIUM",
    "tools/file_tools.py": "MEDIUM",
    "tools/computer_tools.py": "MEDIUM",
    "tools/browser_tools.py": "MEDIUM",
    "tools/discord_tools.py": "MEDIUM",
    "self_awareness.py": "MEDIUM",
    "learner.py": "MEDIUM",
    "sandbox.py": "MEDIUM",
}

# Critical symbols per protected file — if present before, must remain after.
CRITICAL_SYMBOLS = {
    "brain.py": ["def process(", "def _execute_tool(", "def ask_with_tools(", "def classify_intent("],
    "safety.py": ["def check_permission(", "TOOL_PERMISSIONS"],
    "action_sandbox.py": ["def _execute_tool"],
    "terminal.py": ["def main("],
    "server.py": ["FastAPI("],
    "agent.py": ["def run_agent_loop("],
    "tools/__init__.py": ["TOOL_REGISTRY", "TOOL_DEFINITIONS"],
    "config.py": ["GEMINI_DAILY_LIMIT"],
}

_typed_pending: dict = {}
_typed_lock = threading.Lock()

MAX_SIZE_DELTA = 0.5  # self-mod refused if file changes by more than 50%


def enabled() -> bool:
    return os.getenv("JARVIS_ACTION_SANDBOX", "1").lower() in ("1", "true", "yes", "on")


def eval_mode() -> bool:
    return os.getenv("JARVIS_EVAL_MODE", "0").lower() in ("1", "true", "yes", "on")


def _env_extra_protected() -> list:
    raw = os.getenv("JARVIS_PROTECTED_FILES", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _realpath(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def protected_level_for(path: str) -> str | None:
    """Return MAXIMUM/HIGH/MEDIUM for protected files, else None."""
    full = _realpath(path)
    root = os.path.realpath(JARVIS_ROOT)
    try:
        rel = os.path.relpath(full, root)
    except Exception:
        rel = os.path.basename(full)
    if not rel.startswith(".."):
        if rel in PROTECTED_CORE_FILES:
            return PROTECTED_CORE_FILES[rel]
        for extra in _env_extra_protected():
            if full == _realpath(extra):
                return "HIGH"
    return None


def is_selfmod(fn_name: str, args: dict) -> bool:
    if fn_name == SELF_MOD_TOOL:
        return True
    if fn_name in WRITE_TOOLS:
        return protected_level_for(_target_path(fn_name, args)) is not None
    return False


def _target_path(fn_name: str, args: dict) -> str:
    args = args or {}
    if fn_name == SELF_MOD_TOOL:
        target = args.get("tool_file", "")
        cand = os.path.join(JARVIS_ROOT, target)
        if os.path.exists(cand):
            return cand
        return os.path.join(JARVIS_ROOT, "tools", target)
    if fn_name == "create_file":
        folder = args.get("path") or os.path.join(os.path.expanduser("~"), "Desktop")
        return os.path.join(os.path.expanduser(folder), args.get("filename", ""))
    return args.get("path", "")


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


# ─────────────────────────────────────────────
# PREVIEW EXECUTION (real sandboxed runs)
# ─────────────────────────────────────────────


def preview_terminal(command: str, timeout: int = 30, allow_network: bool = False) -> dict:
    from sandbox import run_sandboxed_command

    return run_sandboxed_command(
        command, timeout=timeout, allow_network=allow_network
    )


def preview_python(code: str, timeout: int = 30) -> dict:
    from sandbox import run_sandboxed_python

    return run_sandboxed_python(code, timeout=timeout)


def run_terminal_for_real(command: str, timeout: int = 30) -> str:
    from safety import analyze_command

    allowed, _level, reason = analyze_command(command)
    if not allowed:
        return f"Command blocked: {reason}"
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        parts = [f"Exit code: {proc.returncode}"]
        if proc.stdout.strip():
            parts.append(f"STDOUT:\n{proc.stdout.strip()[:1000]}")
        if proc.stderr.strip():
            parts.append(f"STDERR:\n{proc.stderr.strip()[:1000]}")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    except Exception as e:
        return f"Command failed: {e}"


def run_python_for_real(code: str, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
        )
        parts = [f"Exit code: {proc.returncode}"]
        if proc.stdout.strip():
            parts.append(f"STDOUT:\n{proc.stdout.strip()[:1000]}")
        if proc.stderr.strip():
            parts.append(f"STDERR:\n{proc.stderr.strip()[:1000]}")
        return "\n\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Python timed out after {timeout}s."
    except Exception as e:
        return f"Python failed: {e}"


def real_executor(fn_name: str):
    if fn_name == "run_python":
        return run_python_for_real
    if fn_name == "run_terminal_command":
        return run_terminal_for_real
    return None


def format_exec_preview(fn_name: str, args: dict, result: dict) -> str:
    sandboxed = result.get("sandboxed", False)
    sb_note = " [OS sandboxed]" if sandboxed else " [limited]"
    parts = [f"Command: {args.get('command', args.get('code', ''))[:200]}"]
    parts.append(f"Exit code: {result.get('exit_code', -1)}")
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    if stdout:
        parts.append(f"STDOUT:\n{stdout[:1000]}")
    if stderr:
        parts.append(f"STDERR:\n{stderr[:1000]}")
    if result.get("error"):
        parts.append(f"Error: {result['error']}")
    parts.append(f"Duration: {result.get('duration_ms', 0)}ms{sb_note}")
    return "\n".join(parts)


def run_exec_preview(fn_name: str, args: dict) -> str:
    if fn_name in ("run_terminal_command", "run_command_sandboxed"):
        from safety import analyze_command

        _block = destructive_preview_block(args.get("command", ""))
        if _block:
            return f"[BLOCKED] Command not previewed: {_block}"
        allowed, _level, reason = analyze_command(args.get("command", ""))
        if not allowed:
            return f"[BLOCKED] Command not previewed: {reason}"
        result = preview_terminal(
            args.get("command", ""),
            timeout=int(args.get("timeout", 30) or 30),
            allow_network=bool(args.get("allow_network", False)),
        )
    else:
        _block = destructive_preview_block(args.get("code", ""), is_python=True)
        if _block:
            return f"[BLOCKED] Python not previewed: {_block}"
        result = preview_python(args.get("code", ""), timeout=int(args.get("timeout", 30) or 30))
    return format_exec_preview(fn_name, args, result)


# ─────────────────────────────────────────────
# INTENT PREVIEWS (what WOULD happen)
# ─────────────────────────────────────────────


def format_intent_preview(fn_name: str, args: dict) -> str:
    args = args or {}
    if fn_name == "browser_navigate":
        return f"Navigate to: {args.get('url')} (browser: {args.get('browser', 'default')})"
    if fn_name == "browser_quick_search":
        return f"Search in browser: {args.get('query')}"
    if fn_name == "browser_new_tab":
        return f"Open new browser tab ({args.get('browser', 'default')})"
    if fn_name == "browser_close_tab":
        return "Close current browser tab"
    if fn_name == "browser_scroll":
        return f"Scroll browser: {args.get('direction', 'down')}"
    if fn_name == "browser_back":
        return "Go back in browser history"
    if fn_name == "browser_forward":
        return "Go forward in browser history"
    if fn_name == "browser_reload":
        return "Reload current browser page"
    if fn_name == "browser_current_url":
        return "Read current browser URL"
    if fn_name == "click_on_screen":
        return f"Click element: {args.get('what')}"
    if fn_name == "move_and_click":
        return f"Click at coordinates: ({args.get('x')}, {args.get('y')})"
    if fn_name == "type_text":
        submit = " then press Enter" if args.get("submit") else ""
        return f"Type: '{args.get('text')}'{submit}"
    if fn_name == "press_key":
        return f"Press key: {args.get('key')}"
    if fn_name == "quit_app":
        return f"Quit app: {args.get('app_name')}"
    if fn_name == "organize_downloads":
        return "Move files in ~/Downloads into subfolders by type"
    if fn_name == "discord_open_channel":
        return f"Open Discord channel: {args.get('channel_name')}"
    if fn_name == "discord_send_message":
        return f"Send Discord message: {args.get('text')}"
    if fn_name == "discord_open_and_send":
        return f"Discord → channel {args.get('channel_name')}: '{args.get('message')}'"
    if fn_name == "send_imessage":
        return f"iMessage to {args.get('contact')}: '{args.get('message')}'"
    if fn_name == "gmail_send":
        return f"Gmail to {args.get('to')} | subject: {args.get('subject')} | body: {args.get('body', '')[:200]}"
    if fn_name == "github_create_issue":
        return f"GitHub issue {args.get('owner')}/{args.get('repo')} | title: {args.get('title')}"
    if fn_name == "gdrive_upload":
        return f"Upload {args.get('local_path')} to Drive folder {args.get('folder_id', 'root')}"
    if fn_name == "gdrive_create_folder":
        return f"Create Drive folder: {args.get('name')}"
    if fn_name == "gdrive_share":
        return f"Share Drive file {args.get('file_id')} with {args.get('email')} ({args.get('role', 'reader')})"
    if fn_name == "gdrive_move":
        return f"Move Drive file {args.get('file_id')} → folder {args.get('new_parent_id')}"
    if fn_name == "gdrive_delete":
        return f"DELETE Drive file: {args.get('file_id')}"
    if fn_name == "gsheets_append":
        return (
            f"Append to sheet {args.get('spreadsheet_id')} "
            f"[{args.get('range_name', '')}]: {str(args.get('values'))[:200]}"
        )
    if fn_name == "gsheets_update_range":
        return f"Update sheet {args.get('spreadsheet_id')} [{args.get('range_name', '')}]"
    if fn_name == "gsheets_batch_update":
        return f"Batch update sheet {args.get('spreadsheet_id')}: {str(args.get('requests_list'))[:200]}"
    if fn_name == "gsheets_create":
        return f"Create spreadsheet: {args.get('title')}"
    if fn_name == "gsheets_add_sheet":
        return f"Add sheet '{args.get('title')}' to {args.get('spreadsheet_id')}"
    if fn_name == "docs_create":
        return f"Create Google Doc: {args.get('title')}"
    if fn_name == "docs_append_text":
        return f"Append to Doc {args.get('document_id')}: '{args.get('text', '')[:200]}'"
    if fn_name == "slides_create":
        return f"Create presentation: {args.get('title')}"
    if fn_name == "slides_add_slide":
        return f"Add slide to presentation {args.get('presentation_id')}"
    if fn_name == "slides_replace_text":
        return (
            f"Replace '{args.get('old_text')}' with '{args.get('new_text')}' "
            f"in presentation {args.get('presentation_id')}"
        )
    if fn_name == "agent_spawn":
        return f"Spawn sub-agent: {args.get('goal')}"
    if fn_name == "add_goal":
        return f"Add goal: {args.get('title')}"
    if fn_name == "add_goal_progress":
        return f"Add progress to goal: {args.get('goal_id')}"
    if fn_name == "update_goal_status":
        return f"Update goal {args.get('goal_id')} status"
    if fn_name == "set_goal_next_check":
        return f"Set next check for goal {args.get('goal_id')}"
    if fn_name == "add_to_knowledge_graph":
        return f"Knowledge graph: {str(args)[:200]}"
    if fn_name == "backup_file":
        return f"Backup file: {args.get('path')}"
    if fn_name == "save_api_key":
        return f"Save API key: {args.get('name')}"
    return f"{fn_name}({str(args)[:200]})"


INTENT_PREVIEW_TOOLS = {
    "browser_navigate", "browser_quick_search", "browser_new_tab", "browser_close_tab",
    "browser_scroll", "browser_back", "browser_forward", "browser_reload",
    "browser_current_url", "click_on_screen", "move_and_click", "type_text",
    "press_key", "quit_app", "organize_downloads", "discord_open_channel",
    "discord_send_message", "discord_open_and_send", "send_imessage", "gmail_send",
    "github_create_issue", "gdrive_upload", "gdrive_create_folder", "gdrive_share",
    "gdrive_move", "gdrive_delete", "gsheets_append", "gsheets_update_range",
    "gsheets_batch_update", "gsheets_create", "gsheets_add_sheet", "docs_create",
    "docs_append_text", "slides_create", "slides_add_slide", "slides_replace_text",
    "agent_spawn", "add_goal", "add_goal_progress", "update_goal_status",
    "set_goal_next_check", "add_to_knowledge_graph", "backup_file", "save_api_key",
}


def is_sandboxable(fn_name: str) -> bool:
    return (
        fn_name in WRITE_TOOLS
        or fn_name in EXEC_TOOLS
        or fn_name in INTENT_PREVIEW_TOOLS
        or fn_name == SELF_MOD_TOOL
    )


def confirm_message(fn_name: str, args: dict) -> str:
    if fn_name in EXEC_TOOLS:
        return (
            f"Preview of {fn_name.replace('_', ' ')} shown above. "
            "Say yes to run it for real OUTSIDE the sandbox, or no to cancel."
        )
    if fn_name in WRITE_TOOLS:
        target = _target_path(fn_name, args)
        return f"Write to {target} shown above — say yes to apply, no to cancel."
    return "Preview shown above. Say yes to proceed for real, or no to cancel."


# ─────────────────────────────────────────────
# AUTO-PROCEED (Option B: preview then proceed after 3s)
# ─────────────────────────────────────────────


def auto_proceed(timeout: float = 3.0) -> bool:
    """Show preview already printed — auto-proceed after 3s unless user hits Enter."""
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return True
        import select

        print(f"\nProceeding in {int(timeout)}s — press Enter to cancel...", end="", flush=True)
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            sys.stdin.readline()
            print(" cancelled.")
            return False
        print(" proceeding.")
        return True
    except Exception:
        return True


# ─────────────────────────────────────────────
# SELF-MODIFICATION REVIEW
# ─────────────────────────────────────────────


def analyze_selfmod(target: str, old_code: str, new_code: str) -> dict:
    rel = os.path.relpath(_realpath(target), os.path.realpath(JARVIS_ROOT))
    checks = []
    blocked = []
    level = protected_level_for(target) or "MEDIUM"

    # 1. Syntax
    try:
        ast.parse(new_code)
        checks.append("Syntax valid")
    except SyntaxError as e:
        blocked.append(f"Syntax error on line {e.lineno}: {e.msg}")

    # 2. Size delta — refuse anything that would gut a file
    old_len = len(old_code)
    if old_len:
        delta = abs(len(new_code) - old_len) / old_len
        if delta > MAX_SIZE_DELTA:
            blocked.append(f"Size delta {delta:.0%} exceeds {MAX_SIZE_DELTA:.0%} — split the change into smaller steps")

    # 3. Critical symbols must survive
    missing = []
    for sym in CRITICAL_SYMBOLS.get(rel, []):
        if sym in old_code and sym not in new_code:
            missing.append(sym)
    if missing:
        blocked.append(f"Critical code removed: {', '.join(missing)}")

    # 4. New imports
    try:
        old_imports = {n.name for n in ast.walk(ast.parse(old_code)) if isinstance(n, ast.Import) for a in n.names}
        new_imports = {n.name for n in ast.walk(ast.parse(new_code)) if isinstance(n, ast.Import) for a in n.names}
        added = new_imports - old_imports
        if added:
            checks.append(f"New imports: {', '.join(sorted(added))}")
    except Exception:
        pass

    return {"level": level, "checks": checks, "blocked": blocked}


def dry_run_code(code: str) -> dict:
    """Always-on dry run: syntax + isolated import test."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "error": f"Syntax error: {e.msg} on line {e.lineno}"}
    if not code.strip().startswith("import ") and not code.strip().startswith("from ") and "def " not in code[:200]:
        # Not a module-shaped file (e.g. plain config text) — syntax-only
        return {"ok": True, "note": "syntax only"}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util,sys;"
                f"spec=importlib.util.spec_from_file_location('dryrun','{tmp}');"
                "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            return {"ok": True, "note": "import test passed"}
        err = (proc.stderr or "").strip().splitlines()
        return {"ok": False, "error": "; ".join(err[-3:]) or f"exit {proc.returncode}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "import test timed out"}
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def backup_file(path: str) -> str:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{os.path.basename(path)}.bak"
    dest = os.path.join(BACKUP_DIR, name)
    try:
        shutil.copy2(path, dest)
    except Exception:
        pass
    _prune_backups()
    return dest


def restore_backup(backup_path: str, target: str) -> bool:
    try:
        shutil.copy2(backup_path, target)
        return True
    except Exception:
        return False


def _prune_backups():
    try:
        cutoff = time.time() - BACKUP_TTL_SECONDS
        for f in os.listdir(BACKUP_DIR):
            p = os.path.join(BACKUP_DIR, f)
            try:
                if os.path.getmtime(p) < cutoff:
                    os.unlink(p)
            except Exception:
                pass
    except Exception:
        pass


def apply_selfmod(target: str, new_code: str) -> str:
    """Backup → write → post-apply dry run → rollback on failure → reload registry."""
    backup = backup_file(target)
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(new_code)
    except Exception as e:
        _log_self_mod_audit(target, backup, f"write_failed: {e}")
        return f"Write failed: {e}"
    dry = dry_run_code(new_code)
    if not dry["ok"]:
        restore_backup(backup, target)
        _log_self_mod_audit(target, backup, f"rejected_after_apply_rolled_back: {dry['error']}")
        return f"Change rejected after apply — rolled back to backup. Validation: {dry['error']}"
    reload_if_tool_file(target)
    _log_self_mod_audit(target, backup, "applied")
    return f"Applied to {target}. Backup: {backup}"


def _log_self_mod_audit(target: str, backup: str, outcome: str):
    """Dedicated self-modification audit trail (what JARVIS changed about itself, when, why)."""
    import datetime
    import json

    rel = os.path.relpath(_realpath(target), os.path.realpath(JARVIS_ROOT))
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "target": rel,
        "backup": backup,
        "outcome": outcome,
    }
    try:
        os.makedirs(os.path.dirname(SELF_MOD_AUDIT_PATH), exist_ok=True)
        with open(SELF_MOD_AUDIT_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def get_self_mod_audit(limit: int = 20) -> list[dict]:
    """Read the last `limit` self-modification audit entries."""
    import json

    if not os.path.exists(SELF_MOD_AUDIT_PATH):
        return []
    out = []
    try:
        with open(SELF_MOD_AUDIT_PATH) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out[-limit:]


def reload_if_tool_file(path: str):
    full = _realpath(path)
    root = os.path.realpath(JARVIS_ROOT)
    try:
        rel = os.path.relpath(full, root)
    except Exception:
        return
    if rel.startswith(".."):
        return
    if rel.startswith("tools/") and rel.endswith(".py"):
        _reload_tool_registry()


def _reload_tool_registry():
    """Reload tool modules and rebuild TOOL_REGISTRY / TOOL_DEFINITIONS in place."""
    try:
        import importlib
        import sys as _sys

        modules = [name for name in list(_sys.modules) if name.startswith("tools.") and name.count(".") == 1]
        for name in sorted(modules):
            try:
                importlib.reload(_sys.modules[name])
            except Exception:
                pass
        import tools

        reg = tools.TOOL_REGISTRY
        reg.clear()
        for name in sorted(modules):
            mod = _sys.modules.get(name)
            if not mod:
                continue
            tool_map = getattr(mod, "TOOLS", None)
            if isinstance(tool_map, dict):
                reg.update(tool_map)
        defs = list(tools.TOOL_DEFINITIONS)
        tools.TOOL_DEFINITIONS[:] = defs
    except Exception:
        pass


# ─────────────────────────────────────────────
# TYPED PENDING (self-mod confirmation)
# ─────────────────────────────────────────────


def stage_typed(fn_name: str, fn, args: dict, confirmation_text: str, preview: str, data: dict):
    with _typed_lock:
        _typed_pending.update(
            {
                "tool": fn_name,
                "fn": fn,
                "args": dict(args or {}),
                "confirmation_text": confirmation_text.lower(),
                "preview": preview,
                "data": data,
                "expires_at": time.time() + file_sandbox.PENDING_TTL,
            }
        )


def has_typed_pending() -> bool:
    with _typed_lock:
        if not _typed_pending:
            return False
        if time.time() > _typed_pending.get("expires_at", 0):
            _typed_pending.clear()
            return False
        return True


def get_typed_pending() -> dict | None:
    with _typed_lock:
        if not _typed_pending:
            return None
        if time.time() > _typed_pending.get("expires_at", 0):
            return None
        return dict(_typed_pending)


def clear_typed_pending():
    with _typed_lock:
        _typed_pending.clear()


def confirm_text_for(target: str) -> str:
    return f"confirm self-modify {os.path.basename(target)}"


def format_selfmod_preview(target: str, analysis: dict, dry: dict | None, diff: str) -> str:
    rel = os.path.relpath(_realpath(target), os.path.realpath(JARVIS_ROOT))
    lines = [
        "[SELF-MOD REVIEW]",
        f"Target: {rel} (protection: {analysis['level']})",
        "Static analysis:",
    ]
    for c in analysis["checks"]:
        lines.append(f"  [OK] {c}")
    for b in analysis["blocked"]:
        lines.append(f"  [BLOCKED] {b}")
    if analysis["blocked"]:
        lines.append("RESULT: REFUSED")
        return "\n".join(lines)
    if dry:
        note = dry.get("note", "")
        lines.append(f"Dry run: {'PASS' if dry['ok'] else 'FAIL'}" + (f" ({note})" if note else ""))
        if not dry["ok"]:
            lines.append(f"  Dry run error: {dry['error']}")
            lines.append("RESULT: REFUSED — dry run failed")
            return "\n".join(lines)
    if diff:
        lines.append("\nDiff:")
        lines.append(diff)
    return "\n".join(lines)


def resolve_selfmod(fn_name: str, args: dict, fn) -> dict:
    """Build the self-mod payload: target, old code, new code, change description."""
    target = _target_path(fn_name, args)
    old_code = _read(target)
    if fn_name == SELF_MOD_TOOL:
        new_code = args.get("new_code", "")
        change_desc = args.get("change_description", "")
    elif fn_name == "append_file":
        new_code = old_code + args.get("content", "")
        change_desc = f"append to {os.path.basename(target)}"
    else:
        new_code = args.get("content", "")
        change_desc = f"write to {os.path.basename(target)}"
    return {
        "tool": fn_name,
        "fn": fn,
        "args": dict(args or {}),
        "target": target,
        "old_code": old_code,
        "new_code": new_code,
        "change_desc": change_desc,
    }


def apply_staged_selfmod() -> str:
    pending = get_typed_pending()
    if not pending or not pending.get("data"):
        return "No pending self-modification to apply."
    data = pending["data"]
    return apply_selfmod(data["target"], data["new_code"])


def selfmod_apply_fn() -> callable:
    return apply_staged_selfmod
