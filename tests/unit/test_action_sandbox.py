import os

import pytest

import action_sandbox as ax


@pytest.fixture(autouse=True)
def _cleanup():
    ax.clear_typed_pending()
    yield
    ax.clear_typed_pending()


def test_enabled_by_default():
    assert ax.enabled()
    assert not ax.eval_mode()


def test_protected_levels():
    assert ax.protected_level_for("brain.py") == "MAXIMUM"
    assert ax.protected_level_for("safety.py") == "MAXIMUM"
    assert ax.protected_level_for("action_sandbox.py") == "MAXIMUM"
    assert ax.protected_level_for("server.py") == "HIGH"
    assert ax.protected_level_for("agent.py") == "HIGH"
    assert ax.protected_level_for("tools/code_tools.py") == "MEDIUM"
    assert ax.protected_level_for("sandbox.py") == "MEDIUM"
    assert ax.protected_level_for("notes.md") is None


def test_is_selfmod_routing():
    assert ax.is_selfmod("modify_own_tool", {"tool_file": "brain.py"})
    assert ax.is_selfmod("write_file", {"path": "brain.py"})
    assert not ax.is_selfmod("write_file", {"path": "/tmp/whatever.txt"})
    assert not ax.is_selfmod("run_python", {"code": "x=1"})


def test_target_path():
    assert ax._target_path("modify_own_tool", {"tool_file": "brain.py"}) == os.path.join(ax.JARVIS_ROOT, "brain.py")
    tools_path = os.path.join(ax.JARVIS_ROOT, "tools", "code_tools.py")
    assert ax._target_path("modify_own_tool", {"tool_file": "code_tools.py"}) == tools_path


def test_analyze_selfmod_blocks_gutted_brain():
    old = open(os.path.join(ax.JARVIS_ROOT, "brain.py")).read()
    res = ax.analyze_selfmod("brain.py", old, "print('gutted')")
    assert any("Size delta" in b for b in res["blocked"])
    assert any("Critical code removed" in b for b in res["blocked"])
    assert res["level"] == "MAXIMUM"


def test_analyze_selfmod_passes_benign(tmp_path):
    base = "x = 1\n" * 100
    p = tmp_path / "mod.py"
    p.write_text(base)
    res = ax.analyze_selfmod(str(p), base, base + "# comment\n")
    assert not res["blocked"]


def test_analyze_selfmod_syntax_error(tmp_path):
    base = "x = 1\n" * 100
    res = ax.analyze_selfmod("brain.py", base, "def broken(:\n")
    assert any("Syntax error" in b for b in res["blocked"])


def test_dry_run_good_module():
    assert ax.dry_run_code("def f():\n    return 1\n")["ok"]


def test_dry_run_syntax_error():
    assert not ax.dry_run_code("def f():\nreturn 1\n")["ok"]


def test_dry_run_bad_import():
    assert not ax.dry_run_code("import module_that_does_not_exist_xyz")["ok"]


def test_dry_run_plain_config_passes_syntax_only(tmp_path):
    res = ax.dry_run_code("KEY = 'value'\n# just some config text\n")
    assert res["ok"]
    assert res["note"] == "syntax only"


def test_typed_pending_flow(tmp_path):
    assert not ax.has_typed_pending()
    target = str(tmp_path / "x.py")
    ax.stage_typed(
        "modify_own_tool", "fn", {"tool_file": "x.py"}, "confirm self-modify x.py", "PREVIEW",
        {"target": target, "new_code": "y=2"},
    )
    assert ax.has_typed_pending()
    pending = ax.get_typed_pending()
    assert pending["confirmation_text"] == "confirm self-modify x.py"
    ax.clear_typed_pending()
    assert not ax.has_typed_pending()


def test_confirm_text_for(tmp_path):
    assert ax.confirm_text_for("brain.py") == "confirm self-modify brain.py"


def test_apply_selfmod_success_and_rollback(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("x = 1\n" * 50)
    out = ax.apply_selfmod(str(p), ("x = 1\n" * 50) + "# ok\n")
    assert "Applied to" in out
    assert p.read_text().endswith("# ok\n")
    out = ax.apply_selfmod(str(p), "def broken(:\n")
    assert "rolled back" in out
    assert p.read_text().endswith("# ok\n")


def test_backup_restore(tmp_path):
    p = tmp_path / "b.py"
    p.write_text("original")
    backup = ax.backup_file(str(p))
    assert os.path.exists(backup)
    p.write_text("changed")
    assert ax.restore_backup(backup, str(p))
    assert p.read_text() == "original"


def test_exec_preview_runs_sandboxed_python():
    out = ax.run_exec_preview("run_python", {"code": "print(2+2)"})
    assert "4" in out
    assert "sandboxed" in out.lower()


def test_exec_preview_terminal():
    out = ax.run_exec_preview("run_terminal_command", {"command": "echo hello"})
    assert "hello" in out


def test_destructive_preview_block_repo_file():
    assert "BLOCKED" in ax.run_exec_preview("run_terminal_command", {"command": "rm brain.py"})
    assert "BLOCKED" in ax.run_exec_preview("run_terminal_command", {"command": "rm --force server.py"})
    assert "BLOCKED" in ax.run_exec_preview("run_terminal_command", {"command": "cd tools && rm -f code_tools.py"})
    assert "BLOCKED" in ax.run_exec_preview(
        "run_terminal_command", {"command": "ls /tmp; mv brain.py /tmp/deleted.py"}
    )
    assert "BLOCKED" in ax.run_exec_preview(
        "run_terminal_command", {"command": f"unlink {ax.JARVIS_ROOT}/safety.py"}
    )


def test_destructive_preview_block_outside_scope():
    assert "rm" not in ax.run_exec_preview(
        "run_terminal_command", {"command": "rm /tmp/jarvis_probe_test_ojo7.txt"}
    ).split("Command: ")[0]
    assert "4" in ax.run_exec_preview("run_python", {"code": "print(2+2)"})


def test_destructive_python_block_repo_file():
    assert "BLOCKED" in ax.run_exec_preview(
        "run_python", {"code": "import os; os.remove('brain.py')"}
    )
    assert "BLOCKED" in ax.run_exec_preview(
        "run_python", {"code": "import shutil; shutil.rmtree('tools')"}
    )
    assert "BLOCKED" in ax.run_exec_preview(
        "run_python", {"code": f"from pathlib import Path; Path('{ax.JARVIS_ROOT}/server.py').unlink()"}
    )
    assert "4" in ax.run_exec_preview("run_python", {"code": "print(2+2); import os; os.remove('/tmp/nope_j7.txt')"})


def test_intent_previews():
    assert "Navigate to: https://x" in ax.format_intent_preview("browser_navigate", {"url": "https://x"})
    assert "iMessage to bob" in ax.format_intent_preview("send_imessage", {"contact": "bob", "message": "hi"})
    assert "DELETE Drive file: abc" in ax.format_intent_preview("gdrive_delete", {"file_id": "abc"})
    assert ax.format_intent_preview("zzz_unknown", {"a": 1}) == "zzz_unknown({'a': 1})"


def test_confirm_messages():
    assert "run it for real" in ax.confirm_message("run_python", {"code": "x=1"})
    assert "apply" in ax.confirm_message("write_file", {"path": "/tmp/x", "content": "y"})
    assert "proceed" in ax.confirm_message("browser_navigate", {"url": "https://x"})


def test_auto_proceed_non_tty():
    assert ax.auto_proceed(0.05) is True


def test_selfmod_preview_format(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n" * 100)
    analysis = ax.analyze_selfmod(str(p), p.read_text(), p.read_text() + "# z\n")
    dry = {"ok": True, "note": "import test passed"}
    preview = ax.format_selfmod_preview(str(p), analysis, dry, "-line\n+line")
    assert "SELF-MOD REVIEW" in preview
    assert "Diff:" in preview
    refused = ax.format_selfmod_preview("brain.py", {"level": "MAXIMUM", "checks": [], "blocked": ["nope"]}, None, "")
    assert "REFUSED" in refused
