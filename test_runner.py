"""Run JARVIS tests by piping commands and checking output for expected patterns."""

import os
import re
import subprocess
import sys

# Must be at module level for test names
tests_run = 0
tests_passed = 0
all_output = ""


def run_test(name, commands, expected_patterns, unexpected_patterns=None, timeout=120):
    global tests_run, tests_passed, all_output
    tests_run += 1
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")

    env = os.environ.copy()
    env["JARVIS_USE_SCHEDULER"] = "1"
    env["JARVIS_FG_WORKERS"] = "1"
    env["JARVIS_BG_WORKERS"] = "1"
    env["JARVIS_DEBUG"] = "1"
    env["TERM"] = "xterm"

    input_str = "m\n" + "\n".join(commands) + "\nquit\n"

    try:
        proc = subprocess.Popen(
            [sys.executable, "terminal.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/Users/debasishbeura/Jarvis",
            text=True,
        )
        stdout, _ = proc.communicate(input=input_str, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        print(f"  ⚠  TIMEOUT after {timeout}s")
    except Exception as e:
        print(f"  ⚠  ERROR: {e}")
        stdout = ""

    all_output += f"\n--- {name} ---\n{stdout}\n"

    # Analyze output
    passed = True
    for pat in expected_patterns:
        if not re.search(pat, stdout, re.IGNORECASE):
            print(f"  ✗ MISSING: expected pattern '{pat}'")
            passed = False
        else:
            print(f"  ✓ found: '{pat}'")

    if unexpected_patterns:
        for pat in unexpected_patterns:
            if re.search(pat, stdout, re.IGNORECASE):
                print(f"  ✗ UNEXPECTED: found '{pat}'")
                passed = False
            else:
                print(f"  ✓ absent: '{pat}'")

    if passed:
        tests_passed += 1
        print("  ✅ PASSED")
    else:
        print("  ❌ FAILED")
        # Print last 50 lines for debugging
        lines = stdout.strip().splitlines()
        print(f"  (last {min(30, len(lines))} lines of output:)")
        for l in lines[-30:]:
            print(f"    {l}")

    return passed


if __name__ == "__main__":
    print("JARVIS Integration Test Runner")
    print("==============================\n")

    # ── Test 1: Nemotron naked dicts ──
    run_test(
        "Nemotron naked dict — open github",
        ["open https://github.com"],
        expected_patterns=[
            r"opened",
            r"github",
        ],
        unexpected_patterns=[
            r'{"url"',  # should NOT return raw JSON
        ],
    )

    run_test(
        "Nemotron naked dict — open vscode",
        ["open vscode"],
        expected_patterns=[
            r"opened",
            r"visual studio code",
        ],
        unexpected_patterns=[
            r'{"app_name"',
        ],
    )

    run_test(
        "Nemotron naked dict — quit safari",
        ["quit safari"],
        expected_patterns=[
            r"quit",
        ],
        unexpected_patterns=[
            r'{"app_name"',
        ],
    )

    # ── Test 2: Scheduler ordering ──
    run_test(
        "Scheduler ordering — rapid requests",
        ["weather", "what's my system usage", "check disk space"],
        expected_patterns=[
            r"#\d+ FOREGROUND",
        ],
    )

    # ── Test 3: Paste mode submits one message whole ──
    # The multi-line paste must reach the scheduler as ONE request whose echo
    # contains the full text (blank lines kept) — never as N fragments.
    run_test(
        "Paste mode — one message per paste",
        [
            "/mode p",
            "line one",
            "line two",
            "",
            "line four",
            "/",
        ],
        expected_patterns=[
            r"#\d+ FOREGROUND",
            r"You → line one\nline two\n\nline four",
        ],
    )

    # ── Test 4: Web search ──
    run_test(
        "Web search with query",
        ["search web for python 3.13 release date"],
        expected_patterns=[
            r"python",
            r"3\.13",
        ],
        unexpected_patterns=[
            r"missing.*query",
            r"empty",
        ],
    )

    # ── Test 5: Memory forget ──
    run_test(
        "Memory forget — remember/forget/recall",
        [
            "remember my favorite color is blue",
            "forget my favorite color",
            "whats my favorite color",
        ],
        expected_patterns=[
            r"(don't know|haven't told|don.t remember|not found|nothing)",
        ],
    )

    # ── Test 6: Weather ──
    run_test(
        "Weather query",
        ["whats the weather today"],
        expected_patterns=[
            r"temperature",
            r"°[fF]",
        ],
    )

    # ── Test 7: System usage ──
    run_test(
        "System usage",
        ["what's my system usage"],
        expected_patterns=[
            r"cpu",
            r"ram",
            r"disk",
        ],
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {tests_passed}/{tests_run} passed")
    print(f"{'=' * 60}")

    # Save full output for review
    with open("/tmp/jarvis_test_output.log", "w") as f:
        f.write(all_output)

    sys.exit(0 if tests_passed == tests_run else 1)
