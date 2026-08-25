"""Pytest wrapper for the Phase 5 regression gate.

Runs the unified regression gate in mock mode (golden + gates).
"""

import pytest
import subprocess
import sys
import os


@pytest.mark.integration
def test_regression_gate_mock():
    """Run the regression gate in mock mode (golden + gates)."""
    env = os.environ.copy()
    env["JARVIS_EVAL_MODE"] = "1"
    env["JARVIS_TTS_SILENT"] = "1"
    env["JARVIS_MOCK_PROVIDERS"] = "1"
    env["MOCK_PROVIDER_URL"] = "http://127.0.0.1:18889"
    env["MOCK_PROVIDER_PORT"] = "18889"

    # Start mock provider
    mock_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.mock_provider:app",
         "--host", "127.0.0.1", "--port", "18889", "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        import requests
        for _ in range(30):
            try:
                r = requests.get("http://127.0.0.1:18889/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            import time
            time.sleep(1)
        else:
            pytest.fail("Mock provider failed to start")

        # Run the regression gate
        result = subprocess.run(
            [sys.executable, "benchmarks/regression_gate.py", "--mode", "mock",
             "--only", "golden", "--only", "gates", "--no-api"],
            capture_output=True, text=True, timeout=300, env=env, cwd=os.getcwd()
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        assert result.returncode == 0, f"Regression gate failed: {result.stdout}\n{result.stderr}"
    finally:
        mock_proc.terminate()
        try:
            mock_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mock_proc.kill()