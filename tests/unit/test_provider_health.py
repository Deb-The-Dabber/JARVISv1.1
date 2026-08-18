import os

# Provider-health unit tests — no network, no mock server.
os.environ["JARVIS_LOCAL_INTENT_ENABLED"] = "0"
os.environ["JARVIS_LLM_FIRST"] = "0"

import pytest

import brain
from brain import _classify_error, _execute_tool, _handle_provider_failure, _is_malformed_tool_name


def _fresh_state(monkeypatch, tmp_path):
    monkeypatch.setattr(brain, "_PROVIDER_HEALTH_FILE", str(tmp_path / "health.json"))
    monkeypatch.setattr(brain, "_provider_backoff_until", {})
    monkeypatch.setattr(brain, "_provider_consecutive_failures", {})
    monkeypatch.setattr(brain, "_provider_health_scores", {})
    monkeypatch.setattr(brain, "_provider_health", {})


class TestErrorClassification:
    def test_resolution_label(self):
        assert _classify_error("NIM Fast tool execution did not resolve") == "resolution"
        assert _classify_error("tool execution did not resolve (time budget exceeded)") == "resolution"
        assert (
            _classify_error("tool execution did not resolve (malformed tool call 'x<|y')")
            == "resolution"
        )

    def test_known_labels_unchanged(self):
        assert _classify_error("429 Too Many Requests") == "rate_limit"
        assert _classify_error("ReadTimeout") == "timeout"
        assert _classify_error("503 Service Unavailable") == "internal"
        assert _classify_error("401 Unauthorized") == "auth_error"
        assert _classify_error("exceeded your current quota") == "quota"


class TestResolutionNoBackoff:
    def test_resolution_failure_does_not_trip_circuit(self, monkeypatch, tmp_path):
        # Regression: one "did not resolve" (tool loop spent its rounds) used
        # to take the provider out of rotation for up to an hour, so a single
        # bad request cascaded the ENTIRE fallback chain into backoff.
        _fresh_state(monkeypatch, tmp_path)
        _handle_provider_failure("NIM Fast", Exception("NIM Fast tool execution did not resolve"))
        assert "NIM Fast" not in brain._provider_backoff_until
        assert brain._provider_health["NIM Fast"]["circuit_open"] is False
        assert brain._provider_health["NIM Fast"]["failures"] == 1

    def test_hard_error_still_backs_off(self, monkeypatch, tmp_path):
        _fresh_state(monkeypatch, tmp_path)
        _handle_provider_failure("Groq", Exception("503 Service Unavailable"))
        assert "Groq" in brain._provider_backoff_until
        assert brain._provider_health["Groq"]["circuit_open"] is True


class TestMalformedToolName:
    def test_detects_channel_markers(self):
        assert _is_malformed_tool_name("read_file<|channel|>commentary") is True
        assert _is_malformed_tool_name("scan_project<|") is True
        assert _is_malformed_tool_name("read_file") is False
        assert _is_malformed_tool_name("") is False

    def test_execute_tool_fails_fast(self):
        # Regression: the NIM model emitted 'read_file<|channel|>commentary';
        # the loop executed it as an unknown tool, ate another API round, and
        # finally died with "did not resolve" after ~2 minutes. Now the
        # malformed name is rejected before any work happens.
        with pytest.raises(Exception, match="did not resolve"):
            _execute_tool("read_file<|channel|>commentary", {})
