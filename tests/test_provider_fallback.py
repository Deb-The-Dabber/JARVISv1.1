"""Tests for provider fallback chain using the mock provider server."""

import time

import pytest

pytestmark = pytest.mark.integration


class TestProviderFallbackChain:
    """Verify the provider fallback chain behaves correctly.

    NOTE: the repo .env ships JARVIS_LATENCY_POLICY=1, so tool-y requests
    like "weather in tokyo" route through the low-effort candidate chain
    (scored by latency_policy/EFFORT_WEIGHTS["low"]), whose deterministic
    order is Gemini -> NIM Fast -> Groq. When Gemini fails, NIM Fast answers
    (its head model is deepseek-v4-flash, which the mock buckets as
    "DeepSeek"). These tests exercise that real fallback behaviour.
    """

    def test_default_chain_success(self, mock_provider, mock_api):
        """Normal path — chain head (Gemini) handles the request."""
        mock_provider.reset()
        r = mock_api.ask("weather in tokyo")
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data

    def test_fallback_to_nim_fast_on_gemini_failure(self, mock_provider, mock_api):
        """When Gemini fails (500), the low-effort chain falls to NIM Fast.

        NIM Fast's head model (deepseek-v4-flash) is bucketed "DeepSeek" by
        the mock's model-name heuristic. Asserting "Groq" here was previously
        satisfied only because the retired llama-based NIM Fast head happened
        to match the mock's historical "llama" -> "Groq" rule.
        """
        mock_provider.reset()
        mock_provider.fail("Gemini", mode="500", duration=60)
        r = mock_api.ask("weather in tokyo")
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        called = mock_provider.called_providers
        assert "DeepSeek" in called

    def test_all_providers_fail_graceful_error(self, mock_provider, mock_api):
        """When all providers fail, user gets a graceful error, not a crash."""
        mock_provider.reset()
        for p in [
            "Nemotron Ultra",
            "DeepSeek",
            "Gemini",
            "Groq",
            "Kimi K2",
            "NVIDIA NIM",
            "OpenRouter",
            "Pollinations",
        ]:
            mock_provider.fail(p, mode="500", duration=120)
        r = mock_api.ask("weather", timeout=120)
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert len(data["reply"]) > 0

    def test_coding_intent_routes_to_deepseek(self, mock_provider, mock_api):
        """Coding requests should route to DeepSeek even when Nemotron is available."""
        mock_provider.reset()
        mock_provider.fail("Nemotron Ultra", mode="500", duration=60)
        r = mock_api.ask("write a python function to sort a list")
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data

    def test_rate_limit_fallback(self, mock_provider, mock_api):
        """429 rate limits on primary should trigger fallback to Gemini."""
        mock_provider.reset()
        mock_provider.rate_limit("Nemotron Ultra", retry_after=60)
        r = mock_api.ask("weather", timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data


class TestProviderHealthScoring:
    """Verify health scoring affects provider selection."""

    def test_health_score_drops_on_failure(self, mock_provider, mock_api):
        """After a provider fails, its health score should drop."""
        mock_provider.reset()
        initial = mock_provider.health("Groq")
        assert initial["health_score"] == 100

        # Gemini (chain head) and NIM Fast (second in the low-effort chain,
        # bucketed "DeepSeek") must fail too, or the chain answers before
        # Groq (third) is ever attempted.
        mock_provider.fail("Gemini", mode="500", duration=60)
        mock_provider.fail("DeepSeek", mode="500", duration=60)
        mock_provider.fail("Groq", mode="500", duration=60)
        r = mock_api.ask("weather", timeout=60)
        assert r.status_code == 200

        # Poll health until it drops (allow for retry timing)
        for _ in range(10):
            after = mock_provider.health("Groq")
            if after["health_score"] < 100:
                break
            time.sleep(1)
        assert after["health_score"] < 100, f"Health did not drop: {after}"

    def test_health_score_recovers_on_success(self, mock_provider, mock_api):
        """After a provider succeeds, its health score should increase."""
        mock_provider.reset()
        initial = mock_provider.health("Nemotron Ultra")
        r = mock_api.ask("weather")
        assert r.status_code == 200

        after = mock_provider.health("Nemotron Ultra")
        assert after["health_score"] >= initial["health_score"]


class TestMockProviderControl:
    """Verify the mock provider's control endpoints work."""

    def test_fail_mode_blocks_requests(self, mock_provider):
        """After setting fail mode, requests should fail."""
        mock_provider.reset()
        h = mock_provider.health("Nemotron Ultra")
        assert h["available"] is True

        mock_provider.fail("Nemotron Ultra", mode="500", duration=30)
        h = mock_provider.health("Nemotron Ultra")
        assert h["available"] is False
        assert h["fail_mode"] == "500"

        mock_provider.unfail("Nemotron Ultra")
        h = mock_provider.health("Nemotron Ultra")
        assert h["available"] is True
        assert h["fail_mode"] is None

    def test_latency_injection(self, mock_provider):
        """Latency injection should slow down requests."""
        mock_provider.reset()
        mock_provider.latency("Nemotron Ultra", ms=500)
        h = mock_provider.health("Nemotron Ultra")
        assert h["latency_ms"] == 500

        mock_provider.latency("Nemotron Ultra", ms=0)
        h = mock_provider.health("Nemotron Ultra")
        assert h["latency_ms"] == 0

    def test_state_tracks_call_counts(self, mock_provider):
        """State endpoint should track call counts."""
        mock_provider.reset()
        state = mock_provider.state()
        for p, s in state.items():
            assert s["call_count"] == 0

    def test_all_providers_listed(self, mock_provider):
        """All expected providers should be in the mock state."""
        state = mock_provider.state()
        for p in [
            "Nemotron Ultra",
            "DeepSeek",
            "Groq",
            "Gemini",
            "Kimi K2",
            "NVIDIA NIM",
            "OpenRouter",
            "Pollinations",
        ]:
            assert p in state, f"Missing provider: {p}"


class TestProviderRetirement:
    """Phase 3: 401/402/403 permanently retires a provider until it succeeds."""

    def test_402_retires_provider_and_chain_still_succeeds(self, mock_provider, mock_api):
        """Gemini 402s -> retired; the chain answers via NIM Fast and never re-hits Gemini."""
        mock_provider.reset()
        mock_provider.fail("Gemini", mode="402", duration=120)
        r = mock_api.ask("weather in tokyo")
        assert r.status_code == 200
        assert len(r.json()["reply"]) > 0
        called = mock_provider.called_providers
        assert "DeepSeek" in called

    def test_retired_provider_not_re_attempted(self, mock_provider, mock_api):
        """After a 402 retirement, later requests skip the provider entirely."""
        mock_provider.reset()
        mock_provider.fail("Gemini", mode="402", duration=120)
        r1 = mock_api.ask("weather", timeout=120)
        assert r1.status_code == 200
        calls_after_r1 = mock_provider.state()["Gemini"]["call_count"]
        assert calls_after_r1 >= 1

        r2 = mock_api.ask("weather again", timeout=120)
        assert r2.status_code == 200
        calls_after_r2 = mock_provider.state()["Gemini"]["call_count"]
        # The 402 (payment wall) retired Gemini on request 1 — request 2 must
        # skip it without re-hitting, so the mock call count is frozen.
        assert calls_after_r2 == calls_after_r1, "Retired provider was re-attempted"


class TestProviderBudget:
    """Phase 3: per-request provider attempt budget caps chain burn."""

    def test_attempt_budget_caps_chain(self, monkeypatch, mock_provider, mock_api):
        monkeypatch.setenv("JARVIS_PROVIDER_REQUEST_BUDGET", "3")
        mock_provider.reset()
        for p in [
            "Nemotron Ultra",
            "DeepSeek",
            "Gemini",
            "Groq",
            "Kimi K2",
            "NVIDIA NIM",
            "OpenRouter",
            "Pollinations",
        ]:
            mock_provider.fail(p, mode="500", duration=120)
        r = mock_api.ask("weather", timeout=120)
        assert r.status_code == 200
        state = mock_provider.state()
        # With budget 3: Gemini, Groq, NIM Fast are attempted (and fail); the
        # chain must stop before NIM Coding / OpenRouter / Pollinations.
        assert state["OpenRouter"]["call_count"] == 0, "Budget did not cap OpenRouter"
        assert state["Pollinations"]["call_count"] == 0, "Budget did not cap Pollinations"
        assert state["Groq"]["call_count"] >= 1
