"""Phase 3 unit tests: provider retirement (401/402/403) + per-request attempt budget.

brain is imported lazily with an isolated JARVIS_PROVIDER_HEALTH_FILE so the real
~/.jarvis/provider_health.json is never loaded or written.
"""
import threading


def _brain(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_PROVIDER_HEALTH_FILE", str(tmp_path / "health.json"))
    monkeypatch.setenv("JARVIS_LLM_FIRST", "0")
    monkeypatch.setenv("JARVIS_MOCK_PROVIDERS", "0")
    import brain

    brain._PROVIDER_RETIRED.clear()
    return brain


class TestErrorClassification:
    def test_402_is_permanent(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        assert b._classify_error("402 Payment Required: insufficient credits") == "permanent"
        assert b._classify_error("Payment Required") == "permanent"
        assert b._classify_error("HTTP Error 402") == "permanent"

    def test_transient_5xx_is_not_permanent(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        assert b._classify_error("502 Bad Gateway") == "internal"
        assert b._classify_error("503 Service Unavailable") == "internal"
        assert b._classify_error("timed out after 30s") == "timeout"

    def test_401_403_still_auth_error(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        assert b._classify_error("401 Unauthorized") == "auth_error"
        assert b._classify_error("403 Forbidden") == "auth_error"


class TestRetirement:
    def test_retire_provider_makes_it_unavailable(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        b._retire_provider("OpenRouter", "402 Payment Required: insufficient credits")
        assert b.is_provider_retired("OpenRouter")
        assert b._provider_available("OpenRouter") is False
        assert b._provider_available("Gemini") is True

    def test_success_unretires_provider(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        b._retire_provider("OpenRouter", "no credits")
        assert not b._provider_available("OpenRouter")
        b._record_provider_success("OpenRouter")
        assert b._provider_available("OpenRouter") is True
        assert not b.is_provider_retired("OpenRouter")

    def test_health_exposes_retired_reason(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        b._retire_provider("Pollinations", "402 no credits")
        h = b.get_provider_health().get("Pollinations", {})
        assert h.get("retired") is True
        assert "no credits" in h.get("retired_reason", "")

    def test_retirement_persists_and_restores(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        b._retire_provider("OpenRouter", "402 billing")
        b._PROVIDER_RETIRED.clear()
        b._load_provider_health()
        assert "OpenRouter" in b._PROVIDER_RETIRED
        assert b._provider_available("OpenRouter") is False


class TestProviderBudget:
    def test_budget_init_and_exhaustion(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        assert b._provider_budget_init() == 6
        for _ in range(6):
            assert b._provider_budget_try("X")
        assert b._provider_budget_left() == 0
        assert b._provider_budget_try("X") is False

    def test_budget_resets_per_request(self, monkeypatch, tmp_path):
        b = _brain(monkeypatch, tmp_path)
        b._provider_budget_init()
        for _ in range(6):
            b._provider_budget_try("X")
        assert b._provider_budget_left() == 0
        b._provider_budget_init()
        assert b._provider_budget_left() == 6
        assert b._provider_budget_try("X") is True


class TestProviderHealthConcurrency:
    """Test that provider health updates are thread-safe."""

    def test_concurrent_failure_recording(self, monkeypatch, tmp_path):
        """Concurrent failure recordings should not lose counts and handle circuit breaker correctly."""
        b = _brain(monkeypatch, tmp_path)
        provider = "TestProvider"
        num_threads = 10
        failures_per_thread = 5  # Total 50 failures

        def record_failures():
            for _ in range(failures_per_thread):
                b._record_provider_failure(provider)

        threads = [threading.Thread(target=record_failures) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no crash and state is consistent
        # Note: circuit breaker resets counter at 3 failures, so final count will be
        # (total_failures % 3) but the health score should reflect all failures
        failures = b._provider_consecutive_failures.get(provider, 0)
        health = b.get_provider_health().get(provider, {})
        
        # Verify no crash and state is consistent
        assert isinstance(failures, int)
        assert failures >= 0
        # Health score should have decreased (50 failures * -10 each = -500, min 0)
        assert health.get("health_score", 100) == 0  # min(100, 100 - 50*10) = 0
        assert health.get("failures", 0) >= 50  # total failures recorded
        assert health.get("total", 0) >= 50

    def test_concurrent_success_and_failure(self, monkeypatch, tmp_path):
        """Concurrent success and failure recordings should not corrupt state."""
        b = _brain(monkeypatch, tmp_path)
        provider = "TestProvider2"
        num_threads = 10
        ops_per_thread = 50

        def mixed_ops(is_failure: bool):
            for _ in range(ops_per_thread):
                if is_failure:
                    b._record_provider_failure(provider)
                else:
                    b._record_provider_success(provider)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=mixed_ops, args=(i % 2 == 0,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Just verify no crash and state is consistent
        failures = b._provider_consecutive_failures.get(provider, 0)
        health = b.get_provider_health().get(provider, {})
        assert isinstance(failures, int)
        assert isinstance(health.get("health_score", 0), int)
        assert health.get("health_score", 0) >= 0

    def test_concurrent_retirement(self, monkeypatch, tmp_path):
        """Concurrent retirement and un-retirement should not corrupt."""
        b = _brain(monkeypatch, tmp_path)
        provider = "TestProvider3"
        num_threads = 10

        def retire_ops():
            for _ in range(20):
                b._retire_provider(provider, "test reason")
                b._unretire_provider(provider)

        threads = [threading.Thread(target=retire_ops) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should end up in a consistent state (either retired or not)
        is_retired = b.is_provider_retired(provider)
        assert isinstance(is_retired, bool)

    def test_concurrent_health_snapshot(self, monkeypatch, tmp_path):
        """Concurrent get_provider_health() calls should not crash."""
        b = _brain(monkeypatch, tmp_path)
        b._record_provider_success("ProviderA")
        b._record_provider_failure("ProviderB")
        b._retire_provider("ProviderC", "test")

        def read_health():
            for _ in range(100):
                b.get_provider_health()

        threads = [threading.Thread(target=read_health) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # No crash = pass
