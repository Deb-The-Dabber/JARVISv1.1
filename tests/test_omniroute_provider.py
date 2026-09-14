"""OmniRoute provider integration tests.

Proves JARVIS can route its LLM requests through a local OpenAI-compatible
OmniRoute gateway (default http://localhost:20128/v1) via the existing
`_ask_openai_compatible` client, that the OmniRoute slot participates in the
existing fallback chain behind the OMNIROUTE_API_KEY env gate, and that a dead
gateway fails cleanly through JARVIS's own chain.

Convention follows tests/test_provider_chain.py: brain is imported lazily with
JARVIS_PROVIDER_HEALTH_FILE isolated so the real provider health file is never
touched.
"""


def _brain(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_PROVIDER_HEALTH_FILE", str(tmp_path / "health.json"))
    monkeypatch.setenv("JARVIS_LLM_FIRST", "0")
    import brain

    brain._PROVIDER_RETIRED.clear()
    monkeypatch.setattr(brain, "_PROVIDER_HEALTH_FILE", str(tmp_path / "health.json"))
    # Clean slate for the OmniRoute slot across tests in this process.
    for coll in (
        brain._provider_backoff_until,
        brain._provider_health_scores,
        brain._provider_health,
        brain._provider_consecutive_failures,
        brain._provider_usage_count,
    ):
        coll.pop("OmniRoute", None)
    return brain


def _disable_direct_providers(b, monkeypatch):
    """Make every existing direct provider unavailable except OmniRoute/Pollinations."""
    monkeypatch.setattr(b, "GEMINI_API_KEY", "")
    monkeypatch.setattr(b, "_gemini_available", lambda: False)
    monkeypatch.setattr(b, "GROQ_API_KEY", "")
    monkeypatch.setattr(b, "NVIDIA_NEMOTRON_API_KEY", "")
    monkeypatch.setattr(b, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(b, "JARVIS_MOCK_PROVIDERS", False)


class TestOmniRouteChainGating:
    def test_slot_disabled_without_api_key(self, monkeypatch, tmp_path):
        """No OMNIROUTE_API_KEY -> OmniRoute never attempted, Pollinations serves."""
        b = _brain(monkeypatch, tmp_path)
        _disable_direct_providers(b, monkeypatch)
        monkeypatch.setattr(b, "OMNIROUTE_API_KEY", "")
        hits = []
        monkeypatch.setattr(b, "ask_omniroute", lambda *a, **k: hits.append(True) or "omni-ok")
        monkeypatch.setattr(b, "ask_pollinations", lambda *a, **k: "poll-ok")

        reply = b._dispatch_fallback_chain("hi", [], "test", b.ProviderBudget())

        assert reply == "poll-ok"
        assert hits == []

    def test_slot_enabled_with_api_key_and_precedes_pollinations(self, monkeypatch, tmp_path):
        """With OMNIROUTE_API_KEY set, OmniRoute is attempted before Pollinations
        and the chain records the success under the OmniRoute health slot."""
        b = _brain(monkeypatch, tmp_path)
        _disable_direct_providers(b, monkeypatch)
        monkeypatch.setattr(b, "OMNIROUTE_API_KEY", "irrelevant-nonempty")
        poll_hits = []
        monkeypatch.setattr(b, "ask_omniroute", lambda *a, **k: "omni-ok")
        monkeypatch.setattr(b, "ask_pollinations", lambda *a, **k: poll_hits.append(True) or "poll-ok")

        reply = b._dispatch_fallback_chain("hi", [], "test", b.ProviderBudget())

        assert reply == "omni-ok"
        assert poll_hits == []
        assert b._provider_health.get("OmniRoute", {}).get("successes", 0) >= 1


class TestOmniRouteCallShape:
    def test_uses_existing_openai_compatible_client_with_auto_alias(self, monkeypatch, tmp_path):
        """ask_omniroute must call the existing OpenAI-compatible client with the
        auto-router alias so OmniRoute (not JARVIS) picks the upstream model."""
        b = _brain(monkeypatch, tmp_path)
        captured = {}

        def fake_openai_compat(provider_name, api_key, base_url, model, user_message, tool_results, **kwargs):
            captured.update(
                provider=provider_name, api_key=api_key, base_url=base_url, model=model,
                message=user_message, tool_results=tool_results,
            )
            return "ok"

        monkeypatch.setattr(b, "_ask_openai_compatible", fake_openai_compat)
        monkeypatch.setattr(b, "OMNIROUTE_API_KEY", "x")
        monkeypatch.setattr(b, "OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1")
        monkeypatch.setattr(b, "OMNIROUTE_MODEL", "auto")

        assert b.ask_omniroute("hello", ["ctx"]) == "ok"
        assert captured["provider"] == "OmniRoute"
        assert captured["base_url"] == "http://127.0.0.1:20128/v1"
        assert captured["model"] == "auto"
        assert captured["api_key"] == "x"

    def test_modern_tool_calling_format_enabled(self, monkeypatch, tmp_path):
        """OmniRoute proxies tool calls upstream, so it joins the modern
        tools/tool_choice provider set (same as Groq/OpenRouter)."""
        b = _brain(monkeypatch, tmp_path)
        assert "OmniRoute" in b._TOOLS_FORMAT_PROVIDERS


class TestOmniRouteWire:
    def test_openai_compatible_wire_roundtrip(self, mock_provider_server, monkeypatch, tmp_path):
        """Real HTTP round-trip through brain.ask_omniroute against the mock
        OpenAI-compatible server (the same class of endpoint OmniRoute exposes)."""
        b = _brain(monkeypatch, tmp_path)
        monkeypatch.setattr(b, "JARVIS_MOCK_PROVIDERS", True)
        monkeypatch.setattr(b, "MOCK_PROVIDER_URL", mock_provider_server)
        monkeypatch.setattr(b, "OMNIROUTE_API_KEY", "test-key")

        reply = b.ask_omniroute("just say hello", ["context preloaded; do not use tools"])

        assert isinstance(reply, str) and "Mock" in reply

    def test_mock_mode_chain_records_omniroute_slot(self, mock_provider_server, monkeypatch, tmp_path):
        """Chain-level audit: with direct providers disabled and mock mode on,
        the OmniRoute slot answers and the health ledger records it under the
        OmniRoute name (health recording lives in the chain, not ask_*)."""
        b = _brain(monkeypatch, tmp_path)
        _disable_direct_providers(b, monkeypatch)
        monkeypatch.setattr(b, "JARVIS_MOCK_PROVIDERS", True)
        monkeypatch.setattr(b, "MOCK_PROVIDER_URL", mock_provider_server)
        monkeypatch.setattr(b, "OMNIROUTE_API_KEY", "test-key")

        reply = b._dispatch_fallback_chain("hi", ["ctx"], "test", b.ProviderBudget())

        assert "Mock" in reply
        assert b._provider_health.get("OmniRoute", {}).get("successes", 0) >= 1


class TestOmniRouteFailureBehavior:
    def test_dead_gateway_falls_back_and_backs_off(self, monkeypatch, tmp_path):
        """Closed-port gateway: request must not hang; JARVIS records the failure
        and advances to the next chain slot (local offline last resort here)."""
        import socket

        # Grab a definitely-closed loopback port.
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        b = _brain(monkeypatch, tmp_path)
        _disable_direct_providers(b, monkeypatch)
        monkeypatch.setattr(b, "OMNIROUTE_API_KEY", "test-key")
        monkeypatch.setattr(b, "OMNIROUTE_BASE_URL", f"http://127.0.0.1:{port}/v1")
        poll_hits = []

        def boom(*a, **k):
            poll_hits.append(True)
            raise RuntimeError("pollinations down")

        monkeypatch.setattr(b, "ask_pollinations", boom)
        monkeypatch.setattr(b, "_local_intent_predict", lambda *a: ("chat", 0.9))
        monkeypatch.setattr(b, "ask_local_offline", lambda *a, **k: "local-offline-ok")

        reply = b._dispatch_fallback_chain("hi", [], "test", b.ProviderBudget())

        assert reply == "local-offline-ok"
        assert poll_hits == [True]  # chain advanced past OmniRoute
        assert b._provider_available("OmniRoute") is False
        assert brain_ok(b)


def brain_ok(b) -> bool:
    """No speculative assertions about error text — the chain survived and the
    failure was recorded for the OmniRoute slot."""
    return b._provider_consecutive_failures.get("OmniRoute", 0) >= 1
