import re
from contextlib import contextmanager


def assert_tool_called(response_text: str, tool_name: str) -> bool:
    return tool_name in response_text


def assert_provider_used(response_text: str, provider: str) -> bool:
    provider_keywords = {
        "nemotron": ["Nemotron", "NVIDIA"],
        "groq": ["Groq", "compound-mini"],
        "openrouter": ["OpenRouter"],
        "deepseek": ["DeepSeek"],
        "gemini": ["Gemini"],
        "kimi": ["Kimi"],
        "pollinations": ["Pollinations"],
    }
    keywords = provider_keywords.get(provider.lower(), [provider])
    return any(k in response_text for k in keywords)


def assert_no_raw_json(response_text: str) -> bool:
    patterns = [
        r"```json\{.*?\}```",
        r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\}',
        r'\{"tool_calls":\s*\[',
        r"<tool_name>[^<]+</tool_name>",
        r"<tool_call>.*?function\(",
    ]
    for p in patterns:
        if re.search(p, response_text, re.DOTALL):
            return False
    return True


@contextmanager
def env_var(key: str, value: str):
    import os

    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


@contextmanager
def env_unset(key: str):
    import os

    old = os.environ.get(key)
    os.environ.pop(key, None) if key in os.environ else None
    try:
        yield
    finally:
        if old is not None:
            os.environ[key] = old
