import datetime
import importlib.util
import os
import sqlite3
import threading

from dotenv import load_dotenv

load_dotenv()

HOME = os.path.expanduser("~")
LEARNED_TOOLS_DIR = os.path.join(HOME, "jarvis_learned_tools")
os.makedirs(LEARNED_TOOLS_DIR, exist_ok=True)

GROQ_MODEL = os.environ.get("JARVIS_GROQ_MODEL", "groq/compound-mini")
OPENROUTER_MODEL = "deepseek/deepseek-v3:free"
GEMINI_MODEL = "gemini-2.5-flash"

_speak_fn = None
_on_tool_learned = None


def init(speak_fn, on_tool_learned):
    global _speak_fn, _on_tool_learned
    _speak_fn = speak_fn
    _on_tool_learned = on_tool_learned


def trigger_learning(task_description: str) -> dict:
    """Explicitly trigger the learner for a given task. Returns immediately; learning runs in background."""
    learn_capability(task_description)
    return {"status": "started", "task": task_description}


def get_learned_tools() -> list[dict]:
    """List all learned tools with metadata."""
    tools = []
    if not os.path.exists(LEARNED_TOOLS_DIR):
        return tools
    for filename in sorted(os.listdir(LEARNED_TOOLS_DIR)):
        if not filename.endswith(".py"):
            continue
        filepath = os.path.join(LEARNED_TOOLS_DIR, filename)
        try:
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            task = ""
            with open(filepath) as f:
                first_lines = [next(f) for _ in range(5)]
            for line in first_lines:
                if line.startswith("# Task:"):
                    task = line.replace("# Task:", "").strip()
                    break
            func_name = filename.split("_")[0]
            tools.append(
                {
                    "name": func_name,
                    "filename": filename,
                    "path": filepath,
                    "task": task,
                    "created": mtime,
                }
            )
        except Exception:
            pass
    return tools


def get_learning_stats() -> dict:
    """Get learning/tool usage stats from audit DB."""
    audit_db = os.path.join(HOME, "jarvis_audit.db")
    top_tools = []
    total_actions = 0
    if os.path.exists(audit_db):
        try:
            conn = sqlite3.connect(audit_db)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            row = cur.execute("SELECT COUNT(*) as cnt FROM audit").fetchone()
            total_actions = row["cnt"] if row else 0
            rows = cur.execute("SELECT tool_name, COUNT(*) as cnt FROM audit GROUP BY tool_name ORDER BY cnt DESC LIMIT 10").fetchall()
            top_tools = [{"tool": r["tool_name"], "count": r["cnt"]} for r in rows]
            conn.close()
        except Exception:
            pass

    learned = get_learned_tools()
    return {
        "total_audit_actions": total_actions,
        "top_tools": top_tools,
        "learned_tools_count": len(learned),
        "recent_learnings": learned[-5:] if learned else [],
    }


def delete_learned_tool(name: str) -> dict:
    """Delete a learned tool by name."""
    tools = get_learned_tools()
    for t in tools:
        if t["name"] == name or t["filename"] == name:
            try:
                os.remove(t["path"])
                return {"status": "deleted", "name": name}
            except Exception as e:
                return {"status": "error", "error": str(e)}
    return {"status": "not_found", "name": name}


def _speak(text):
    if _speak_fn:
        _speak_fn(text)


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _ask_openai_compatible(prompt: str, api_key: str, base_url: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=1200,
    )
    return (response.choices[0].message.content or "").strip()


def _ask_llm(prompt):
    errors = []
    groq_key = _env("GROQ_API_KEY")
    if groq_key:
        try:
            return _ask_openai_compatible(
                prompt,
                groq_key,
                "https://api.groq.com/openai/v1",
                GROQ_MODEL,
            )
        except Exception as e:
            errors.append(f"Groq: {e}")

    openrouter_key = _env("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            return _ask_openai_compatible(
                prompt,
                openrouter_key,
                "https://openrouter.ai/api/v1",
                OPENROUTER_MODEL,
            )
        except Exception as e:
            errors.append(f"OpenRouter: {e}")

    gemini_key = _env("GOOGLE_GENAI_API_KEY")
    if gemini_key:
        try:
            from google import genai

            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as e:
            errors.append(f"Gemini: {e}")

    return "Error: all learner LLM providers failed. " + "; ".join(errors)


def learn_capability(task_description: str):
    """
    Background task: research and write a new Python tool for Jarvis.
    Speaks updates as it works.
    """

    def _learn():
        _speak("I don't know how to do that yet. Let me figure it out. In the meantime, is there anything else you'd like me to work on?")

        # Step 1: Research
        _speak("Researching how to do this...")
        research_prompt = f"""I need to write a Python function for a macOS AI assistant called Jarvis.
The function should: {task_description}

Requirements:
- Must work on macOS (Mac Mini M1)
- Use only standard Python libraries or: requests, subprocess, psutil, pyautogui
- Must be a single Python function
- Must return a string describing what happened
- Must handle errors gracefully

First, briefly explain your approach in 2-3 sentences.
Then write the complete Python function."""

        research = _ask_llm(research_prompt)

        # Step 2: Extract code
        code_prompt = f"""Based on this plan:
{research}

Write ONLY the Python function code — no explanation, no markdown, no backticks.
Just the raw Python function starting with 'def '.
The function must return a string result."""

        code = _ask_llm(code_prompt)

        # Clean up code
        code = code.replace("```python", "").replace("```", "").strip()

        if not code.startswith("def "):
            _speak("I had trouble writing that capability. I'll try again later.")
            return

        # Step 3: Extract function name
        try:
            func_name = code.split("def ")[1].split("(")[0].strip()
        except Exception:
            _speak("Something went wrong while learning. I'll try again later.")
            return

        # Step 4: Save to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{func_name}_{timestamp}.py"
        filepath = os.path.join(LEARNED_TOOLS_DIR, filename)

        full_code = f"""# Learned by Jarvis on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
# Task: {task_description}

import subprocess
import os
import requests
import psutil

HOME = os.path.expanduser("~")

{code}
"""
        with open(filepath, "w") as f:
            f.write(full_code)

        # Step 5: Try to load it
        try:
            spec = importlib.util.spec_from_file_location(func_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, func_name)

            # Register the new tool
            if _on_tool_learned:
                _on_tool_learned(func_name, fn, task_description)

            _speak(f"I figured it out. I've learned how to {task_description}. You can ask me to do it now.")

        except Exception as e:
            _speak(f"I wrote the code but ran into an issue loading it: {str(e)[:100]}. I've saved it to {filepath} if you want to review it.")

    t = threading.Thread(target=_learn, daemon=True)
    t.start()


def load_learned_tools():
    """Load all previously learned tools on startup."""
    loaded = {}
    if not os.path.exists(LEARNED_TOOLS_DIR):
        return loaded
    for filename in os.listdir(LEARNED_TOOLS_DIR):
        if not filename.endswith(".py"):
            continue
        filepath = os.path.join(LEARNED_TOOLS_DIR, filename)
        try:
            task = ""
            with open(filepath) as f:
                first_lines = [next(f) for _ in range(5)]
            for line in first_lines:
                if line.startswith("# Task:"):
                    task = line.replace("# Task:", "").strip()
                    break
            func_name = filename.split("_")[0]
            spec = importlib.util.spec_from_file_location(func_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, func_name):
                fn = getattr(module, func_name)
                loaded[func_name] = fn
                if _on_tool_learned:
                    _on_tool_learned(func_name, fn, task)
        except Exception:
            pass
    if loaded:
        print(f"  Loaded {len(loaded)} learned tool(s): {', '.join(loaded.keys())}")
    return loaded
