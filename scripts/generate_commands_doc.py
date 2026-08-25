"""Generate COMMANDS.md from commands_registry — the single source of truth.

Usage:  python scripts/generate_commands_doc.py
Run it whenever COMMANDS.md must reflect the current command set (i.e.
after editing commands_registry.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commands_registry import CAPABILITIES, REGISTRY

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COMMANDS.md")


def _table(entries: list) -> list[str]:
    lines = ["| Command | Aliases | What it does |", "|---|---|---|"]
    for e in entries:
        aliases = ", ".join(e["names"][1:]) if len(e["names"]) > 1 else "—"
        about = e["about"].replace("\n", "  \n")
        lines.append(f"| `{e['names'][0]}` | {aliases} | {about} |")
    return lines


def main() -> None:
    terminal = [e for e in REGISTRY if e["surface"] in ("terminal", "both")]
    chat = [e for e in REGISTRY if e["surface"] == "chat"]

    lines = [
        "# Jarvis Command Reference",
        "",
        "Generated from `commands_registry.py` — the single source of truth. "
        "**No manual edits**: if a command is missing, add it to the registry "
        "and run `python scripts/generate_commands_doc.py`.",
        "",
        "## The `//` namespace",
        "",
        "`//`-prefixed input is Jarvis's own command space: instant, offline, "
        "handled locally — no model round-trip. `//help` works both in the "
        "terminal and as a chat message (phone + API).",
        "",
        "```",
        "//help              full list",
        "//help <command>    usage + aliases for one command",
        "```",
        "",
        "> **Single-slash `/` input deliberately goes to the language model** — "
        "`/help` is *not* implemented on purpose: it's the reserved easter-egg "
        "namespace. Try it in a while.",
        "",
        "## Terminal commands",
        "",
        "Typed in the TUI at the `>` prompt.",
        "",
        *(_table(terminal)),
        "",
        "## Chat commands",
        "",
        "Natural-language triggers; type them anywhere (terminal, phone UI, "
        "API) or say them. Terminal-only commands above also type-able in the "
        "terminal regardless.",
        "",
        *(_table(chat)),
        "",
        "## Capabilities — no fixed phrase, just ask",
        "",
        *([f"- **{name}**: {blurb}" for name, blurb in CAPABILITIES] or []),
        "",
        "## Sessions",
        "",
        "Conversation threads are persistent and shared across devices. "
        "Phone: session dropdown in the UI header (top right). "
        "Terminal: `session list`, `session new <name>`, "
        "`session switch <id>`, `session rename <id> <name>`, "
        "`session reset`, `session delete <id>`.",
        "",
        "Stored as JSONL in `~/.jarvis/sessions/` (`index.json` + "
        "`<id>.jsonl`); concurrency-safe across processes (flock + atomic "
        "index writes).",
        "",
        "## Related reference files",
        "",
        "- `AGENTS.md` — architecture, provider chain, gotchas, safety model",
        "- `TEST_PLAN.md` — regression checklist",
        "",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT} ({len(REGISTRY)} command entries)")
    for e in REGISTRY:
        if not all(k in e for k in ("names", "surface", "usage", "about")):
            print(f"  WARN: entry missing keys: {e}")


if __name__ == "__main__":
    main()
