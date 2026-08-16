"""Single source of truth for every Jarvis command.

Consumed by:
  - terminal.py              //help, //help <command>
  - brain.py                 chat //help (phone / API / spoken)
  - scripts/generate_commands_doc.py  → writes COMMANDS.md

Add or edit commands here, then regenerate the markdown doc:
    python scripts/generate_commands_doc.py

Design rule: the double-slash namespace (//help) is Jarvis's command
space. Single-slash input (/help) deliberately falls through to the
language model — it's reserved for easter eggs.
"""

# surface: "terminal" = TUI-only commands, "chat" = natural-language /
# typed-anywhere commands (routed by the brain), "both" = either.
# Two-space indentation in `about` is intentional — it's expanded by the
# doc generator (Markdown needs two trailing spaces for a line break).
REGISTRY = [
    # ── Help / meta ────────────────────────────────────────────────
    dict(
        names=["//help", "//commands", "//?"],
        surface="both",
        usage="//help [command]",
        about="List every Jarvis command. With an argument, show usage and "
        "aliases for one command. `//help` is instant and offline — the "
        "single-slash namespace (/help) goes to the model instead, as the "
        "designated easter-egg space.",
    ),
    # ── Terminal mode ──────────────────────────────────────────────
    dict(
        names=["quit", "exit", "q"],
        surface="terminal",
        usage="quit",
        about="Leave Jarvis.",
    ),
    dict(
        names=["wake"],
        surface="terminal",
        usage="wake",
        about="Switch to wake-word mode ('Hey Jarvis').",
    ),
    dict(
        names=["mic"],
        surface="terminal",
        usage="mic",
        about="Switch the microphone (first working input is picked at "
        "startup; use this after replugging).",
    ),
    dict(
        names=["/mode"],
        surface="terminal",
        usage="/mode text|paste|queue  (or /m t|p|q)",
        about="Change input mode: text (type or speak), paste (multi-line "
        "paste), queue (type and press Enter once to queue commands).",
    ),
    dict(
        names=["//paste", "//clipboard"],
        surface="terminal",
        usage="//paste",
        about="Read the clipboard (pbcopy/pbpaste) and submit it as a "
        "single message. The reliable way to hand Jarvis a multi-line task "
        "without paste-mode fragmentation. Use /mode paste for an "
        "interactive multi-line editor instead (submit with '/').",
    ),
    dict(
        names=["/queue show", "/queue clear"],
        surface="terminal",
        usage="/queue show | /queue clear",
        about="Inspect or flush the scheduled task queue.",
    ),
    dict(
        names=["/context"],
        surface="terminal",
        usage="/context",
        about="Show the current conversation context snapshot (state, last "
        "intent, tools used, elapsed).",
    ),
    dict(
        names=["/provider-status", "/ps", "/providers"],
        surface="terminal",
        usage="/provider-status",
        about="Show provider health scores, circuit breakers and backoff "
        "state for every model tier.",
    ),
    # ── Plugins / workflows / triggers ─────────────────────────────
    dict(
        names=["plugins", "plugins list", "plugins reload", "plugin install"],
        surface="terminal",
        usage="plugins | plugins reload | plugin install <name>",
        about="Manage Jarvis plugins: list loaded plugins, reload them from "
        "disk, or install a new one by name.",
    ),
    dict(
        names=["workflows", "wf"],
        surface="terminal",
        usage="workflows | workflow run <name> | wf history",
        about="List, run, and review workflow automation. `workflow run "
        "<name>` executes a saved workflow; `wf history` shows past runs.",
    ),
    dict(
        names=["triggers"],
        surface="terminal",
        usage="triggers | trigger add <cron> <command> | trigger "
        "remove|pause|resume <id> | trigger history",
        about="Schedule background triggers (cron or interval) that fire "
        "commands.",
    ),
    # ── Memory / RAG / knowledge graph ─────────────────────────────
    dict(
        names=["ingest"],
        surface="terminal",
        usage="ingest <path>",
        about="Ingest a file or folder into the RAG knowledge base.",
    ),
    dict(
        names=["rag search", "rag stats", "rag prune"],
        surface="terminal",
        usage="rag search <query> | rag stats | rag prune",
        about="Search vectors in the RAG index, show index stats, or prune "
        "stale chunks.",
    ),
    dict(
        names=["graph"],
        surface="terminal",
        usage="graph stats | graph extract <text> | graph neighbors "
        "<entity> | graph search <query>",
        about="Query and grow the knowledge graph: stats overview, extract "
        "relations from text, list an entity's neighbors, search nodes.",
    ),
    dict(
        names=["memory prune", "prune memory"],
        surface="both",
        usage="memory prune <days>  (default 30)",
        about="Delete explicit memories older than N days. Works typed in "
        "the terminal or as a chat message.",
    ),
    # ── Vision ─────────────────────────────────────────────────────
    dict(
        names=["vision"],
        surface="terminal",
        usage="vision analyze <image> | vision ocr <image> | vision video "
        "<path>",
        about="Run the vision model on an image (describe it, OCR it) or "
        "sample frames from a video.",
    ),
    # ── Agents / self-test / maintenance ───────────────────────────
    dict(
        names=["agents", "agent stop"],
        surface="terminal",
        usage="agents | agent stop <id>",
        about="List running autonomous agents and stop one by id.",
    ),
    dict(
        names=["test"],
        surface="both",
        usage="test | test run | test logs | test status | test report | "
        "test findings | test history | test stop | test confirm <id> | "
        "test dismiss <id>",
        about="Self-test agent: scan recent logs for bugs (heuristics + "
        "optional LLM triage), manage findings. Chat trigger: 'test "
        "yourself' / 'check your code for bugs'.",
    ),
    dict(
        names=["backup", "backups"],
        surface="both",
        usage="backup | backups | backup list",
        about="Snapshot state (watchlog DB, vector DB, NN weights, "
        "self-test findings, cost ledger) to ~/.jarvis/backups/ with "
        "retention. Chat trigger: 'backup now'.",
    ),
    dict(
        names=["health"],
        surface="both",
        usage="health | health check | status check",
        about="Startup-style health check: DBs, vector index, NN weights, "
        "API keys, RAG folder, internet. Chat trigger: 'health check'.",
    ),
    dict(
        names=["selfmod log", "self-mod log"],
        surface="terminal",
        usage="selfmod log",
        about="Show the self-modification audit trail (what Jarvis changed "
        "about itself, when, outcome).",
    ),
    # ── Sessions ───────────────────────────────────────────────────
    dict(
        names=["session"],
        surface="both",
        usage="session | session list | session new <name> | session "
        "switch <id> | session rename <id> <name> | session reset | "
        "session delete <id>",
        about="Persistent conversation threads shared between computer and "
        "phone. `session` lists all sessions with message counts; switch by "
        "id or name. The phone UI has the same switcher in the header.",
    ),
    # ── Chat (natural language, anywhere) ──────────────────────────
    dict(
        names=["remember that ..."],
        surface="chat",
        usage="remember that I like black coffee",
        about="Store a fact in long-term memory (global across sessions and "
        "devices).",
    ),
    dict(
        names=["what do you know about me"],
        surface="chat",
        usage="what do you know about me",
        about="Recall everything stored about you.",
    ),
    dict(
        names=["forget ..."],
        surface="chat",
        usage="forget my favorite color",
        about="Delete a stored memory (case/word order tolerant).",
    ),
    dict(
        names=["set a timer"],
        surface="chat",
        usage="set a timer called tea for 10 seconds",
        about="Start a countdown timer; Jarvis speaks an alert when it "
        "finishes. 'cancel the timer' stops it.",
    ),
    dict(
        names=["use local model", "use cloud"],
        surface="chat",
        usage="use local model | use cloud",
        about="Switch the brain between the offline local model and cloud "
        "providers.",
    ),
]

# ── Capabilities (natural-language chat, no fixed phrase) ─────────
CAPABILITIES = [
    ("Weather", "current conditions and detailed forecasts"),
    ("Web", "search, browse pages, open URLs in Safari/Chrome"),
    ("Apps", "open, focus, quit macOS apps; list running apps"),
    ("Spotify", "play, pause, skip; search and play songs"),
    ("Discord", "open channels, send messages"),
    ("Calendar", "read today's events, add new events"),
    ("System", "CPU/RAM/disk usage, disk space, open apps"),
    ("Files", "find files, show largest, organize downloads, open in Finder"),
    ("Screen", "read and summarize screen contents (vision)"),
    ("Computer", "move/click mouse, type text, press keys, screenshot "
     "(requires confirmation)"),
    ("Terminal", "run shell commands (dangerous ones require confirmation)"),
    ("iMessage", "send messages (requires confirmation)"),
    ("Knowledge", "query and grow the knowledge graph"),
    ("Notes", "search personal notes via RAG"),
    ("Timers", "set/cancel countdown timers with spoken alerts"),
    ("Memory", "remember facts, forget them, semantic search"),
    ("Proactive", "background monitors for CPU, internet, calendar events"),
    ("Google Docs/Slides/Forms", "create, read, update (OAuth)"),
    ("Inspect", "ask what tools Jarvis has and how they work"),
    ("Learner", "LLM-generated tool creation with CRUD API and web UI"),
    ("Sessions", "named conversation threads shared across devices"),
]
