# JARVIS — AI Personal Assistant (macOS)

## Quick Start

```bash
# Terminal mode (default)
cd /Users/debasishbeura/Jarvis
source venv/bin/activate
./run_jarvis.sh              # or: python terminal.py
JARVIS_DEBUG=1 ./run_jarvis.sh  # enable [DEBUG] brain logs (also appends to ~/.jarvis/logs/debug.log)

# Server/API mode (mobile UI via Tailscale)
python server.py             # listens on :8002
```

Terminal modes: `m` (manual — press Enter to record or type), `w` (wake word — "Hey Jarvis"), `q` (quit).

## Dev Commands

```bash
# Lint (ruff)
ruff check .                  # catch errors
ruff check . --fix            # auto-fix
ruff format .                 # format whole project

# Test (pytest)
pytest                        # run all tests/
pytest -v                     # verbose
pytest tests/test_foo.py      # single file

# Regression gate (pre-merge)
pytest tests/regression/ -m "regression and not local"  # CI-safe (no display)
pytest tests/regression/                                 # full (needs display on macOS)

# Eval harness
python eval_runner.py tests/eval/golden_set.jsonl

# Fallback tests
pytest tests/test_provider_fallback.py -v
```

Testing stack: `pytest` + `ruff`. Config in `pyproject.toml` at root.

## Critical `.env` Keys

**Never set `GOOGLE_API_KEY`** — the GenAI SDK auto-detects it and overrides `GOOGLE_GENAI_API_KEY`.

- `GOOGLE_GENAI_API_KEY` — Gemini (primary, tool calling)
- `NVIDIA_API_KEY` — vision/screen analysis
- `NVIDIA_NEMOTRON_API_KEY` — Nemotron & NIM models
- `DEEPSEEK_API_KEY` — vestigial since Phase 2A (dedicated DeepSeek route removed); kept for `.env` compatibility, no longer read by `brain.py`
- `GROQ_API_KEY`, `OPENROUTER_API_KEY` — fallback providers
- `ELEVENLABS_API_KEY` — TTS (free tier → Edge-TTS → macOS `say`; quota warning prints once/session)
- `EDGE_TTS_VOICE` — Microsoft Edge TTS voice (default: `en-US-JennyNeural`)
- `TAVILY_API_KEY` — web search
- `HUGGINGFACE_TOKEN` — embeddings/downloads

Constants in `config.py` (user info, paths, TTLs, models).

## Entrypoints

| File | Role |
|------|------|
| `terminal.py` | CLI: voice/text → brain → TTS, wake word loop, proactive engine. Textual "control console" TUI by default on real terminals (ctrl+1..7 switch panes — digit keys never collide with TextArea editing, unlike ctrl+w/v which are word-delete/paste there; ctrl+r record, ctrl+c copy/cancel in composer, ctrl+q quit; composer bridges the real macOS clipboard — cmd+v/super+v and ctrl+v paste it (Textual's internal clipboard is bypassed), cmd+c/ctrl+c push selections out to it; Catppuccin Mocha theme); `JARVIS_TUI=0` forces the classic REPL, `JARVIS_TUI_VISION=0` skips camera autostart, `JARVIS_TUI_ANNOUNCE=1` speaks flavor lines, `JARVIS_CAMERA_INDEX=<0-4>` forces the vision camera (auto-pick prefers whatever camera actually delivers frames — virtual/Continuity cameras that only produce black frames are skipped; camera streams only while the VISION tab is open). The SYSTEM tab is a mini Activity Monitor: live per-process table (2s refresh only while the tab is open), keys c/m/d/n/e switch the metric like Activity Monitor's tabs — CPU / MEM / DISK (system-wide I/O rates) / NET (system-wide rates; macOS psutil has no per-process net/disk counters, so those two show the busiest-by-CPU processes underneath) / ENERGY (CPU-weighted EMA). Shell aliases: `js1` = classic REPL (`JARVIS_TUI=0`), `je1` = control console (`JARVIS_TUI=1`) — defined in `~/.zshrc` |
| `server.py` | FastAPI :8002: `/ask`, `/ask-voice`, `/health`, `/system`, `/weather`, `/recap`, `/memories`, `/priorities`, `/audit`, `/brain/reset`, `/oauth/*`, `/learner/*`, `/inspect` |
| `brain.py` | Provider chain, tool calling, safety gating, tool definitions, intent router, circuit breaker, `_tool_call_names` tracking, `get_last_tool_calls()` |
| `routing_policy.py` | Optional policy-driven provider selection (Phase 2A): capability/latency/cost/health scoring, hard gates, classifier-output parsing. Wired via `JARVIS_ROUTER_POLICY` / `JARVIS_ROUTER_CHEAP` |
| `benchmarks/` | Coding-routing benchmark harness (`coding_routing_bench.py`, `compare_bench.py`, `tests/eval/routing_golden.jsonl`, `results/`) |
| `jarvis_local_nn/` | From-scratch NumPy tiny tensor library + **two-stage intent router**. Stage 1: coarse 6-way router (MiniLM 384-dim → MLP 384→128→6), weights in `weights/intent_router.npz`. Stage 2: one specialist MLP per bucket (same arch → N fine classes, 89 total) in `weights/specialists/<bucket>.npz`, taxonomy in `taxonomy.yaml` (fine classes, tool→intent map, `execution_primitives` excluded from the classifier). Retrain coarse: `python -m jarvis_local_nn.training.train`; specialists: `python -m jarvis_local_nn.training.train_specialist --all`. Eval: `python -m jarvis_local_nn.training.evaluate` (golden set, scores coarse + fine via `expected_fine`). Calibrate: `python -m jarvis_local_nn.training.calibrate` (coarse per-intent gates) and `python -m jarvis_local_nn.training.calibrate_fine --all` (writes `weights/specialists/thresholds.json` per-class gates). Auto-retrain: `jarvis_local_nn/training/auto_retrain.py` — debounced (default 600s) background retrain triggered from `brain._on_tool_learned` when the learner adds a new tool (`JARVIS_AUTO_RETRAIN_ENABLED=0` off, `JARVIS_AUTO_RETRAIN_INTERVAL` window). Toggles: `JARVIS_LOCAL_INTENT_ENABLED=0` (router; independent of the MLX local-chat `JARVIS_LOCAL_ENABLED`); coarse gates: global `JARVIS_LOCAL_INTENT_CONFIDENCE=0.85` + per-intent `JARVIS_LOCAL_INTENT_CONFIDENCE_{CHAT,CODING,TOOL_USE,REASONING,SELF_MOD,AUTOMATION}` (the repo `.env` currently ships 2A-calibrated values — chat 0.59, coding 0.5, tool_use 0.82, reasoning 0.5); fine gates: `weights/specialists/thresholds.json` + per-class `JARVIS_FINE_CONFIDENCE_<CLASS>` overrides (bucket defaults 0.85/0.90); **agreement gate** `JARVIS_LOCAL_AGREEMENT_GATE=1` (2B.2): when the coarse router is confident but the specialist/fine label is withheld, treat as suspicious and escalate to LLM classify — applies to the local_nn AND cheap (Groq) paths. Benchmarking: `benchmarks/classifier_gate_calibrate.py` (offline threshold sweep over the 30-case golden set, no API) and `benchmarks/classifier_gate_bench.py` (A/D/B/C gate configs via fresh-env subprocess replays; results in `benchmarks/results/gate_bench_*`), designed to A/B the same frozen gates against Groq when available |. Embeddings cached in `~/.jarvis/nn_cache/embeddings.npz` (keyed by text hash; incremental retrains embed only new entries). First call per process loads MiniLM (~8s), then 10-70ms/call; reuses vector_memory's embedder. Also acts as last-resort offline provider in `ask_with_tools` (`local_nn` canned reply + locally prefetched tool results). Runtime: `brain.classify_intent` fast-path → coarse bucket → `predict_fine_gated` (per-class gate; below gate the fine label is withheld and routing falls to the LLM) → exposed via `brain.get_last_fine_intent()`, consumed by `_FINE_PREFETCH` (weather_current/sys_info/disk_usage → safe no-arg tool prefetch) |
| `tools/` | 13+ modules: system, browser, spotify, discord, calendar, file, code, computer, vision, communication, **google_docs**, **google_slides**, **google_forms**, **inspect_tools** |
| `learner.py` | LLM code-gen learning: `trigger_learning()`, CRUD for learned tools, web API endpoints |
| `memory.py` | Vector (ChromaDB) + RAG + graph + associative memory |
| `proactive.py` | Background monitors → priority engine → spoken alerts |
| `self_test/` | **Self-test agent** (Phase 1: log-analysis oracle). Scans `~/.jarvis/logs/jarvis.jsonl` (last 24h by default) with objective heuristic detectors (provider errors, tool-call errors, timeouts > `JARVIS_SELF_TEST_TIMEOUT_SECONDS` default 120s, empty replies for tool intents), then optionally asks the LLM to surface *possible* issues (triage only — every finding starts `open` and needs user `test confirm <id>` / `test dismiss <id>`). Findings + run history persist as JSONL in `~/.jarvis/self_test/` (`JARVIS_SELF_TEST_DIR` override). Trigger via chat ("test yourself", "check your code for bugs") or terminal (`test`, `test run`, `test logs` (heuristics only), `test status`, `test report`, `test findings`, `test history`, `test stop`). Live progress streams to the terminal as `[Self-Test]` lines via event_bus `self_test` events. LLM review toggle `JARVIS_SELF_TEST_LLM=0`. Scheduled auto-runs: `JARVIS_SELF_TEST_SCHEDULE=1` + `JARVIS_SELF_TEST_INTERVAL` (default 6h). PII redacted (reuses `agent._sanitize_goal`) before LLM review. Out of scope (Phase 2+): autonomous test generation / exploration / real tool execution.
| `backup.py` | **State backup**: snapshots watchlog DB, Chroma vector DB, trained NN weights, self-test findings, daily cost ledger to `~/.jarvis/backups/` with retention (`JARVIS_BACKUP_KEEP` default 7). Trigger via chat ("backup now") or terminal (`backup`, `backups`).
| `healthcheck.py` | **Startup health check**: verifies watchlog DB, Chroma vector DB, NN weights, primary API keys, RAG folder, self-test store, internet reachable. Reports at terminal startup + `/health` endpoint.
| `action_sandbox.py` | + dedicated self-modification audit trail to `~/.jarvis/self_mod_audit.jsonl` (what changed, when, outcome, backup). View via terminal `selfmod log`.
| `jarvis_logger.py` | + daily cost ledger (`~/.jarvis/cost_daily.jsonl`) for budget guardrails.
| `brain.py` | + daily budget guardrail (`JARVIS_DAILY_BUDGET_USD`, 0 = unlimited); when exceeded, pauses cloud calls, tries local model, returns message. Chat routes: "backup now", "health check", "prune memory [N days]".
| `terminal.py` | Commands: `backup`, `backups`, `health`, `selfmod log`, `memory prune [days]`.
| `server.py` | `/health` now includes dependency check results.
| `safety.py` | Tool confirmation/blocking: `SAFE` (auto), `WARNING` (confirm), `DANGEROUS` (confirm), `BLOCKED` (deny) |
| `vector_memory.py` | ChromaDB memory index; remote embeddings via NIM `nemotron-3-embed-1b` (2048-dim) by default, `JARVIS_EMBEDDING=local` for offline/CI (MiniLM 384-dim; repo `.env` pins `local`). Single persistent `httpx.Client` shared for all embed calls (never per-request — ENOBUFS guard), batch `JARVIS_EMBED_BATCH` (64), truncate `JARVIS_EMBED_MAX_CHARS` (6000), in-memory LRU for repeats, `get_embedding()` alias for perf_router. **Embedding migration is EXPLICIT-ONLY**: importing vector_memory never deletes/recreates/migrates a collection (background init opens/reads/checks; any model/dimension mismatch is logged as deferred to `~/.jarvis/embeddings_backup/deferred_migration.log`); run `python scripts/migrate_embeddings.py --mode nemo|local --yes` for the verified foreground sequence (verify source → timestamped backup+`snapshot.jsonl` → export → recreate → batched re-embed → count/id/dimension verification, exit 1 with restore command on any failure). Benchmark: `scripts/retrieval_eval.py` (`--mode local|nemo` then `--compare`) |
| `scripts/migrate_embeddings.py` | **Only** sanctioned embedding migration for the memory index. Foreground-only — never invoked by imports, background threads, or smoke tests. Stages: verify mode/model → verify source exists + count → export ids/docs/metas → **timestamped backup before any destructive op** → recreate → batched re-embed → verify count/id-set/dimensions → print backup path. `--dry-run` checks preconditions, `--yes` skips confirmation; exits 1 with restore command on any failure. `--rebuild-from-sources` mode (additive, local mode only): reconstructs lost rows in the CURRENT index from `~/jarvis_watchlog.db` (only proactive-vectorized event signatures), `~/.jarvis/logs/jarvis.jsonl` (non-error requests between index epoch 2026-08-07 and the incident cutoff), and SQLite as authority — snapshots first, dedupes by id + base-window content match, ends with full statistics |
| `file_sandbox.py` | File-write sandbox: stages create/write/append as diffs, approval before apply |
| `action_sandbox.py` | Universal action sandbox: previews (sandboxed run / intent), self-mod review, typed confirm |
| `eval_runner.py` | Eval harness: runs golden set through brain, measures intent/tool/keyword accuracy |
| `tests/mock_provider.py` | Deterministic mock LLM provider server (8 providers, rate limits, health scoring, fail injection) |
| `tests/conftest.py` | Pytest fixtures: `has_display`, `api`, `jarvis_server`, `mock_provider`, `mock_api`; `_free_port()`, `_stop_proc()` helpers |
| `tests/regression/test_regressions.py` | P0 regression tests: `TestFunctionCallingRegressions` (3) + `TestP0Regressions` (5) |
| `tests/test_provider_fallback.py` | Provider fallback chain tests (11 tests) |
| `.github/workflows/ci.yml` | CI/CD: lint (ruff), unit (pytest + Xvfb + mock provider), regression (real providers, secrets-gated) |
| `ci-requirements.txt` | Portable pip-only deps for CI (249 packages, no conda artifacts) |

## Capabilities

| Category | What Jarvis can do |
|----------|-------------------|
| **Chat** | Voice or text conversation, remembers context, semantic memory |
| **Weather** | Current conditions, detailed forecast (temp, humidity, wind, rain) |
| **Web** | Search, browse pages, open URLs in Safari/Chrome |
| **Apps** | Open, focus, quit any macOS app; list running apps |
| **Spotify** | Play/pause/skip, search and play songs |
| **Discord** | Open channels, send messages |
| **Calendar** | Read today's events, add new events |
| **System** | CPU/RAM/disk usage, disk space, open apps |
| **Files** | Find files, show largest, organize downloads, open in Finder |
| **Screen** | Read & summarize screen contents (vision) |
| **Computer** | Mouse move/click, type text, press keys, screenshot, click elements by description (vision) |
| **Terminal** | Run shell commands (dangerous ones require confirmation) |
| **iMessage** | Send messages (requires confirmation) |
| **Knowledge** | Query/add relationships in knowledge graph |
| **Notes** | Search personal notes via RAG |
| **Timers** | Set/cancel countdown timers with spoken alerts |
| **Memory** | Remember facts, forget them, semantic search |
| **Proactive** | Background monitoring for CPU, internet, calendar events |
| **Google Docs** | Create, read, update documents (OAuth) |
| **Google Slides** | Create, read, update presentations (OAuth) |
| **Google Forms** | Create, read, update forms/responses (OAuth) |
| **Inspect** | Query own tool inventory and capabilities |
| **Learner** | LLM-generated tool creation with CRUD API and web UI |

## Provider Chain (auto-failover)

```
User → Intent Router → Nemotron Ultra (primary) → Fallbacks
```

| Tier | Provider | Role |
|------|----------|------|
| 1 | **Nemotron 3 Ultra** (→ MiniMax M3 understudy) | Primary — tool-first + final response (frontier slot head; in-slot understudy auto-fails over) |
| 2 | **Gemini 2.5 Flash** | Fallback — excellent tool calling, used when Nemotron unavailable |
| 3 | **Groq llama-3.3-70b** | Fallback — full tools support via modern `tool_choice` format |
| 4 | **NIM Fast** | super-49b-v1.5 → nano-30b → step-3.7-flash (low-effort/chat fast path) |
| 5 | **NIM Coding** | minimax-m3 → gpt-oss-20b → deepseek-v4-flash (coding/agentic) |
| 6 | **NIM Frontier** (scoring slot) | ultra → minimax-m3 understudy; the probe/low-effort chain's frontier entry |
| 7 | **OpenRouter deepseek-r1** | Last-resort fallback |
| 8 | **Pollinations.ai** | No-key emergency fallback |

NIM slots replace the legacy Tier 5/6 lists (Llama 4 Maverick, MiniMax M2.7, Qwen 3.5, Mistral Large 3 — all removed from the build.nvidia.com catalog). `NIM_MODEL_REASONING` (nemotron-3-super-120b-a12b) is **defined but not registered** — the E4 probe (`scripts/nim_probe.py`, `~/.jarvis/nim_probe.jsonl`) gates its activation (<10s median). Vision still uses `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` (working endpoint). Nano-Omni (`nemotron-3-nano-omni-30b-a3b-reasoning`) is the designated replacement — 400/Function-gate with both keys on 2026-08-13; the E4 probe (`scripts/nim_probe.py`) tracks its recovery and the swap flips when it serves 200. All slots are free endpoints under one `NVIDIA_NEMOTRON_API_KEY`.

### Phase 2A routing policy (optional, default off)

- `JARVIS_ROUTER_POLICY=1` — policy-driven provider selection (`routing_policy.py`): scores candidates by capability fit / observed latency / cost / health; hard-gates unavailable or unhealthy (health < 30) providers; fallback chain ordered by score when the primary fails.
- `JARVIS_ROUTER_CHEAP=1` — fast Groq classify before Nemotron when the user message isn't tool-y, avoiding the expensive Nemotron classify round-trip. Overrides fine-intent labels from the cheap path.
- Dedicated DeepSeek v4 primary/coding route and NIM DeepSeek tier removed in Phase 2A (bench-verified: intent 19.5s→7.5s avg, routing 86.5s→9.2s avg, stall errors 12→0, intent accuracy -0.09 with cheap classifier on — see `benchmarks/`).

## Gotchas

| Issue | Why | Mitigation |
|-------|-----|------------|
| `Error querying device -1` | No default mic | `terminal.py` auto-picks first working input |
| ElevenLabs quota | Free tier exhausted | Prints once/session, falls back to Edge-TTS → `say` |
| ChromaDB dimension error | Embedding model changed | Deferred by design — imports never migrate. Run `python scripts/migrate_embeddings.py --mode nemo --yes` explicitly (backup in `~/.jarvis/embeddings_backup/` for rollback) |
| Wake word unresponsive | Mic permission lost | Check System Settings → Privacy → Microphone |
| Any 503 on Gemini | `GOOGLE_API_KEY` set to non-Google key | Delete it from `.env` — keep only `GOOGLE_GENAI_API_KEY` |
| Stale :8002 process | Previous server didn't shut down cleanly | `_free_port()` in conftest.py kills it automatically |
| Mock provider refused | Port 18889 in use | `_free_port()` kills stale mock server processes |

## macOS Permissions

- **Accessibility**: System Settings → Privacy & Security → Accessibility → Terminal (or app running Python)
- **Microphone**: Privacy → Microphone
- **Calendar, Contacts, Spotify, Messages**: needed for respective tool modules

## Safety & Audit

- `safety.py` gates tools: `quit_app`, `send_imessage`, `run_terminal_command` require confirm
- `/audit` endpoint logs: `ALLOWED` / `EXECUTED` / `DENIED` / `BLOCKED` decisions
- Confirmation state persists per session
- File/computer tools (`create_file`, `write_file`, `move_and_click`, etc.) upgraded from `WARNING` to `SAFE`
- **File sandbox** (`file_sandbox.py`): all writes via `create_file` / `write_file` / `append_file` are staged as diffs and printed to the terminal; the write only happens after the user says yes (or `POST /confirm`). No discards it. View pending diff via `GET /files/pending`. Disable with `JARVIS_FILE_SANDBOX=0`; `JARVIS_EVAL_MODE` bypasses it
- **Action sandbox** (`action_sandbox.py`, default on): every dangerous action runs in a sandbox FIRST, the outcome is shown, then the user decides whether to proceed for real
  - Terminal/Python (`run_terminal_command`, `run_python`, `run_python_sandboxed`, `run_command_sandboxed`) → real OS-sandboxed run, output shown (`[Jarvis Sandbox] Preview`). `run_python` / `run_terminal_command` execute **outside** the sandbox after approval; the `*_sandboxed` variants stay sandboxed. Blocked patterns (`rm -rf /` etc.) are rejected before any preview runs, and `destructive_preview_block` additionally rejects `rm`/`mv`/`unlink`/`shred`/`rmdir`/`truncate` (shell or Python) targeting the Jarvis repo or protected core files — those previews never execute
  - Browser/computer/mail/drive/docs/goals tools → intent preview ("what WOULD happen") before running
  - First approval per session → later calls preview + auto-proceed after 3s unless cancelled (`auto_proceed`); non-tty (API) auto-proceeds immediately
  - **Self-modification review** (`modify_own_tool` or writes to protected files — brain.py, safety.py, server.py, tools/ etc.): static analysis (syntax, size delta ≤ 50%, critical symbols like `def process(` must survive), always-on isolated dry run, then **typed confirmation** (`confirm self-modify <file>` in terminal, or `POST /confirm` with that text) — never auto-proceeds. Backup to `/tmp/jarvis_sandbox_backups` (24h), auto-rollback on post-apply failure, tool registry reloaded in place. `JARVIS_PROTECTED_FILES` adds extra protected paths (comma-separated)
  - View pending: `GET /actions/pending` (`self_mod` / `file_write` / `confirm` types). Disable with `JARVIS_ACTION_SANDBOX=0`; `JARVIS_EVAL_MODE=1` bypasses (set in `tests/conftest.py` for regression fixtures)
- `/brain/reset` clears pending safe state, conversation history, and context

## Quick Test

Quickest way to sanity-check Jarvis. Run terminal, choose `m`, type these:

| Prompt | Expected behavior |
|--------|------------------|
| `weather` | Spoken + printed temp, humidity, conditions |
| `open safari` | Safari launches |
| `search web for python 3.13` | Returns web results |
| `what's my system usage` | CPU, RAM, disk |
| `remember I like black coffee` | Saves to memory |
| `what do you know about me` | Recalls saved facts |
| `set a timer called tea for 10 seconds` | Speaks "Timer done!" after 10s |

Full regression checklist at `TEST_PLAN.md`.

```bash
source venv/bin/activate && python terminal.py
# Choose 'm' and try any prompt above
```

API smoke test:
```bash
curl http://localhost:8002/health && echo ""
curl http://localhost:8002/system && echo ""
```

