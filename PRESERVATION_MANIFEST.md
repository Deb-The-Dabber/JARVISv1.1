# Preservation Manifest — JARVIS 2026-08-24T18:30:00Z

## Repository State
- Branch: feature/persistent-sessions
- HEAD: a3d29c3 feat: copy Jarvis replies + thinking to the OS clipboard from the TUI
- Remote: https://github.com/Deb-The-Dabber/JARVISv.1
- Ancestry: feature/persistent-sessions is 14 commits ahead of feat/embeddings-rag-nemo

## Working Tree
### Modified Tracked Files (21)
- .github/workflows/ci.yml
- AGENTS.md
- agent.py
- benchmarks/results/gate_bench_A.json
- benchmarks/results/gate_bench_B.json
- benchmarks/results/gate_bench_C.json
- benchmarks/results/gate_bench_D.json
- benchmarks/results/gate_bench_E.json
- benchmarks/results/gate_bench_summary.json
- brain.py
- config.py
- jarvis_local_nn/weights/intent_router.npz
- learner.py
- parsers/format_log.jsonl
- proactive.py
- terminal.py
- test_runner.py
- tests/conftest.py
- tests/helpers.py
- tests/mock_provider.py
- tests/test_provider_fallback.py

### Untracked Files (26)
- .opencode/
- COMMANDS.md
- PHASE1_FINAL_REPORT.md
- PHASE1_VALIDATED_REPORT.md
- PHASE1_VALIDATION_REPORT.md
- benchmarks/regression_gate.py
- benchmarks/results/latest_regression.json
- benchmarks/results/regression_gate_2026-08-20T20-26-47-393693.json
- benchmarks/results/regression_gate_2026-08-20T20-29-12-139771.json
- benchmarks/results/regression_gate_2026-08-20T20-30-00-433598.json
- benchmarks/results/regression_gate_2026-08-20T20-31-00-450212.json
- benchmarks/results/regression_gate_2026-08-20T20-35-43-506312.json
- benchmarks/results/regression_gate_2026-08-20T20-36-08-639467.json
- c.txt
- experiment.py
- experiment.py.broken
- experiment.py.corrupt2
- experiment_clean.py
- experiment_copy.py
- experiment_fresh.py
- experiment_new.py
- experiment_test.py
- jarvis_vision_experiment/
- scripts/generate_commands_doc.py
- summary.txt
- task.py
- test_class.py
- test_cls.py
- test_compile.py
- test_compile2.py
- test_compile3.py
- test_compile4.py
- test_indent.py
- test_indent2.py
- tests/test_agent_loop.py
- tests/test_agent_loop.py.bak
- tests/test_paste_mode.py
- tests/test_planner_json.py
- tests/test_provider_chain.py
- tests/test_regression_gate.py
- tests/test_tty_paste.py
- tests/test_verifier_adversarial.py

## Critical Artifacts (SHA-256)
- task.py: 3da8c627e7eae81f55a7464e843da0311de8db49eb21ba822ea98be0e8aeab6e
- experiment.py: c6c5fb5cb4a1778c66bc21cb90d41b75782fd996a5af4c6aa55570c00487e10b
- jarvis_local_nn/weights/intent_router.npz: b3bb71afdd237119dd98b317dcf4758686b195a84f81aaa0c15c7f2bfcec15c4
- benchmarks/results/before_2a.json: 743769129ec6b024bac31c16f7eb0fd0262630c8047c106bb3af5f91cdd4599f
- benchmarks/results/after_2a.json: 5db25f4c89b95e07d5a9c173b256b8b7a7ff86d70c8a357414c1f89c0902986a
- benchmarks/results/before_2b.json: 6353f2f84de1e800c8c18a2caa28d7d54069da90a98e40e0bcc3b5483e4f9d72

## Vision Strategy (Resolved)
- Runtime dependency: tui.py loads ArtificialRetina from vision.py
- Resolution: Absorb ArtificialRetina into tools/vision_retina.py
- External repo: https://github.com/Deb-The-Dabber/visiooon-.git (1 commit, vision.py only)

## ML Weights
- intent_router.npz: b3bb71afdd237119dd98b317dcf4758686b195a84f81aaa0c15c7f2bfcec15c4

## Environment (names only, no secret values)
- JARVIS_TUI, JARVIS_TTY_RAW, JARVIS_GROQ_MODEL, JARVIS_PROVIDER_REQUEST_BUDGET
- JARVIS_EVAL_MODE, JARVIS_MOCK_PROVIDERS, JARVIS_TTS_SILENT

## Python/OS
- Python: 3.12.0
- OS: Darwin 24.5.0
