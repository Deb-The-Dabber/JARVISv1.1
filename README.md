J.A.R.V.I.S. (Will be renamed to K.A.I.A soon though)

JARVIS is my personal AI assistant project. Started early 2025 as a local Ollama/llama3.1 brain, now it's a full desktop assistant with tools, memory, RAG, multi-provider routing, and a coding agent bolted on that can work on JARVIS's own codebase, including improving itself. Been through more rewrites than I can count at this point.

Architecture (current)

                         JARVIS
                            │
                            ▼
                 Two-stage intent classifier
                    (coarse 6-class router,
                     91.5% val accuracy)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
            Chat          Tools         Coding
                            │
                   89-class specialist MLPs
                    (fine-grained routing)
                            │
                            ▼
                  Model / Provider chain
        Gemini 2.5 Flash → Llama 4 Maverick (NIM)
              → Groq → OpenRouter → Pollinations
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           Memory                        RAG
   SQLite + ChromaDB + NetworkX
    + procedural + associative
What it does

Voice: Whisper.cpp for STT, say → ElevenLabs for TTS, OpenWakeWord
Memory: multi-layer. SQLite, ChromaDB, NetworkX for graph relationships, plus procedural and associative memory on top of that
Tools: modular tool folder. weather, file ops, browser automation, WiFi/network diagnostics, Onshape/CAD right now. Building out email (IMAP), a contacts CRUD, and PDF Q&A next
Routing: spent the most time here out of anything. Two-stage neural intent classifier — coarse router sorts requests into 6 broad classes (91.5% val accuracy), then fine-grained specialist MLPs across 89 classes do the actual tool match. Backup/healthcheck/cost guardrails sit on top so one bad routing decision doesn't turn into a burned API budget
Interfaces: web PWA, iPhone PWA, pywebview desktop shell. Remote over Tailscale. Launches via a macOS Launch Agent
Frontend: rebuilt in Svelte 5 + Vite + TS. Canvas 2D orb reacting to status, Svelte 5 runes ($state) instead of the old store pattern. Also have a parallel React version OpenCode scaffolded, mostly just to compare against
How I build this

I write the core architecture and skeleton myself, then bring in AI for the more complex coding tasks and wiring things together, stuff like the routing logic or connecting providers where getting the plumbing right matters more than typing speed. OpenCode + DeepSeek V4 Pro (via Nvidia NIM) handles a lot of that. Claude and Gemini get pulled in for architectural review, and honestly just for hardware questions too when I'm trying to figure out what to build next for KAIA. For stuff I'm actually trying to learn (Svelte frontend, mainly), I write it myself start to finish and only use AI to fill small gaps.

Bugs I found and fixed

Turns out JARVIS being "dumb" for a while wasn't a model problem, it was infrastructure — well, specifically these four things:

Circuit-breaker had a math error, was tripping the fallback chain way earlier than it should've
LLM routing call firing on requests that didn't need one at all. Just wasted latency
Reply truncation bug. Silently cutting off responses and I didn't notice for way too long
Keyword short-circuit grabbing requests before they even hit the real classifier
Fixed all four and the ceiling on JARVIS's actual performance is a lot higher than the "dumb" baseline I'd been debugging against for weeks. Pretty annoying in hindsight.

Other stuff that's broken

Providers timing out / erroring mid-conversation, no consistent pattern
Embedding migration to NVIDIA NeMo — had to write real migration/recovery tooling instead of just nuking the old vector index
Local classifiers being confidently, completely wrong
Fallback chains that ended up adding latency instead of saving it (the opposite of the point)
Tests silently hitting the wrong local server for who knows how long
State sprawl in general, makes debugging a slog
Why

I wanted to understand how all the pieces of an AI assistant actually fit together instead of just treating an LLM as the whole app. Routing, memory, tool calling, reliability all matter as much as the model itself, maybe more.

Not done. I'll keep updating this when I remember to.
