# FRIDAY

FRIDAY is a lightweight, local-first assistant scaffold tuned for very-low-spec systems (Intel i5-6500, 8GB RAM). This document explains the purpose, quick start, and recommendations for running FRIDAY locally without paid APIs.

Key goals
- Zero-budget: local-only inference using quantized GGML/GPT4All/llama.cpp backends.
- Low memory footprint: defaults tuned for q4 quantized 3B-class models and aggressive memory/pruning.
- Streamed responses and async-friendly pipeline to reduce latency and peak RAM.
- Simple, auditable code: no autopilot/remote execution enabled by default.

Quick setup (Windows 11)
1. Create and activate a Python venv:
   python -m venv .venv
   .\.venv\Scripts\activate
2. Install Python packages (may install heavy deps for embeddings):
   pip install -r requirements.txt

3. Download a quantized GGML/gpt4all model (3B or smaller, q4) and place it in a folder `models/`.
   - Recommended: gpt4all-j from the GPT4All project or a q4_0 Vicuna/Alpaca 3B conversion.
   - Set LOCAL_MODEL_PATH in a .env file or environment variables to point to that binary.

4. Increase Windows pagefile to 8-12 GB to reduce OOM risk.

Run the demo (after model placed and .venv activated):
   set LOCAL_MODEL_PATH=C:\path\to\models\your-model.bin
   python demo\friday_demo.py --mode very-low

Files added
- src/friday/local_model.py — local backend abstraction (gpt4all/llama.cpp compatible) with streaming.
- src/friday/pipeline.py — low-spec pipeline: prompt management + memory retrieval + model call.
- src/friday/memory.py — small SQLite-backed embedding store with simple LRU pruning.
- src/friday/voice.py — optional TTS/STT hooks (disabled by default).
- demo/friday_demo.py — runnable demo showing streaming output.
- config.example.env — example configuration variables (do NOT commit secrets).
- requirements.txt — minimal dependency list.
- .gitignore — ignores models, .env, venv, caches.
- tests/basic_test.py — smoke tests (skips heavy tests if no model).

Notes and next steps
- This scaffold intentionally avoids calling paid APIs. If later you want to switch to OpenRouter/Gemini or other cloud backends, we can add an adapter layer to route calls.
- I will not commit any model binaries or secrets.
