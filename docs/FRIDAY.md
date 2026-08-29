# FRIDAY

FRIDAY is a lightweight, local-first assistant scaffold tuned for very-low-spec systems (Intel i5-6500, 8GB RAM). This document explains the purpose, quick start, and recommendations for running FRIDAY locally without paid APIs.

Key goals
- Zero-budget: local-only inference using quantized GGML/GPT4All/llama.cpp backends.
- Low memory footprint: defaults tuned for small quantized models and aggressive memory/pruning.
- Streamed responses and async-friendly pipeline to reduce latency and peak RAM.
- Simple, auditable code: no autopilot/remote execution enabled by default.

Quick setup (Windows 11)
1. Create and activate a Python venv (CMD):
   python -m venv .venv
   .\.venv\Scripts\activate

   PowerShell:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install Python packages required for tests/demo. Keep heavy packages optional:
   pip install -r requirements.txt

3. (Optional) Install local model backends if you want to run the demo:
   - gpt4all (recommended on Windows): follow gpt4all install instructions.
   - llama-cpp-python: install only if you have compatible build tools.

   These are optional. The core FRIDAY code and unit tests do not require them.

4. Download a quantized GGML/gpt4all model (prefer < 3B for 8GB RAM) and place it in a folder `models/`.
   - Recommended: a small q4_0 3B or 1-2B model (or gpt4all-j small model) to fit 8GB RAM.
   - Set LOCAL_MODEL_PATH in a .env file or environment variables to point to that binary.

Set LOCAL_MODEL_PATH (CMD)
   set LOCAL_MODEL_PATH=C:\path\to\models\your-model.bin

PowerShell
   $env:LOCAL_MODEL_PATH = 'C:\path\to\models\your-model.bin'

4. Increase Windows pagefile to 8-12 GB to reduce OOM risk (optional but helpful on 8GB RAM).

Run the demo (after model placed and .venv activated):
   python demo\friday_demo.py --mode very-low

Files of interest
- src/friday/local_model.py — local backend abstraction (gpt4all/llama.cpp compatible) with streaming and robust error messages.
- src/friday/pipeline.py — low-spec pipeline: prompt management, memory retrieval, context trimming, streaming generation, and memory ingestion.
- src/friday/memory.py — SQLite-backed embedding store with simple LRU pruning and safe serialization.
- src/friday/voice.py — optional TTS/STT hooks (disabled by default).
- demo/friday_demo.py — runnable demo showing streaming output.
- config.example.env — example configuration variables (do NOT commit secrets).
- requirements.txt — minimal dependency list for tests.
- .gitignore — ignores models, .env, venv, caches.
- tests/basic_test.py — smoke tests (skips heavy tests if no model).

Notes and next steps
- This scaffold intentionally avoids calling paid APIs. If later you want to switch to OpenRouter/Gemini or other cloud backends, we can add an adapter layer to route calls.
- I will not commit any model binaries or secrets.
