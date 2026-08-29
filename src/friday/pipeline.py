# FRIDAY pipeline: retrieval-augmented prompt building and model orchestration

import os
from typing import List, Optional
from .local_model import LocalModel
from .memory import MemoryStore

class FridayPipeline:
    def __init__(self, model_path: Optional[str] = None, threads: int = 4, mode: str = 'very-low'):
        self.model = LocalModel(model_path=model_path, threads=threads)
        # derive max entries from env if provided
        max_entries = os.environ.get('FRIDAY_MAX_ENTRIES')
        max_mb = os.environ.get('FRIDAY_MAX_MEMORY_MB')
        if max_entries:
            try:
                max_entries = int(max_entries)
            except Exception:
                max_entries = 500
        elif max_mb:
            try:
                max_mb_i = int(max_mb)
                # rough heuristic: ~1KB per entry
                max_entries = max(50, int((max_mb_i * 1024)))
            except Exception:
                max_entries = 500
        else:
            max_entries = 500

        self.memory = MemoryStore(db_path=os.environ.get('FRIDAY_MEMORY_DB', 'friday_memory.db'), max_entries=max_entries)
        self.mode = mode
        # context trimming
        self.context_max_chars = int(os.environ.get('CONTEXT_MAX_CHARS', 4000))

    def build_prompt(self, user_query: str, k: int = 3) -> str:
        # retrieve top-k memories
        ctx = self.memory.search(user_query, top_k=k)
        ctx_text = "\n".join([f"- {r['text']}" for r in ctx]) if ctx else ''
        # ensure context is not too long by trimming oldest characters
        if len(ctx_text) > self.context_max_chars:
            ctx_text = ctx_text[-self.context_max_chars:]
        prompt = f"You are FRIDAY, an assistant. Use these memories:\n{ctx_text}\nUser: {user_query}\nAssistant:"
        return prompt

    def ask(self, user_query: str, max_tokens: int = 256):
        # set conservative defaults for very-low mode
        if self.mode == 'very-low':
            max_tokens = min(max_tokens, 128)
        prompt = self.build_prompt(user_query)
        # stream response while collecting full text for memory ingestion
        parts = []
        for token in self.model.generate(prompt, max_tokens=max_tokens):
            parts.append(token)
            yield token
        full = ''.join(parts).strip()
        # ingest memory only if non-empty and reasonably sized
        try:
            if full:
                mem_text = full if len(full) <= 2000 else full[:2000]
                self.memory.add(mem_text, metadata={"query": user_query})
        except Exception:
            # swallowing memory write errors to keep pipeline resilient
            pass

    def ingest_memory(self, text: str, metadata: Optional[dict] = None):
        self.memory.add(text, metadata=metadata)
