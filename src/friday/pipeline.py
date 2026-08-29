# FRIDAY pipeline: retrieval-augmented prompt building and model orchestration

import os
from typing import List, Optional
from .local_model import LocalModel
from .memory import MemoryStore

class FridayPipeline:
    def __init__(self, model_path: Optional[str] = None, threads: int = 4, mode: str = 'very-low'):
        self.model = LocalModel(model_path=model_path, threads=threads)
        self.memory = MemoryStore(db_path=os.environ.get('FRIDAY_MEMORY_DB', 'friday_memory.db'))
        self.mode = mode

    def build_prompt(self, user_query: str, k: int = 3) -> str:
        # retrieve top-k memories
        ctx = self.memory.search(user_query, top_k=k)
        ctx_text = "\n".join([f"- {r['text']}" for r in ctx]) if ctx else ''
        prompt = f"You are FRIDAY, an assistant. Use these memories:\n{ctx_text}\nUser: {user_query}\nAssistant:"
        return prompt

    def ask(self, user_query: str, max_tokens: int = 256):
        prompt = self.build_prompt(user_query)
        # stream response
        for token in self.model.generate(prompt, max_tokens=max_tokens):
            yield token

    def ingest_memory(self, text: str, metadata: Optional[dict] = None):
        self.memory.add(text, metadata=metadata)

