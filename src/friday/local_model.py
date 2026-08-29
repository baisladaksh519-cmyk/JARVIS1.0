# Local model abstraction for FRIDAY

# This module selects a local LLM backend if available (gpt4all or llama-cpp-python) and
# provides a simple streaming interface.

import os
import shutil
from typing import Generator, Optional

class LocalModel:
    """Simple wrapper around a local LLM backend.

    Usage:
        m = LocalModel(model_path=os.environ.get('LOCAL_MODEL_PATH'), threads=4)
        for token in m.generate("Hello", max_tokens=128):
            print(token, end="", flush=True)

    Backends supported (if installed): gpt4all, llama_cpp
    """
    def __init__(self, model_path: Optional[str] = None, threads: int = 4):
        self.model_path = model_path or os.environ.get('LOCAL_MODEL_PATH')
        self.threads = int(os.environ.get('THREADS', threads))
        self.backend = None
        self.model = None
        self._detect_backend()

    def _detect_backend(self):
        # Prefer gpt4all if available (easier Windows install in many cases)
        try:
            import gpt4all
            self.backend = 'gpt4all'
            self.gpt4all = gpt4all
            return
        except Exception:
            pass
        try:
            from llama_cpp import Llama
            self.backend = 'llama_cpp'
            self.Llama = Llama
            return
        except Exception:
            pass
        # No supported backend installed
        self.backend = None

    def open(self):
        if not self.model_path:
            raise RuntimeError('LOCAL_MODEL_PATH not set. Please set it to your model binary.')
        if self.backend == 'gpt4all':
            # lazy import and instantiate
            self.model = self.gpt4all.GPT4All(self.model_path, n_ctx=512)
        elif self.backend == 'llama_cpp':
            self.model = self.Llama(model_path=self.model_path, n_ctx=512, n_threads=self.threads)
        else:
            raise RuntimeError('No local backend found. Install gpt4all or llama-cpp-python.')

    def close(self):
        # placeholder for cleanup
        self.model = None

    def generate(self, prompt: str, max_tokens: int = 256) -> Generator[str, None, None]:
        """Generate tokens from the local model as a synchronous generator.

        This yields partial text tokens that can be printed incrementally.
        """
        if self.model is None:
            self.open()

        if self.backend == 'gpt4all':
            # gpt4all supports a streaming-like interface via `generate` iterator
            try:
                for tok in self.model.generate(prompt, max_tokens=max_tokens, streaming=True):
                    yield tok
            except TypeError:
                # older gpt4all API
                text = self.model.generate(prompt, max_tokens=max_tokens)
                yield text
        elif self.backend == 'llama_cpp':
            # llama-cpp-python streaming
            for tok in self.model.create_completion(prompt=prompt, max_tokens=max_tokens, stream=True):
                # create_completion yields dicts with 'choices'
                try:
                    delta = tok['choices'][0]['delta']
                    text = delta.get('content', '')
                    if text:
                        yield text
                except Exception:
                    # fallback: yield the raw chunk
                    yield str(tok)
        else:
            raise RuntimeError('No supported backend available; install gpt4all or llama-cpp-python')

