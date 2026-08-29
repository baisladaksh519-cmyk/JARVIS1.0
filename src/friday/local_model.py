# Local model abstraction for FRIDAY (robust error handling and backend compatibility)

import os
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
            raise RuntimeError('LOCAL_MODEL_PATH not set. Please set it to your model binary (see docs/FRIDAY.md).')
        if self.backend == 'gpt4all':
            try:
                # gpt4all has varied APIs; handle common constructor signature
                self.model = self.gpt4all.GPT4All(self.model_path, n_ctx=512)
            except Exception as e:
                raise RuntimeError(f'Failed to load gpt4all backend: {e}. Check gpt4all installation and model path.')
        elif self.backend == 'llama_cpp':
            try:
                self.model = self.Llama(model_path=self.model_path, n_ctx=512, n_threads=self.threads)
            except Exception as e:
                raise RuntimeError(f'Failed to load llama-cpp-python backend: {e}. See docs/FRIDAY.md for install tips.')
        else:
            raise RuntimeError('No local backend found. Install gpt4all or llama-cpp-python. See docs/FRIDAY.md for instructions.')

    def close(self):
        # placeholder for cleanup
        self.model = None

    def generate(self, prompt: str, max_tokens: int = 256) -> Generator[str, None, None]:
        """Generate tokens from the local model as a synchronous generator.

        This yields partial text tokens that can be printed incrementally.
        """
        if self.model is None:
            self.open()

        try:
            if self.backend == 'gpt4all':
                # gpt4all generate may yield strings or objects depending on version
                try:
                    gen = self.model.generate(prompt, max_tokens=max_tokens, streaming=True)
                    for tok in gen:
                        # yield stringified token or chunk
                        yield str(tok)
                except TypeError:
                    # older gpt4all API that returns a single string
                    text = self.model.generate(prompt, max_tokens=max_tokens)
                    yield str(text)
            elif self.backend == 'llama_cpp':
                # llama-cpp-python streaming: yields dict chunks
                try:
                    for chunk in self.model.create_completion(prompt=prompt, max_tokens=max_tokens, stream=True):
                        # chunk could be a dict with choices -> delta/content or text
                        if isinstance(chunk, dict):
                            choices = chunk.get('choices') or []
                            if choices:
                                first = choices[0]
                                # delta style
                                delta = first.get('delta') or {}
                                if isinstance(delta, dict):
                                    text = delta.get('content') or delta.get('text') or ''
                                else:
                                    text = str(delta)
                                if text:
                                    yield text
                                continue
                        # fallback
                        yield str(chunk)
                except TypeError:
                    # non-streaming fallback
                    res = self.model.create_completion(prompt=prompt, max_tokens=max_tokens)
                    # try multiple shapes
                    text = None
                    if isinstance(res, dict):
                        choices = res.get('choices') or []
                        if choices:
                            text = choices[0].get('text') or choices[0].get('message', {}).get('content')
                    if text:
                        yield text
                    else:
                        yield str(res)
            else:
                raise RuntimeError('No supported backend available; install gpt4all or llama-cpp-python')
        except Exception as e:
            # yield a short error so caller can display it in a streamed UI
            yield f"\n\n[FRIDAY] Local model generation error: {e}\nPlease check your backend installation and model path." 
