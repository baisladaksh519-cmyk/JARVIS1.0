import os
from src.friday.pipeline import FridayPipeline


def test_pipeline_build_prompt_and_context_trimming(tmp_path, monkeypatch):
    dbpath = tmp_path / 'mem.db'
    monkeypatch.setenv('FRIDAY_MEMORY_DB', str(dbpath))
    monkeypatch.setenv('CONTEXT_MAX_CHARS', '200')
    p = FridayPipeline(model_path=None, threads=1, mode='very-low')
    # ingest some memories
    p.ingest_memory('A' * 150)
    p.ingest_memory('B' * 150)
    prompt = p.build_prompt('hello', k=5)
    assert 'User: hello' in prompt
    # context should be trimmed to the last CONTEXT_MAX_CHARS characters
    assert len(prompt) <= (200 + 200)  # rough bounding: context + prompt text
