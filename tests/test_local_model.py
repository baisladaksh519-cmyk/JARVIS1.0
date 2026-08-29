from src.friday.local_model import LocalModel


def test_local_model_generates_friendly_error_when_missing():
    lm = LocalModel(model_path=None)
    gen = lm.generate("Hello", max_tokens=10)
    first = next(gen)
    assert "Local model generation error" in first or "LOCAL_MODEL_PATH" in first or "No local backend" in first
