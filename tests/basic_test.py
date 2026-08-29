import os
import pytest

from src.friday import local_model


@pytest.mark.skipif('LOCAL_MODEL_PATH' not in os.environ, reason='No local model configured')
def test_local_model_import():
    # Simple import smoke test
    lm = local_model.LocalModel(model_path=os.environ.get('LOCAL_MODEL_PATH'))
    assert lm is not None
