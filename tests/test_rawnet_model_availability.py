import pytest
pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("soundfile")

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def rawnet_app(monkeypatch):
    monkeypatch.setenv('RAWNET_MODEL_PATH', '/tmp/does-not-exist.pth')
    sys.path.insert(0, str(Path('rawnet-service').resolve()))
    import importlib
    mod = importlib.import_module('main')
    importlib.reload(mod)
    return mod.app


def test_health_reports_model_missing(rawnet_app):
    with TestClient(rawnet_app) as client:
        r = client.get('/health')
        assert r.status_code == 200
        data = r.json()
        assert data['model_ready'] is False


def test_predict_returns_503_when_model_missing(rawnet_app):
    with TestClient(rawnet_app) as client:
        r = client.post('/predict', json={'base64_audio': 'QQ=='})
        assert r.status_code == 503
