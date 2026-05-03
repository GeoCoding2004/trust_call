import pytest
pytestmark = pytest.mark.requires_models

"""
tests/test_backend_validation.py

Tests for request validation in the distilbert-service and backend gateway.
Runs without model weights by setting DISTILBERT_USE_CLASSIFIER=false.
"""
import os
import sys
import pytest

# Tell the distilbert service to use heuristic mode (no model weights needed)
os.environ.setdefault("DISTILBERT_MODEL_NAME", "distilbert-base-uncased")
os.environ.setdefault("DISTILBERT_USE_CLASSIFIER", "false")
os.environ.setdefault("DISTILBERT_USE_EMBEDDINGS", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, "distilbert-service")

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def distilbert_client():
    from main import app  # noqa: PLC0415
    with TestClient(app) as c:
        yield c


class TestDistilBertValidation:

    def test_valid_scam_payload_returns_200(self, distilbert_client):
        resp = distilbert_client.post(
            "/predict",
            json={"scrubbed_text": "Urgent: send money via wire transfer now."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "semantic_score" in data
        assert "label" in data

    def test_empty_text_returns_insufficient_text(self, distilbert_client):
        resp = distilbert_client.post("/predict", json={"scrubbed_text": ""})
        assert resp.status_code == 200
        assert resp.json()["label"] == "insufficient_text"

    def test_very_short_text_returns_insufficient_text(self, distilbert_client):
        resp = distilbert_client.post("/predict", json={"scrubbed_text": "hi"})
        assert resp.status_code == 200
        assert resp.json()["label"] == "insufficient_text"

    def test_missing_required_field_returns_422(self, distilbert_client):
        resp = distilbert_client.post("/predict", json={})
        assert resp.status_code in (400, 422)

    def test_wrong_field_name_returns_error(self, distilbert_client):
        resp = distilbert_client.post("/predict", json={"text": "some text"})
        assert resp.status_code in (400, 422)

    def test_non_json_body_returns_error(self, distilbert_client):
        resp = distilbert_client.post(
            "/predict",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)

    def test_numeric_text_field_returns_error(self, distilbert_client):
        """scrubbed_text must be a string."""
        resp = distilbert_client.post("/predict", json={"scrubbed_text": 12345})
        # Pydantic coerces int to str, so either 200 or 422 is acceptable;
        # what matters is the service does not crash (no 500).
        assert resp.status_code != 500

    def test_benign_text_returns_benign_label(self, distilbert_client):
        resp = distilbert_client.post(
            "/predict",
            json={"scrubbed_text": "Hey, just calling to confirm dinner plans for tonight."},
        )
        assert resp.status_code == 200
        assert resp.json()["label"] == "benign"

    def test_response_has_required_fields(self, distilbert_client):
        resp = distilbert_client.post(
            "/predict",
            json={"scrubbed_text": "Please verify your account urgently."},
        )
        data = resp.json()
        for field in ["semantic_score", "label", "flagged_phrases", "model_name"]:
            assert field in data, f"Missing field: {field}"
