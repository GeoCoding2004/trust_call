import pytest
from fastapi.testclient import TestClient
from main import app, normalize_text, extract_flagged_phrases

# Ensures lifespan models are loaded correctly without running a full server
client = TestClient(app)

def test_normalize_text_removes_extra_spaces():
    """Unit Test: Proves the isolated function normalizes text accurately."""
    assert normalize_text("   Hello     World  ") == "Hello World"

def test_normalize_text_handles_none():
    """Unit Test: Proves the isolated function doesn't crash on None."""
    assert normalize_text(None) == ""

def test_extract_flagged_phrases_catches_urgent_and_bank():
    """Unit Test: Proves Regex heuristic accurately detects known scam phrases in isolation."""
    phrases = extract_flagged_phrases("This is urgent and I need your bank details.")
    assert "urgent" in phrases
    assert "bank details" in phrases
    assert len(phrases) == 2

def test_extract_flagged_phrases_passes_safe_text():
    """Unit Test: Proves safe text returns zero flagged patterns."""
    phrases = extract_flagged_phrases("Hey Mom, how was your day today? I am eating lunch.")
    assert len(phrases) == 0

def test_predict_endpoint_returns_suspicious_for_scam():
    """Integration Test: Simulates an EEP Gateway payload and expects a 'suspicious' verdict."""
    payload = {"scrubbed_text": "Hey this is urgent, please send money via wire transfer right now to verify your account."}
    
    # We use 'with TestClient' to trigger the lifespan (loading the Warm Start model for testing)
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["label"] == "suspicious"
        assert "urgent" in data["flagged_phrases"]
        assert "send money" in data["flagged_phrases"]
        assert data["semantic_score"] >= 0.6

def test_predict_endpoint_returns_benign_for_safe_call():
    """Integration Test: Simulates safe conversation and expects 'benign'."""
    payload = {"scrubbed_text": "Just calling to check in on the dinner plans for tonight. See you soon."}
    
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["label"] == "benign"
        assert data["semantic_score"] < 0.6
        assert len(data["flagged_phrases"]) == 0
