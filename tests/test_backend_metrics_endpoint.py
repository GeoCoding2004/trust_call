"""Lightweight test that backend /metrics appends Prometheus-client output."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from trust_call_backend import observability_metrics


def _install_server_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    aiortc = types.ModuleType("aiortc")

    class _RTCConfiguration:
        def __init__(self, iceServers=None):
            self.iceServers = iceServers

    class _RTCIceServer:
        def __init__(self, urls=None):
            self.urls = urls

    class _RTCSessionDescription:
        def __init__(self, sdp: str, type: str):
            self.sdp = sdp
            self.type = type

    class _RTCPeerConnection:
        def __init__(self, *_args, **_kwargs):
            self.connectionState = "new"
            self.localDescription = types.SimpleNamespace(sdp="stub", type="answer")

        def on(self, _event: str):
            def _decorator(func):
                return func

            return _decorator

        async def setRemoteDescription(self, _offer):
            return None

        async def createAnswer(self):
            return self.localDescription

        async def setLocalDescription(self, answer):
            self.localDescription = answer

    aiortc.RTCConfiguration = _RTCConfiguration
    aiortc.RTCIceServer = _RTCIceServer
    aiortc.RTCPeerConnection = _RTCPeerConnection
    aiortc.RTCSessionDescription = _RTCSessionDescription
    monkeypatch.setitem(sys.modules, "aiortc", aiortc)

    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)

    identity_config = types.ModuleType("trust_call_backend.identity_config")

    class _Config:
        def resolve_model_savedir(self):
            return Path(".")

    identity_config.load_identity_auditor_config = lambda: _Config()
    monkeypatch.setitem(sys.modules, "trust_call_backend.identity_config", identity_config)

    identity_auditor = types.ModuleType("trust_call_backend.identity_auditor")

    class _IdentityPolicyError(ValueError):
        def __init__(self, code="stub", message="stub", http_status=400):
            super().__init__(message)
            self.code = code
            self.message = message
            self.http_status = http_status

        def to_response(self):
            return {"error": self.code, "message": self.message}

    class _IdentityResult:
        def __init__(
            self,
            caller_id="unknown",
            status="not_enrolled",
            identity_score=0.0,
            similarity=None,
            match_confidence=None,
            display_text="stub",
            reason=None,
            enrolled=False,
            duration_seconds=0.0,
            identification_mode="verification",
            identified_caller_id=None,
            candidate_count=0,
        ):
            self.caller_id = caller_id
            self.status = status
            self.identity_score = identity_score
            self.similarity = similarity
            self.match_confidence = match_confidence
            self.display_text = display_text
            self.reason = reason
            self.enrolled = enrolled
            self.duration_seconds = duration_seconds
            self.identification_mode = identification_mode
            self.identified_caller_id = identified_caller_id
            self.candidate_count = candidate_count

        def to_api_response(self):
            return {
                "caller_id": self.caller_id,
                "status": self.status,
                "identity_score": self.identity_score,
                "similarity": self.similarity,
                "match_confidence": self.match_confidence,
                "display_text": self.display_text,
                "reason": self.reason,
                "enrolled": self.enrolled,
                "duration_seconds": self.duration_seconds,
                "identification_mode": self.identification_mode,
                "identified_caller_id": self.identified_caller_id,
                "candidate_count": self.candidate_count,
                "candidates": [],
                "confidence": "none",
                "mismatch": self.identity_score,
                "eep": {"status": self.status},
            }

        def to_telemetry(self):
            return {"identity_status": self.status}

    class _IdentityEnrollmentStore:
        def __init__(self, _root):
            self._profiles = []

        def load_all(self):
            return list(self._profiles)

        def load(self, _caller_id):
            return None

        def delete(self, _caller_id):
            return False

    class _Embedder:
        def __init__(self, *args, **kwargs):
            pass

    class _IdentityAuditor:
        def __init__(self, *args, **kwargs):
            pass

        def get_enrollment_status(self, caller_id):
            return {"caller_id": caller_id, "enrolled": False}

        def get_policy_snapshot(self):
            return {"policy": "stub"}

        def enroll_from_embeddings(self, **kwargs):
            return {"caller_id": kwargs["caller_id"], "enrolled": True}

        def enroll_from_base64(self, **kwargs):
            return {"caller_id": kwargs["caller_id"], "enrolled": True}

        def verify_from_base64(self, caller_id, base64_audio):
            return _IdentityResult(caller_id=caller_id, duration_seconds=0.5)

        def identify_from_base64(self, base64_audio, claimed_caller_id, top_k):
            return _IdentityResult(caller_id=claimed_caller_id, duration_seconds=0.5)

        def verify_chunk(self, caller_id, audio, sample_rate):
            return _IdentityResult(caller_id=caller_id, duration_seconds=0.5)

        def identify_chunk(self, audio, sample_rate, claimed_caller_id, top_k):
            return _IdentityResult(
                caller_id=claimed_caller_id,
                status="unknown_speaker",
                duration_seconds=0.5,
            )

        def extract_candidate_embedding(self, audio, sample_rate):
            return np.zeros(3, dtype=np.float32), {"duration_seconds": 0.5, "rms": 0.1}

    def _decode_base64_audio(_base64_audio):
        return np.zeros(8000, dtype=np.float32), 16000

    identity_auditor.ECAPASpeakerEmbedder = _Embedder
    identity_auditor.IdentityAuditor = _IdentityAuditor
    identity_auditor.IdentityEnrollmentStore = _IdentityEnrollmentStore
    identity_auditor.IdentityPolicyError = _IdentityPolicyError
    identity_auditor.IdentityResult = _IdentityResult
    identity_auditor.decode_base64_audio = _decode_base64_audio
    monkeypatch.setitem(sys.modules, "trust_call_backend.identity_auditor", identity_auditor)

    faster_whisper = types.ModuleType("faster_whisper")

    class _WhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    faster_whisper.WhisperModel = _WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)


@pytest.fixture()
def backend_server_module(monkeypatch: pytest.MonkeyPatch):
    _install_server_import_stubs(monkeypatch)
    module_path = Path("trust_call_backend/server.py").resolve()
    spec = importlib.util.spec_from_file_location("test_backend_server_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metrics_endpoint_appends_prometheus_client_metrics(backend_server_module):
    observability_metrics.trust_call_whisper_model_ready.set(1)
    with TestClient(backend_server_module.app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "trust_call_gateway_active_sessions" in body
    assert "trust_call_whisper_model_ready" in body
    assert "trust_call_gateway_invalid_payload_total" in body
