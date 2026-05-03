import asyncio
import base64
import io
import os
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Keep model downloads inside the project instead of the Windows user cache,
# which can be locked down on some machines.
PROJECT_STATE_ROOT = Path(__file__).resolve().parent / "state"
os.environ.setdefault("HF_HOME", str(PROJECT_STATE_ROOT / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_STATE_ROOT / "huggingface" / "hub"))

import numpy as np
import soundfile as sf
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from trust_call_backend.fusion import build_fusion_status
except ModuleNotFoundError:
    from fusion import build_fusion_status  # type: ignore

from trust_call_backend.metrics_registry import MetricsRegistry, _escape_metric_label
from trust_call_backend.schemas import (
    IdentityEnrollmentPayload,
    IdentityIdentificationPayload,
    IdentityVerificationPayload,
    LiveIdentityValidationPayload,
    LiveSessionEnrollmentPayload,
    Offer,
)
from trust_call_backend.service_clients import fetch_distilbert_prediction, fetch_rawnet_prediction

try:
    from trust_call_backend.fusion import build_fusion_status
except ModuleNotFoundError:
    from fusion import build_fusion_status  # type: ignore

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    WhisperModel = None  # type: ignore[assignment]

try:
    from trust_call_backend.identity_config import load_identity_auditor_config
    from trust_call_backend.identity_auditor import (
        ECAPASpeakerEmbedder,
        IdentityAuditor,
        IdentityEnrollmentStore,
        IdentityPolicyError,
        IdentityResult,
        decode_base64_audio,
    )
except ModuleNotFoundError:
    from identity_config import load_identity_auditor_config  # type: ignore
    from identity_auditor import (  # type: ignore
        ECAPASpeakerEmbedder,
        IdentityAuditor,
        IdentityEnrollmentStore,
        IdentityPolicyError,
        IdentityResult,
        decode_base64_audio,
    )


RAWNET_URL = os.getenv("RAWNET_URL", "http://127.0.0.1:8000/predict")
DISTILBERT_URL = os.getenv("DISTILBERT_URL", "http://127.0.0.1:8002/predict")
WEBRTC_STUN_URL = os.getenv("TRUST_CALL_WEBRTC_STUN_URL", "stun:stun.l.google.com:19302")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "tiny.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SIGNAL_MIN_RMS = float(os.getenv("TRUST_CALL_SIGNAL_MIN_RMS", "0.0025"))
SIGNAL_MIN_PEAK = float(os.getenv("TRUST_CALL_SIGNAL_MIN_PEAK", "0.02"))
SIGNAL_MIN_ACTIVE_RATIO = float(os.getenv("TRUST_CALL_SIGNAL_MIN_ACTIVE_RATIO", "0.015"))
SEMANTIC_MIN_CONTEXT_SECONDS = float(os.getenv("TRUST_CALL_SEMANTIC_CONTEXT_SECONDS", "8.0"))
SEMANTIC_MAX_CONTEXT_SECONDS = float(os.getenv("TRUST_CALL_SEMANTIC_MAX_CONTEXT_SECONDS", "12.0"))
SEMANTIC_MIN_CHARS = int(os.getenv("TRUST_CALL_SEMANTIC_MIN_CHARS", "20"))

app = FastAPI(title="Trust-Call WebRTC Gateway", version="1.0")


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ORIGINS = _parse_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


metrics = MetricsRegistry()


class Offer(BaseModel):
    sdp: str = Field(..., min_length=1, max_length=200_000)
    type: str = Field(..., min_length=1, max_length=32)
    caller_id: str = Field(default="unknown", max_length=128)


class IdentityEnrollmentPayload(BaseModel):
    caller_id: str = Field(..., min_length=1, max_length=128)
    base64_audio: str = Field(..., min_length=1, max_length=12_000_000)
    allow_update: bool = False
    ema_alpha: float | None = None
    safe_to_enroll: bool = False
    safe_to_update: bool = False
    synthetic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    coercion_score: float | None = Field(default=None, ge=0.0, le=1.0)


class IdentityVerificationPayload(BaseModel):
    caller_id: str = Field(..., min_length=1, max_length=128)
    base64_audio: str = Field(..., min_length=1, max_length=12_000_000)


class IdentityIdentificationPayload(BaseModel):
    base64_audio: str = Field(..., min_length=1, max_length=12_000_000)
    claimed_caller_id: str = Field(default="unknown", max_length=128)
    top_k: int = Field(default=3, ge=1, le=10)


class LiveIdentityValidationPayload(BaseModel):
    caller_id: str = Field(..., min_length=1, max_length=128)
    base64_audio: str = Field(..., min_length=1, max_length=12_000_000)
    session_id: str | None = Field(default=None, max_length=128)
    dispatch_signal_auditor: bool = False
    identify_if_unenrolled: bool = False
    top_k: int = Field(default=3, ge=1, le=10)


class LiveSessionEnrollmentPayload(BaseModel):
    safe_to_enroll: bool = True
    synthetic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    coercion_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            print("WARNING: telemetry emitted without an attached WebSocket client.")
            return

        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as exc:
                print(f"Failed to send websocket message: {exc}")


class LiveIdentityMonitor:
    def __init__(self, max_sessions: int = 20, max_events_per_session: int = 8):
        self.max_sessions = max_sessions
        self.max_events_per_session = max_events_per_session
        self._sessions: dict[str, dict] = {}
        self._order: deque[str] = deque()

    def start_session(self, caller_id: str, source: str) -> dict:
        session_id = str(uuid4())
        now = _utc_now()
        metrics.inc("trust_call_gateway_sessions_started_total", source=source)
        session = {
            "session_id": session_id,
            "caller_id": caller_id,
            "source": source,
            "state": "offer_received" if source == "webrtc" else "validation_started",
            "created_at_utc": now,
            "last_updated_at_utc": now,
            "track_connected_at_utc": None,
            "track_kind": None,
            "sample_rate_hz": None,
            "frames_received": 0,
            "last_frame_at_utc": None,
            "last_frame_shape": None,
            "buffered_duration_seconds": 0.0,
            "chunks_processed": 0,
            "last_chunk_peak": None,
            "last_chunk_duration_seconds": None,
            "last_identity_result": None,
            "recent_identity_events": [],
            "last_signal_score": None,
            "last_signal_threat": False,
            "last_semantic_intent": None,
            "last_semantic_score": None,
            "last_semantic_label": None,
            "last_transcript": "",
            "last_fusion_status": "WAITING",
            "last_error": None,
            "candidate_embeddings": [],
            "candidate_embedding_metadata": [],
            "candidate_status": "idle",
            "candidate_enrollment_result": None,
        }
        self._sessions[session_id] = session
        self._order.append(session_id)
        while len(self._order) > self.max_sessions:
            evicted = self._order.popleft()
            self._sessions.pop(evicted, None)
        return session

    def mark_track_connected(self, session_id: str, track_kind: str | None = None) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session["track_connected_at_utc"] = _utc_now()
        session["track_kind"] = track_kind
        session["state"] = "track_connected"
        session["last_updated_at_utc"] = session["track_connected_at_utc"]

    def record_frame(
        self,
        session_id: str,
        sample_rate: int,
        frame_shape: tuple[int, ...],
        buffered_duration_seconds: float,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        now = _utc_now()
        session["frames_received"] = int(session["frames_received"]) + 1
        metrics.inc("trust_call_gateway_audio_frames_total")
        session["sample_rate_hz"] = sample_rate
        session["last_frame_at_utc"] = now
        session["last_frame_shape"] = list(frame_shape)
        session["buffered_duration_seconds"] = round(float(buffered_duration_seconds), 3)
        session["state"] = "receiving_audio"
        session["last_updated_at_utc"] = now

    def record_chunk(
        self,
        session_id: str,
        identity_result: IdentityResult,
        sample_rate: int,
        chunk_peak: float,
    ) -> dict | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None

        now = _utc_now()
        chunk_index = int(session["chunks_processed"]) + 1
        metrics.inc("trust_call_gateway_audio_chunks_total")
        metrics.inc("trust_call_iep3_identity_results_total", status=identity_result.status)
        event = {
            "chunk_index": chunk_index,
            "processed_at_utc": now,
            "sample_rate_hz": sample_rate,
            "chunk_peak": round(float(chunk_peak), 4),
            "candidate_embedding_count": len(session["candidate_embeddings"]),
            "candidate_enrollment_ready": bool(session["candidate_embeddings"]),
            **identity_result.to_api_response(),
        }
        session["chunks_processed"] = chunk_index
        session["sample_rate_hz"] = sample_rate
        session["last_chunk_peak"] = event["chunk_peak"]
        session["last_chunk_duration_seconds"] = event["duration_seconds"]
        session["last_identity_result"] = event
        session["state"] = (
            "candidate_collecting"
            if identity_result.status == "candidate_collecting"
            else "identity_verified"
        )
        session["last_updated_at_utc"] = now

        recent_events = session["recent_identity_events"]
        recent_events.append(event)
        if len(recent_events) > self.max_events_per_session:
            del recent_events[0 : len(recent_events) - self.max_events_per_session]
        return event

    def record_candidate_embedding(
        self,
        session_id: str,
        embedding: np.ndarray,
        metadata: dict[str, float],
    ) -> int:
        session = self._sessions.get(session_id)
        if session is None:
            return 0

        embeddings = session["candidate_embeddings"]
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        metrics.inc("trust_call_iep3_candidate_embeddings_total")
        session["candidate_embedding_metadata"].append(dict(metadata))
        session["candidate_status"] = "collecting"
        session["last_updated_at_utc"] = _utc_now()
        return len(embeddings)

    def get_candidate_embeddings(
        self,
        session_id: str,
    ) -> tuple[str, list[np.ndarray], list[dict[str, float]]] | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return (
            session["caller_id"],
            list(session["candidate_embeddings"]),
            list(session["candidate_embedding_metadata"]),
        )

    def mark_candidate_enrolled(self, session_id: str, enrollment: dict) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session["candidate_status"] = "enrolled"
        metrics.inc("trust_call_iep3_candidate_enrollments_total", status="enrolled")
        session["candidate_enrollment_result"] = enrollment
        session["candidate_embeddings"] = []
        session["candidate_embedding_metadata"] = []
        session["state"] = "candidate_enrolled"
        session["last_updated_at_utc"] = _utc_now()

    def discard_candidate_embeddings(self, session_id: str) -> int | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        discarded = len(session["candidate_embeddings"])
        if discarded:
            metrics.inc("trust_call_iep3_candidate_enrollments_total", status="discarded")
        session["candidate_embeddings"] = []
        session["candidate_embedding_metadata"] = []
        session["candidate_status"] = "discarded" if discarded else "idle"
        session["last_updated_at_utc"] = _utc_now()
        return discarded

    def record_fusion_result(
        self,
        session_id: str,
        signal_score: str,
        is_threat: bool,
        semantic_intent: str,
        semantic_score: float,
        semantic_label: str,
        transcript: str,
        fusion_status: str,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session["last_signal_score"] = signal_score
        session["last_signal_threat"] = bool(is_threat)
        session["last_semantic_intent"] = semantic_intent
        session["last_semantic_score"] = round(float(semantic_score), 4)
        session["last_semantic_label"] = semantic_label
        session["last_transcript"] = transcript
        session["last_fusion_status"] = fusion_status
        metrics.inc("trust_call_gateway_fusion_results_total", status=fusion_status)
        session["state"] = "telemetry_broadcast"
        session["last_updated_at_utc"] = _utc_now()

    def record_error(self, session_id: str, error: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session["last_error"] = error
        metrics.inc("trust_call_gateway_errors_total", source=session.get("source", "unknown"))
        session["state"] = "error"
        session["last_updated_at_utc"] = _utc_now()

    def list_sessions(self) -> list[dict]:
        sessions = [self._sessions[session_id] for session_id in reversed(self._order)]
        return [self._summary(session) for session in sessions]

    def active_session_count(self) -> int:
        return len(self._sessions)

    def get_session(self, session_id: str) -> dict | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            **self._summary(session),
            "last_signal_score": session["last_signal_score"],
            "last_signal_threat": session["last_signal_threat"],
            "last_semantic_intent": session["last_semantic_intent"],
            "last_semantic_score": session["last_semantic_score"],
            "last_semantic_label": session["last_semantic_label"],
            "last_transcript": session["last_transcript"],
            "last_fusion_status": session["last_fusion_status"],
            "last_error": session["last_error"],
            "recent_identity_events": list(session["recent_identity_events"]),
        }

    def _summary(self, session: dict) -> dict:
        return {
            "session_id": session["session_id"],
            "caller_id": session["caller_id"],
            "source": session["source"],
            "state": session["state"],
            "created_at_utc": session["created_at_utc"],
            "last_updated_at_utc": session["last_updated_at_utc"],
            "track_connected_at_utc": session["track_connected_at_utc"],
            "track_kind": session["track_kind"],
            "sample_rate_hz": session["sample_rate_hz"],
            "frames_received": session["frames_received"],
            "last_frame_at_utc": session["last_frame_at_utc"],
            "last_frame_shape": session["last_frame_shape"],
            "buffered_duration_seconds": session["buffered_duration_seconds"],
            "chunks_processed": session["chunks_processed"],
            "last_chunk_peak": session["last_chunk_peak"],
            "last_chunk_duration_seconds": session["last_chunk_duration_seconds"],
            "last_identity_result": session["last_identity_result"],
            "candidate_embedding_count": len(session["candidate_embeddings"]),
            "candidate_enrollment_ready": bool(session["candidate_embeddings"]),
            "candidate_status": session["candidate_status"],
            "candidate_enrollment_result": session["candidate_enrollment_result"],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_whisper_model():
    if WhisperModel is None:
        print("Whisper unavailable: faster_whisper is not installed.")
        return None
    try:
        print(f"Loading Whisper STT model: {WHISPER_MODEL_NAME}")
        model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("Whisper loaded.")
        return model
    except Exception as exc:
        print(f"Whisper unavailable: {exc}")
        return None


manager = ConnectionManager()
live_identity_monitor = LiveIdentityMonitor()
peer_connections: set[RTCPeerConnection] = set()
state_root = PROJECT_STATE_ROOT
identity_config = load_identity_auditor_config()
identity_store = IdentityEnrollmentStore(state_root / "identity_profiles")
identity_auditor = IdentityAuditor(
    store=identity_store,
    config=identity_config,
    embedder=ECAPASpeakerEmbedder(
        config=identity_config,
        savedir=identity_config.resolve_model_savedir(),
    ),
)


@app.get("/metrics")
async def metrics_endpoint():
    try:
        profile_count = float(len(identity_store.load_all()))
    except Exception:
        profile_count = 0.0
        metrics.inc("trust_call_gateway_errors_total", source="metrics")

    body = metrics.render(
        gauges={
            "trust_call_gateway_active_sessions": float(live_identity_monitor.active_session_count()),
            "trust_call_iep3_enrolled_profiles": profile_count,
        }
    )
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


whisper_model = _load_whisper_model()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("Incoming WebSocket connection.")
    await manager.connect(websocket)
    print("WebSocket accepted.")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket client disconnected.")
    except Exception as exc:
        print(f"WebSocket crashed: {exc}")


@app.get("/identity/enrollment/{caller_id}")
async def get_identity_enrollment_status(caller_id: str):
    return identity_auditor.get_enrollment_status(caller_id)


@app.get("/identity/config")
async def get_identity_config():
    return identity_auditor.get_policy_snapshot()


@app.get("/identity/live/sessions")
async def list_live_identity_sessions():
    sessions = live_identity_monitor.list_sessions()
    return {
        "count": len(sessions),
        "sessions": sessions,
    }


@app.get("/identity/live/sessions/{session_id}")
async def get_live_identity_session(session_id: str):
    session = live_identity_monitor.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Live identity session not found")
    return session


@app.post("/identity/live/sessions/{session_id}/enroll-candidate")
async def enroll_live_identity_candidate(
    session_id: str,
    payload: LiveSessionEnrollmentPayload,
):
    candidate = live_identity_monitor.get_candidate_embeddings(session_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Live identity session not found")

    caller_id, embeddings, metadata = candidate
    try:
        enrollment = identity_auditor.enroll_from_embeddings(
            caller_id=caller_id,
            embeddings=embeddings,
            metadata=metadata,
            safe_to_enroll=payload.safe_to_enroll,
        )
    except IdentityPolicyError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.to_response()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    live_identity_monitor.mark_candidate_enrolled(session_id, enrollment)
    return {
        "session_id": session_id,
        "caller_id": caller_id,
        "enrollment": enrollment,
        "session": live_identity_monitor.get_session(session_id),
    }


@app.delete("/identity/live/sessions/{session_id}/candidate")
async def discard_live_identity_candidate(session_id: str):
    discarded = live_identity_monitor.discard_candidate_embeddings(session_id)
    if discarded is None:
        raise HTTPException(status_code=404, detail="Live identity session not found")
    return {
        "session_id": session_id,
        "discarded_candidate_embedding_count": discarded,
        "session": live_identity_monitor.get_session(session_id),
    }


@app.post("/identity/enroll")
async def enroll_identity(payload: IdentityEnrollmentPayload):
    try:
        return identity_auditor.enroll_from_base64(
            caller_id=payload.caller_id,
            base64_audio=payload.base64_audio,
            allow_update=payload.allow_update,
            ema_alpha=payload.ema_alpha,
            safe_to_enroll=payload.safe_to_enroll,
            safe_to_update=payload.safe_to_update,
            synthetic_score=payload.synthetic_score,
            coercion_score=payload.coercion_score,
        )
    except IdentityPolicyError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.to_response()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/identity/verify")
async def verify_identity(payload: IdentityVerificationPayload):
    try:
        result = identity_auditor.verify_from_base64(
            caller_id=payload.caller_id,
            base64_audio=payload.base64_audio,
        )
        return result.to_api_response()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/identity/identify")
async def identify_identity(payload: IdentityIdentificationPayload):
    try:
        result = identity_auditor.identify_from_base64(
            base64_audio=payload.base64_audio,
            claimed_caller_id=payload.claimed_caller_id,
            top_k=payload.top_k,
        )
        return result.to_api_response()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/identity/live/validate")
async def validate_live_identity_chunk(payload: LiveIdentityValidationPayload):
    try:
        session = (
            live_identity_monitor.get_session(payload.session_id)
            if payload.session_id
            else None
        )
        if session is None:
            session = live_identity_monitor.start_session(
                caller_id=payload.caller_id,
                source="live_validation",
            )

        audio, sample_rate = decode_base64_audio(payload.base64_audio)
        identity_result = await process_identity_chunk(
            caller_id=payload.caller_id,
            audio=audio,
            sample_rate=sample_rate,
            session_id=session["session_id"],
            broadcast_identity=True,
            identify_if_unenrolled=payload.identify_if_unenrolled,
            top_k=payload.top_k,
        )
        if payload.dispatch_signal_auditor:
            wav_io = io.BytesIO()
            sf.write(wav_io, audio, sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = wav_io.getvalue()
            base64_audio = base64.b64encode(wav_bytes).decode("utf-8")
            asyncio.create_task(
                orchestrate_late_fusion(
                    base64_audio=base64_audio,
                    text_context="",
                    identity_result=identity_result,
                    session_id=session["session_id"],
                )
            )
        return {
            "session_id": session["session_id"],
            "identity": identity_result.to_api_response(),
            "session": live_identity_monitor.get_session(session["session_id"]),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/identity/enrollment/{caller_id}")
async def delete_identity_enrollment(caller_id: str):
    deleted = identity_store.delete(caller_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    return {
        "caller_id": caller_id,
        "deleted": True,
    }


async def fetch_rawnet(base64_audio: str) -> float:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RAWNET_URL,
                json={"base64_audio": base64_audio},
                timeout=5.0,
            )
            if response.status_code == 200:
                data = response.json()
                return float(data.get("spoof_probability_percent", 0.0))
            print(f"RawNet service error: {response.status_code} {response.text}")
    except Exception as exc:
        print(f"RawNet service unavailable: {exc}")
    return 0.0


async def fetch_distilbert(text: str) -> dict:
    if not text.strip():
        return {"semantic_score": 0.0, "label": "insufficient_text"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                DISTILBERT_URL,
                json={"scrubbed_text": text},
                timeout=5.0,
            )
            if response.status_code == 200:
                return response.json()
            print(f"DistilBERT service error: {response.status_code} {response.text}")
    except Exception as exc:
        print(f"DistilBERT service unavailable: {exc}")
    return {"semantic_score": 0.0, "label": "semantic_unavailable"}


async def orchestrate_late_fusion(
    base64_audio: str,
    text_context: str,
    identity_result: IdentityResult,
    session_id: str,
    signal_quality: dict | None = None,
    semantic_context_seconds: float = 0.0,
) -> None:
    signal_quality = signal_quality or {"usable": True, "reason": "not_measured"}
    rawnet_task = (
        fetch_rawnet_prediction(base64_audio, RAWNET_URL)
        if signal_quality.get("usable", True)
        else _immediate_float(0.0)
    )
    distilbert_task = (
        fetch_distilbert_prediction(text_context, DISTILBERT_URL)
        if _semantic_context_ready(text_context, semantic_context_seconds)
        else _immediate_dict(_semantic_waiting_result(text_context, semantic_context_seconds))
    )
    synthetic_score, distilbert_data = await asyncio.gather(rawnet_task, distilbert_task)

    semantic_score = float(distilbert_data.get("semantic_score", 0.0))
    semantic_label = str(distilbert_data.get("label", "benign"))
    real_score = max(0.0, 100.0 - synthetic_score)
    if signal_quality.get("usable", True):
        signal_score = (
            f"{synthetic_score:.2f}% AI (Deepfake)"
            if synthetic_score > 50.0
            else f"{real_score:.2f}% Human"
        )
    else:
        signal_score = "Need clearer speech"
    semantic_intent = f"{semantic_label.upper()} ({semantic_score:.2f})"
    fusion_status, is_threat = build_fusion_status(
        identity_result=identity_result,
        synthetic_score=synthetic_score,
        semantic_score=semantic_score,
    )

    live_identity_monitor.record_fusion_result(
        session_id=session_id,
        signal_score=signal_score,
        is_threat=is_threat,
        semantic_intent=semantic_intent,
        semantic_score=semantic_score,
        semantic_label=semantic_label,
        transcript=text_context,
        fusion_status=fusion_status,
    )
    session = live_identity_monitor.get_session(session_id)
    candidate_embedding_count = (
        session["candidate_embedding_count"] if session is not None else 0
    )
    await manager.broadcast(
        {
            "session_id": session_id,
            "caller_id": identity_result.caller_id,
            "signal_score": signal_score,
            "is_threat": is_threat,
            "semantic_intent": semantic_intent,
            "semantic_score": round(semantic_score, 4),
            "semantic_label": semantic_label,
            "transcript_context": text_context,
            "semantic_context_seconds": round(float(semantic_context_seconds), 2),
            "signal_quality": signal_quality,
            "fusion_status": fusion_status,
            "identity_chunk_count": (
                session["chunks_processed"] if session is not None else None
            ),
            "candidate_embedding_count": candidate_embedding_count,
            "candidate_enrollment_ready": candidate_embedding_count > 0,
            **identity_result.to_telemetry(),
        }
    )


async def _immediate_float(value: float) -> float:
    return value


async def _immediate_dict(value: dict) -> dict:
    return value


def _semantic_context_ready(text_context: str, context_seconds: float) -> bool:
    return (
        context_seconds >= SEMANTIC_MIN_CONTEXT_SECONDS
        and len(text_context.strip()) >= SEMANTIC_MIN_CHARS
    )


def _semantic_waiting_result(text_context: str, context_seconds: float) -> dict:
    if not text_context.strip():
        return {"semantic_score": 0.0, "label": "need_speech"}
    return {
        "semantic_score": 0.0,
        "label": f"building_context_{context_seconds:.0f}s",
    }


async def broadcast_identity_telemetry(identity_result: IdentityResult, session_id: str) -> None:
    session = live_identity_monitor.get_session(session_id)
    candidate_embedding_count = (
        session["candidate_embedding_count"] if session is not None else 0
    )
    await manager.broadcast(
        {
            "session_id": session_id,
            "caller_id": identity_result.caller_id,
            "signal_score": "Analyzing...",
            "is_threat": False,
            "semantic_intent": "Transcribing...",
            "fusion_status": "ANALYZING",
            "identity_chunk_count": (
                session["chunks_processed"] if session is not None else None
            ),
            "candidate_embedding_count": candidate_embedding_count,
            "candidate_enrollment_ready": candidate_embedding_count > 0,
            **identity_result.to_telemetry(),
        }
    )


async def process_identity_chunk(
    caller_id: str,
    audio: np.ndarray,
    sample_rate: int,
    session_id: str,
    broadcast_identity: bool = True,
    identify_if_unenrolled: bool = False,
    top_k: int = 3,
) -> IdentityResult:
    identity_result = identity_auditor.verify_chunk(
        caller_id=caller_id,
        audio=audio,
        sample_rate=sample_rate,
    )
    if identify_if_unenrolled and identity_result.status in {
        "missing_caller_id",
        "not_enrolled",
        "profile_incompatible",
    }:
        identity_result = identity_auditor.identify_chunk(
            audio=audio,
            sample_rate=sample_rate,
            claimed_caller_id=caller_id,
            top_k=top_k,
        )
    elif identity_result.status == "not_enrolled":
        try:
            embedding, metadata = identity_auditor.extract_candidate_embedding(
                audio=audio,
                sample_rate=sample_rate,
            )
            candidate_count = live_identity_monitor.record_candidate_embedding(
                session_id=session_id,
                embedding=embedding,
                metadata=metadata,
            )
            identity_result = IdentityResult(
                caller_id=caller_id,
                status="candidate_collecting",
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text=f"Collecting TOFU ({candidate_count})",
                reason="temporary_embeddings_buffered_until_user_confirmation",
                enrolled=False,
                duration_seconds=identity_result.duration_seconds,
                identification_mode="tofu_candidate",
                identified_caller_id=None,
                candidate_count=candidate_count,
            )
        except ValueError as exc:
            reason = str(exc)
            display_text = (
                "Need More Speech" if reason == "insufficient_audio" else "Speech Too Quiet"
            )
            identity_result = IdentityResult(
                caller_id=caller_id,
                status="candidate_waiting_for_speech",
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text=display_text,
                reason=f"tofu_candidate_{reason}",
                enrolled=False,
                duration_seconds=identity_result.duration_seconds,
                identification_mode="tofu_candidate",
                identified_caller_id=None,
                candidate_count=0,
            )
            print(f"Skipped TOFU candidate chunk: {reason}")
        except RuntimeError as exc:
            print(f"Unable to collect TOFU candidate chunk: {exc}")

    chunk_peak = float(np.max(np.abs(np.asarray(audio)))) if np.asarray(audio).size else 0.0
    live_identity_monitor.record_chunk(
        session_id=session_id,
        identity_result=identity_result,
        sample_rate=sample_rate,
        chunk_peak=chunk_peak,
    )
    if broadcast_identity:
        await broadcast_identity_telemetry(identity_result, session_id)
    return identity_result


def sync_transcribe_wav_bytes(wav_bytes: bytes) -> str:
    if whisper_model is None:
        return ""

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(wav_bytes)
            temp_path = temp_file.name
        segments, _ = whisper_model.transcribe(temp_path, beam_size=1)
        return " ".join(segment.text for segment in segments).strip()
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def measure_audio_quality(audio: np.ndarray) -> dict[str, float | bool | str]:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return {
            "usable": False,
            "reason": "empty_audio",
            "rms": 0.0,
            "peak": 0.0,
            "active_ratio": 0.0,
        }

    if np.max(np.abs(samples)) > 1.5:
        samples = samples / 32768.0

    abs_samples = np.abs(samples)
    rms = float(np.sqrt(np.mean(np.square(samples))))
    peak = float(np.max(abs_samples))
    active_ratio = float(np.mean(abs_samples >= SIGNAL_MIN_PEAK))
    usable = (
        rms >= SIGNAL_MIN_RMS
        and peak >= SIGNAL_MIN_PEAK
        and active_ratio >= SIGNAL_MIN_ACTIVE_RATIO
    )
    if usable:
        reason = "usable_speech"
    elif rms < SIGNAL_MIN_RMS:
        reason = "low_rms"
    elif peak < SIGNAL_MIN_PEAK:
        reason = "low_peak"
    else:
        reason = "low_speech_activity"
    return {
        "usable": usable,
        "reason": reason,
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "active_ratio": round(active_ratio, 5),
    }


async def consume_audio_track(track, caller_id: str, session_id: str):
    print("Audio buffer engine started.")
    audio_buffer = []
    context_memory: deque[tuple[str, float]] = deque()
    sample_rate = 0
    target_seconds = 3.0
    overlap_ratio = 0.5
    chunk_counter = 0

    while True:
        try:
            frame = await track.recv()
            raw_audio = frame.to_ndarray()
            audio_array = _audio_frame_to_mono_int16(raw_audio, len(frame.layout.channels))

            if sample_rate == 0:
                sample_rate = frame.sample_rate
                print(f"Audio format locked at {sample_rate}Hz.")
                live_identity_monitor.mark_track_connected(
                    session_id,
                    track_kind=getattr(track, "kind", None),
                )

            audio_buffer.append(audio_array)
            total_samples = sum(arr.shape[1] for arr in audio_buffer)
            current_duration = total_samples / sample_rate
            live_identity_monitor.record_frame(
                session_id=session_id,
                sample_rate=sample_rate,
                frame_shape=tuple(raw_audio.shape),
                buffered_duration_seconds=current_duration,
            )

            if current_duration >= target_seconds:
                chunk_counter += 1
                combined_audio = np.concatenate(audio_buffer, axis=1)
                chunk_audio = combined_audio.T
                max_volume = np.max(np.abs(chunk_audio))
                signal_quality = measure_audio_quality(chunk_audio)
                print(
                    "Dispatching full pipeline chunk "
                    f"{chunk_counter} | peak={float(max_volume):.2f} "
                    f"| rms={signal_quality['rms']} | active={signal_quality['active_ratio']} "
                    f"| quality={signal_quality['reason']}"
                )

                wav_io = io.BytesIO()
                sf.write(wav_io, chunk_audio, sample_rate, format="WAV", subtype="PCM_16")
                wav_bytes = wav_io.getvalue()
                base64_audio = base64.b64encode(wav_bytes).decode("utf-8")

                if signal_quality["usable"]:
                    transcription = await asyncio.to_thread(sync_transcribe_wav_bytes, wav_bytes)
                    print(f"🔥 WHISPER HEARD: '{transcription}'")
                    if transcription:
                        context_memory.append((transcription, target_seconds))
                else:
                    transcription = ""
                    print(f"Skipping RawNet/STT for weak audio: {signal_quality['reason']}")
                while sum(duration for _, duration in context_memory) > SEMANTIC_MAX_CONTEXT_SECONDS:
                    context_memory.popleft()
                semantic_context_seconds = sum(duration for _, duration in context_memory)
                recent_context = " ".join(text for text, _ in context_memory)

                identity_result = await process_identity_chunk(
                    caller_id=caller_id,
                    audio=chunk_audio,
                    sample_rate=sample_rate,
                    session_id=session_id,
                    broadcast_identity=True,
                )
                asyncio.create_task(
                    orchestrate_late_fusion(
                        base64_audio=base64_audio,
                        text_context=recent_context,
                        identity_result=identity_result,
                        session_id=session_id,
                        signal_quality=signal_quality,
                        semantic_context_seconds=semantic_context_seconds,
                    )
                )

                overlap_samples = max(int(sample_rate * target_seconds * overlap_ratio), 1)
                audio_buffer = [combined_audio[:, -overlap_samples:]]
        except Exception as exc:
            print(f"Audio stream ended or disconnected: {exc}")
            live_identity_monitor.record_error(session_id, str(exc))
            break


def _audio_frame_to_mono_int16(audio_array: np.ndarray, num_channels: int) -> np.ndarray:
    audio_array = np.asarray(audio_array)
    is_float = np.issubdtype(audio_array.dtype, np.floating)

    if audio_array.ndim == 1:
        mono = audio_array.reshape(1, -1)
    elif num_channels > 1:
        if audio_array.shape[0] == 1:
            reshaped = audio_array.reshape(-1, num_channels)
            mono = np.mean(reshaped, axis=1, keepdims=True).T
        else:
            mono = np.mean(audio_array, axis=0, keepdims=True)
    else:
        mono = audio_array.reshape(1, -1)

    if is_float:
        return np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
    return mono.astype(np.int16)


@app.post("/offer")
async def process_offer(params: Offer):
    print("Received WebRTC offer from Trust-Call.")
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)
    ice_servers = [RTCIceServer(urls=WEBRTC_STUN_URL)] if WEBRTC_STUN_URL else []
    pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
    peer_connections.add(pc)
    live_session = live_identity_monitor.start_session(
        caller_id=params.caller_id,
        source="webrtc",
    )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Peer connection state: {pc.connectionState}")
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            peer_connections.discard(pc)
            live_identity_monitor.record_error(
                live_session["session_id"],
                f"peer_connection_{pc.connectionState}",
            )

    @pc.on("track")
    def on_track(track):
        print("Live audio track connected.")
        asyncio.ensure_future(
            consume_audio_track(track, params.caller_id, live_session["session_id"])
        )

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print("Sending WebRTC answer back to mobile app.")
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "session_id": live_session["session_id"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
