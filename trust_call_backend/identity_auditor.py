from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    from trust_call_backend.identity_config import IdentityAuditorConfig
except ModuleNotFoundError:
    from identity_config import IdentityAuditorConfig  # type: ignore

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - dependency installed at runtime
    torch = None  # type: ignore[assignment]

try:
    from speechbrain.inference.speaker import EncoderClassifier as SpeechBrainEncoderClassifier
except ModuleNotFoundError:  # pragma: no cover - fallback for older SpeechBrain
    try:
        from speechbrain.pretrained import EncoderClassifier as SpeechBrainEncoderClassifier  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - dependency installed at runtime
        SpeechBrainEncoderClassifier = None  # type: ignore[assignment]

try:
    from speechbrain.utils.fetching import LocalStrategy
except ModuleNotFoundError:  # pragma: no cover - dependency installed at runtime
    LocalStrategy = None  # type: ignore[assignment]


EMBEDDER_NAME = "speechbrain_ecapa_tdnn_voxceleb"
ECAPA_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_IDENTITY_CONFIG = IdentityAuditorConfig()
TARGET_SAMPLE_RATE = DEFAULT_IDENTITY_CONFIG.target_sample_rate_hz
DEFAULT_MATCH_THRESHOLD = DEFAULT_IDENTITY_CONFIG.match_threshold
DEFAULT_REVIEW_THRESHOLD = DEFAULT_IDENTITY_CONFIG.review_threshold
DEFAULT_EMA_ALPHA = DEFAULT_IDENTITY_CONFIG.ema_alpha

IDENTITY_STATUS_CONTRACT = {
    "missing_caller_id": "No claimed identity was provided for the call.",
    "not_enrolled": "No local identity profile exists for the claimed caller.",
    "no_enrolled_profiles": "No local identity profiles exist for 1:N identification.",
    "insufficient_audio": "The speech segment is too short for reliable verification.",
    "low_energy": "The speech segment is too quiet for reliable verification.",
    "profile_incompatible": "The stored profile was created with a different embedder.",
    "model_unavailable": "The speaker verification model could not be loaded.",
    "candidate_collecting": "Temporary TOFU embeddings are being collected for this caller.",
    "candidate_waiting_for_speech": "TOFU enrollment is waiting for enough clear speech.",
    "match": "The current speaker matches the enrolled profile.",
    "review": "The current speaker is borderline and should be corroborated.",
    "mismatch": "The current speaker does not match the enrolled profile.",
    "identified": "The current speaker strongly matches a known local profile.",
    "identity_candidate": "The current speaker weakly matches a known local profile.",
    "unknown_speaker": "The current speaker does not match any local profile.",
}

IDENTITY_POLICY_CONTRACT = {
    "require_safe_call_confirmation_for_enrollment": (
        "Initial TOFU enrollment requires an explicitly trusted safe call."
    ),
    "require_safe_call_confirmation_for_update": (
        "EMA profile updates require an explicitly trusted safe call."
    ),
    "min_match_confidence_for_update": (
        "Profile updates require strong identity agreement before blending."
    ),
    "max_synthetic_score_for_update": (
        "Profile updates are blocked when synthetic-speech risk is too high."
    ),
    "max_coercion_score_for_update": (
        "Profile updates are blocked when coercion risk is too high."
    ),
    "existing_profile_requires_update_mode": (
        "Existing profiles cannot be overwritten unless update mode is explicitly requested."
    ),
}


class IdentityPolicyError(ValueError):
    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status

    def to_response(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
        }


@dataclass
class IdentityResult:
    caller_id: str
    status: str
    identity_score: float
    similarity: float | None
    match_confidence: float | None
    display_text: str
    reason: str | None
    enrolled: bool
    duration_seconds: float
    identification_mode: str = "verification"
    identified_caller_id: str | None = None
    candidate_count: int = 0
    candidates: list[dict[str, Any]] | None = None

    def confidence_level(self) -> str:
        if self.match_confidence is None:
            return "none"
        if self.status == "match":
            return "high"
        if self.status == "review":
            return "medium"
        if self.status == "mismatch":
            return "low"
        return "none"

    def to_eep_response(self) -> dict[str, Any]:
        similarity = (
            round(self.match_confidence, 6)
            if self.match_confidence is not None
            else None
        )
        mismatch = round(1.0 - self.match_confidence, 6) if similarity is not None else None
        return {
            "iep": "identity",
            "contact_id": self.caller_id,
            "status": self.status,
            "similarity": similarity,
            "mismatch": mismatch,
            "confidence": self.confidence_level(),
            "enrolled": self.enrolled,
            "reason": self.reason,
        }

    def to_telemetry(self) -> dict[str, Any]:
        return {
            "caller_id": self.caller_id,
            "identity_mode": self.identification_mode,
            "identity_identified_caller_id": self.identified_caller_id,
            "identity_candidate_count": self.candidate_count,
            "identity_candidates": self.candidates or [],
            "identity_score": round(self.identity_score, 4),
            "identity_similarity": (
                round(self.similarity, 6) if self.similarity is not None else None
            ),
            "identity_match_confidence": (
                round(self.match_confidence, 6)
                if self.match_confidence is not None
                else None
            ),
            "identity_duration_seconds": round(self.duration_seconds, 6),
            "identity_match": self.display_text,
            "identity_status": self.status,
            "identity_reason": self.reason,
            "identity_enrolled": self.enrolled,
            "identity_mismatch": round(self.identity_score, 6),
            "identity_confidence_level": self.confidence_level(),
            "identity_eep": self.to_eep_response(),
        }

    def to_api_response(self) -> dict[str, Any]:
        return {
            "caller_id": self.caller_id,
            "identification_mode": self.identification_mode,
            "identified_caller_id": self.identified_caller_id,
            "candidate_count": self.candidate_count,
            "candidates": self.candidates or [],
            "status": self.status,
            "identity_score": round(self.identity_score, 4),
            "similarity": round(self.similarity, 6) if self.similarity is not None else None,
            "match_confidence": round(self.match_confidence, 6)
            if self.match_confidence is not None
            else None,
            "mismatch": round(self.identity_score, 6),
            "confidence": self.confidence_level(),
            "display_text": self.display_text,
            "reason": self.reason,
            "enrolled": self.enrolled,
            "duration_seconds": round(self.duration_seconds, 6),
            "eep": self.to_eep_response(),
        }


class IdentityEnrollmentStore:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _caller_filename(self, caller_id: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", caller_id.strip())
        normalized = normalized.strip("._")
        if not normalized:
            raise ValueError("caller_id must contain at least one valid character")
        return f"{normalized}.json"

    def path_for(self, caller_id: str) -> Path:
        return self.root_dir / self._caller_filename(caller_id)

    def exists(self, caller_id: str) -> bool:
        return self.path_for(caller_id).is_file()

    def load(self, caller_id: str) -> dict[str, Any] | None:
        path = self.path_for(caller_id)
        if not path.is_file():
            return None

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_all(self) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for path in sorted(self.root_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                profiles.append(json.load(handle))
        return profiles

    def save(self, caller_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self.path_for(caller_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        return payload

    def delete(self, caller_id: str) -> bool:
        path = self.path_for(caller_id)
        if not path.is_file():
            return False
        path.unlink()
        return True


class ECAPASpeakerEmbedder:
    def __init__(
        self,
        config: IdentityAuditorConfig | None = None,
        target_sample_rate: int | None = None,
        model_source: str | None = None,
        savedir: Path | None = None,
        device: str | None = None,
    ):
        self.config = config or DEFAULT_IDENTITY_CONFIG
        self.target_sample_rate = target_sample_rate or self.config.target_sample_rate_hz
        self.model_source = model_source or self.config.model_source or ECAPA_MODEL_SOURCE
        self.savedir = savedir
        self.device = device
        self._classifier = None

    @property
    def name(self) -> str:
        return EMBEDDER_NAME

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier
        if torch is None or SpeechBrainEncoderClassifier is None:
            raise RuntimeError("speaker_model_dependencies_missing")
        if LocalStrategy is None:
            raise RuntimeError("speaker_model_dependencies_missing")

        run_device = self.device
        if run_device is None:
            run_device = "cuda" if torch.cuda.is_available() else "cpu"

        savedir = str(self.savedir) if self.savedir is not None else None
        self._classifier = SpeechBrainEncoderClassifier.from_hparams(
            source=self.model_source,
            savedir=savedir,
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": run_device},
        )
        return self._classifier

    def extract(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict[str, float]]:
        waveform = _prepare_waveform(audio, sample_rate, self.target_sample_rate)
        duration_seconds = len(waveform) / self.target_sample_rate
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0

        if duration_seconds < self.config.min_speech_seconds:
            raise ValueError("insufficient_audio")
        if rms < self.config.min_rms:
            raise ValueError("low_energy")
        classifier = self._get_classifier()
        signal = torch.from_numpy(waveform).float().unsqueeze(0)
        lengths = torch.tensor([1.0], dtype=torch.float32)
        embeddings = classifier.encode_batch(signal, lengths)
        embedding = embeddings.detach().cpu().numpy().reshape(-1).astype(np.float32)
        embedding = _l2_normalize(embedding)

        metadata = {
            "duration_seconds": round(duration_seconds, 6),
            "rms": round(rms, 8),
            "sample_rate_hz": float(self.target_sample_rate),
            "embedding_dim": float(embedding.shape[0]),
        }
        return embedding, metadata


class IdentityAuditor:
    def __init__(
        self,
        store: IdentityEnrollmentStore,
        config: IdentityAuditorConfig | None = None,
        embedder: ECAPASpeakerEmbedder | None = None,
        match_threshold: float | None = None,
        review_threshold: float | None = None,
    ):
        self.config = config or DEFAULT_IDENTITY_CONFIG
        match_threshold = self.config.match_threshold if match_threshold is None else match_threshold
        review_threshold = (
            self.config.review_threshold if review_threshold is None else review_threshold
        )
        if review_threshold >= match_threshold:
            raise ValueError("review_threshold must be lower than match_threshold")

        self.store = store
        self.embedder = embedder or ECAPASpeakerEmbedder(config=self.config)
        self.match_threshold = match_threshold
        self.review_threshold = review_threshold

    def get_enrollment_status(self, caller_id: str) -> dict[str, Any]:
        profile = self.store.load(caller_id)
        if profile is None:
            return {
                "caller_id": caller_id,
                "enrolled": False,
            }

        return {
            "caller_id": caller_id,
            "enrolled": True,
            "embedder": profile.get("embedder"),
            "embedding_dim": profile.get("embedding_dim"),
            "created_at_utc": profile.get("created_at_utc"),
            "updated_at_utc": profile.get("updated_at_utc"),
            "num_updates": profile.get("num_updates", 0),
            "last_duration_seconds": profile.get("last_duration_seconds"),
        }

    def enroll_from_base64(
        self,
        caller_id: str,
        base64_audio: str,
        allow_update: bool = False,
        ema_alpha: float | None = None,
        safe_to_enroll: bool = False,
        safe_to_update: bool = False,
        synthetic_score: float | None = None,
        coercion_score: float | None = None,
    ) -> dict[str, Any]:
        if ema_alpha is None:
            ema_alpha = self.config.ema_alpha
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in the interval (0, 1]")
        synthetic_score = _validate_optional_unit_interval("synthetic_score", synthetic_score)
        coercion_score = _validate_optional_unit_interval("coercion_score", coercion_score)

        existing_profile = self.store.load(caller_id)
        existing_is_compatible = (
            existing_profile is not None
            and existing_profile.get("embedder") == self.embedder.name
        )

        mode = "initial_enrollment"
        pre_update_match_confidence = None

        if existing_profile is None:
            if allow_update:
                raise IdentityPolicyError(
                    code="update_requires_existing_profile",
                    message="EMA update requested but no existing profile was found.",
                    http_status=409,
                )
            if self.config.require_safe_call_confirmation_for_enrollment and not safe_to_enroll:
                raise IdentityPolicyError(
                    code="tofu_requires_safe_call_confirmation",
                    message="TOFU enrollment is blocked until the call is explicitly marked safe.",
                    http_status=403,
                )
        else:
            if not allow_update:
                raise IdentityPolicyError(
                    code="identity_profile_already_exists",
                    message="A profile already exists for this caller. Delete it first or use update mode.",
                    http_status=409,
                )
            if not existing_is_compatible:
                raise IdentityPolicyError(
                    code="profile_incompatible_reenroll_required",
                    message="The stored profile was created with a different embedder. Re-enroll after deleting the old profile.",
                    http_status=409,
                )
            if self.config.require_safe_call_confirmation_for_update and not safe_to_update:
                raise IdentityPolicyError(
                    code="ema_update_requires_safe_call_confirmation",
                    message="EMA update is blocked until the call is explicitly marked safe.",
                    http_status=403,
                )
            if (
                synthetic_score is not None
                and synthetic_score > self.config.max_synthetic_score_for_update
            ):
                raise IdentityPolicyError(
                    code="ema_update_blocked_by_synthetic_risk",
                    message="EMA update is blocked because the synthetic-speech risk is too high.",
                    http_status=403,
                )
            if (
                coercion_score is not None
                and coercion_score > self.config.max_coercion_score_for_update
            ):
                raise IdentityPolicyError(
                    code="ema_update_blocked_by_coercion_risk",
                    message="EMA update is blocked because the coercion risk is too high.",
                    http_status=403,
                )

        audio, sample_rate = decode_base64_audio(base64_audio)
        embedding, metadata = self.embedder.extract(audio, sample_rate)
        now = _utc_now()

        if existing_profile is None:
            embedding_to_store = embedding.tolist()
            created_at = now
            num_updates = 0
        else:
            previous = _l2_normalize(np.array(existing_profile["embedding"], dtype=np.float32))
            pre_similarity = float(np.dot(previous, embedding))
            pre_update_match_confidence = float(np.clip((pre_similarity + 1.0) / 2.0, 0.0, 1.0))
            if pre_update_match_confidence < self.config.min_match_confidence_for_update:
                raise IdentityPolicyError(
                    code="ema_update_blocked_by_identity_mismatch",
                    message="EMA update is blocked because the new sample does not match the enrolled profile strongly enough.",
                    http_status=403,
                )

            blended = ((1.0 - ema_alpha) * previous) + (ema_alpha * embedding)
            embedding_to_store = _l2_normalize(blended).tolist()
            created_at = existing_profile.get("created_at_utc", now)
            num_updates = int(existing_profile.get("num_updates", 0)) + 1
            mode = "ema_update"

        payload = {
            "caller_id": caller_id,
            "embedder": self.embedder.name,
            "model_source": self.embedder.model_source,
            "embedding": embedding_to_store,
            "embedding_dim": int(embedding.shape[0]),
            "created_at_utc": created_at,
            "updated_at_utc": now,
            "num_updates": num_updates,
            "last_duration_seconds": metadata["duration_seconds"],
            "last_rms": metadata["rms"],
            "target_sample_rate_hz": self.embedder.target_sample_rate,
        }
        self.store.save(caller_id, payload)
        return {
            "caller_id": caller_id,
            "enrolled": True,
            "mode": mode,
            "updated": mode == "ema_update",
            "embedder": self.embedder.name,
            "duration_seconds": metadata["duration_seconds"],
            "rms": metadata["rms"],
            "embedding_dim": int(embedding.shape[0]),
            "num_updates": payload["num_updates"],
            "safe_to_enroll": safe_to_enroll,
            "safe_to_update": safe_to_update,
            "synthetic_score": synthetic_score,
            "coercion_score": coercion_score,
            "pre_update_match_confidence": (
                round(pre_update_match_confidence, 6)
                if pre_update_match_confidence is not None
                else None
            ),
            "ema_alpha": ema_alpha
            if mode == "ema_update"
            else None,
        }

    def enroll_from_embeddings(
        self,
        caller_id: str,
        embeddings: list[np.ndarray],
        metadata: list[dict[str, float]] | None = None,
        safe_to_enroll: bool = False,
    ) -> dict[str, Any]:
        if not caller_id or caller_id == "unknown":
            raise IdentityPolicyError(
                code="missing_caller_id",
                message="A resolved caller identity is required before TOFU enrollment.",
                http_status=400,
            )
        if not embeddings:
            raise IdentityPolicyError(
                code="candidate_embeddings_missing",
                message="No temporary voice embeddings were collected for this live session.",
                http_status=409,
            )
        if self.store.load(caller_id) is not None:
            raise IdentityPolicyError(
                code="identity_profile_already_exists",
                message="A profile already exists for this caller. Use EMA update policy instead.",
                http_status=409,
            )
        if self.config.require_safe_call_confirmation_for_enrollment and not safe_to_enroll:
            raise IdentityPolicyError(
                code="tofu_requires_safe_call_confirmation",
                message="TOFU enrollment is blocked until the call is explicitly marked safe.",
                http_status=403,
            )

        vectors = [
            _l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
            for embedding in embeddings
        ]
        dimensions = {int(vector.shape[0]) for vector in vectors}
        if len(dimensions) != 1:
            raise IdentityPolicyError(
                code="candidate_embedding_dim_mismatch",
                message="Temporary embeddings have incompatible dimensions.",
                http_status=409,
            )

        averaged_embedding = _l2_normalize(np.mean(np.stack(vectors), axis=0))
        metadata = metadata or []
        durations = [
            float(item["duration_seconds"])
            for item in metadata
            if "duration_seconds" in item
        ]
        rms_values = [float(item["rms"]) for item in metadata if "rms" in item]
        now = _utc_now()
        payload = {
            "caller_id": caller_id,
            "embedder": self.embedder.name,
            "model_source": self.embedder.model_source,
            "embedding": averaged_embedding.tolist(),
            "embedding_dim": int(averaged_embedding.shape[0]),
            "created_at_utc": now,
            "updated_at_utc": now,
            "num_updates": 0,
            "last_duration_seconds": round(float(np.mean(durations)), 6)
            if durations
            else None,
            "last_rms": round(float(np.mean(rms_values)), 8) if rms_values else None,
            "target_sample_rate_hz": self.embedder.target_sample_rate,
            "candidate_embedding_count": len(vectors),
            "enrollment_source": "live_tofu_session",
        }
        self.store.save(caller_id, payload)
        return {
            "caller_id": caller_id,
            "enrolled": True,
            "mode": "tofu_live_session",
            "updated": False,
            "embedder": self.embedder.name,
            "duration_seconds": payload["last_duration_seconds"],
            "rms": payload["last_rms"],
            "embedding_dim": int(averaged_embedding.shape[0]),
            "num_updates": 0,
            "safe_to_enroll": safe_to_enroll,
            "candidate_embedding_count": len(vectors),
        }

    def extract_candidate_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        return self.embedder.extract(audio, sample_rate)

    def verify_chunk(
        self,
        caller_id: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> IdentityResult:
        duration_seconds = _duration_seconds(audio, sample_rate)

        if not caller_id or caller_id == "unknown":
            return IdentityResult(
                caller_id=caller_id or "unknown",
                status="missing_caller_id",
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text="Missing Caller ID",
                reason="identity_auditor_requires_claimed_identity",
                enrolled=False,
                duration_seconds=duration_seconds,
            )

        profile = self.store.load(caller_id)
        if profile is None:
            return IdentityResult(
                caller_id=caller_id,
                status="not_enrolled",
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text="Not Enrolled",
                reason="no_local_master_vector",
                enrolled=False,
                duration_seconds=duration_seconds,
            )
        if profile.get("embedder") != self.embedder.name:
            return IdentityResult(
                caller_id=caller_id,
                status="profile_incompatible",
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text="Re-enroll Required",
                reason="stored_profile_model_mismatch",
                enrolled=True,
                duration_seconds=duration_seconds,
            )

        try:
            embedding, _ = self.embedder.extract(audio, sample_rate)
        except ValueError as exc:
            reason = str(exc)
            display_text = (
                "Need More Speech" if reason == "insufficient_audio" else "Speech Too Quiet"
            )
            return IdentityResult(
                caller_id=caller_id,
                status=reason,
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text=display_text,
                reason=reason,
                enrolled=True,
                duration_seconds=duration_seconds,
            )
        except RuntimeError as exc:
            reason = str(exc)
            return IdentityResult(
                caller_id=caller_id,
                status="model_unavailable",
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text="Model Unavailable",
                reason=reason,
                enrolled=True,
                duration_seconds=duration_seconds,
            )

        master_vector = _l2_normalize(np.array(profile["embedding"], dtype=np.float32))
        similarity = float(np.dot(master_vector, embedding))
        match_confidence = float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))
        identity_score = float(1.0 - match_confidence)

        if match_confidence >= self.match_threshold:
            status = "match"
            display_text = f"Match ({match_confidence:.2f})"
        elif match_confidence >= self.review_threshold:
            status = "review"
            display_text = f"Review ({match_confidence:.2f})"
        else:
            status = "mismatch"
            display_text = f"Mismatch ({match_confidence:.2f})"

        return IdentityResult(
            caller_id=caller_id,
            status=status,
            identity_score=identity_score,
            similarity=similarity,
            match_confidence=match_confidence,
            display_text=display_text,
            reason=None,
            enrolled=True,
            duration_seconds=duration_seconds,
        )

    def verify_from_base64(self, caller_id: str, base64_audio: str) -> IdentityResult:
        audio, sample_rate = decode_base64_audio(base64_audio)
        return self.verify_chunk(caller_id=caller_id, audio=audio, sample_rate=sample_rate)

    def identify_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        claimed_caller_id: str = "unknown",
        top_k: int = 3,
    ) -> IdentityResult:
        duration_seconds = _duration_seconds(audio, sample_rate)
        profiles = [
            profile
            for profile in self.store.load_all()
            if profile.get("embedder") == self.embedder.name
            and isinstance(profile.get("embedding"), list)
            and profile.get("caller_id")
        ]

        if not profiles:
            return IdentityResult(
                caller_id=claimed_caller_id or "unknown",
                status="no_enrolled_profiles",
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text="No Known Voices",
                reason="no_local_identity_profiles",
                enrolled=False,
                duration_seconds=duration_seconds,
                identification_mode="identification",
            )

        try:
            embedding, _ = self.embedder.extract(audio, sample_rate)
        except ValueError as exc:
            reason = str(exc)
            display_text = (
                "Need More Speech" if reason == "insufficient_audio" else "Speech Too Quiet"
            )
            return IdentityResult(
                caller_id=claimed_caller_id or "unknown",
                status=reason,
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text=display_text,
                reason=reason,
                enrolled=True,
                duration_seconds=duration_seconds,
                identification_mode="identification",
                candidate_count=len(profiles),
            )
        except RuntimeError as exc:
            reason = str(exc)
            return IdentityResult(
                caller_id=claimed_caller_id or "unknown",
                status="model_unavailable",
                identity_score=0.0,
                similarity=None,
                match_confidence=None,
                display_text="Model Unavailable",
                reason=reason,
                enrolled=True,
                duration_seconds=duration_seconds,
                identification_mode="identification",
                candidate_count=len(profiles),
            )

        candidates = []
        for profile in profiles:
            master_vector = _l2_normalize(np.array(profile["embedding"], dtype=np.float32))
            similarity = float(np.dot(master_vector, embedding))
            match_confidence = float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))
            candidates.append(
                {
                    "caller_id": profile["caller_id"],
                    "similarity": round(similarity, 6),
                    "match_confidence": round(match_confidence, 6),
                    "identity_score": round(1.0 - match_confidence, 4),
                }
            )

        candidates.sort(key=lambda candidate: candidate["match_confidence"], reverse=True)
        top_candidates = candidates[: max(top_k, 1)]
        best = top_candidates[0]
        best_confidence = float(best["match_confidence"])
        best_similarity = float(best["similarity"])
        identified_caller_id = str(best["caller_id"])

        if best_confidence >= self.match_threshold:
            status = "identified"
            display_text = f"Identified: {identified_caller_id} ({best_confidence:.2f})"
            reason = None
        elif best_confidence >= self.review_threshold:
            status = "identity_candidate"
            display_text = f"Candidate: {identified_caller_id} ({best_confidence:.2f})"
            reason = "candidate_below_match_threshold"
        else:
            status = "unknown_speaker"
            display_text = "Unknown Speaker"
            reason = "no_candidate_above_review_threshold"
            identified_caller_id = None

        return IdentityResult(
            caller_id=claimed_caller_id or "unknown",
            status=status,
            identity_score=float(1.0 - best_confidence),
            similarity=best_similarity,
            match_confidence=best_confidence,
            display_text=display_text,
            reason=reason,
            enrolled=True,
            duration_seconds=duration_seconds,
            identification_mode="identification",
            identified_caller_id=identified_caller_id,
            candidate_count=len(candidates),
            candidates=top_candidates,
        )

    def identify_from_base64(
        self,
        base64_audio: str,
        claimed_caller_id: str = "unknown",
        top_k: int = 3,
    ) -> IdentityResult:
        audio, sample_rate = decode_base64_audio(base64_audio)
        return self.identify_chunk(
            audio=audio,
            sample_rate=sample_rate,
            claimed_caller_id=claimed_caller_id,
            top_k=top_k,
        )

    def get_policy_snapshot(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.name,
            "model_source": self.embedder.model_source,
            "target_sample_rate_hz": self.embedder.target_sample_rate,
            "min_speech_seconds": self.config.min_speech_seconds,
            "min_rms": self.config.min_rms,
            "match_threshold": self.match_threshold,
            "review_threshold": self.review_threshold,
            "ema_alpha": self.config.ema_alpha,
            "require_safe_call_confirmation_for_enrollment": (
                self.config.require_safe_call_confirmation_for_enrollment
            ),
            "require_safe_call_confirmation_for_update": (
                self.config.require_safe_call_confirmation_for_update
            ),
            "min_match_confidence_for_update": self.config.min_match_confidence_for_update,
            "max_synthetic_score_for_update": self.config.max_synthetic_score_for_update,
            "max_coercion_score_for_update": self.config.max_coercion_score_for_update,
            "status_contract": IDENTITY_STATUS_CONTRACT,
            "policy_contract": IDENTITY_POLICY_CONTRACT,
        }


def decode_base64_audio(base64_audio: str) -> tuple[np.ndarray, int]:
    raw_bytes = base64.b64decode(base64_audio)
    with io.BytesIO(raw_bytes) as buffer:
        waveform, sample_rate = sf.read(buffer, dtype="float32")
    return waveform, int(sample_rate)


def _prepare_waveform(
    audio: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        if waveform.shape[1] == 1:
            waveform = waveform[:, 0]
        else:
            waveform = np.mean(waveform, axis=1)
    elif waveform.ndim != 1:
        waveform = waveform.reshape(-1)

    if waveform.size == 0:
        raise ValueError("insufficient_audio")

    peak = float(np.max(np.abs(waveform)))
    if peak > 1.5:
        waveform = waveform / 32768.0

    waveform = waveform - np.mean(waveform)
    if sample_rate != target_sample_rate:
        waveform = _resample_linear(waveform, sample_rate, target_sample_rate)
    return waveform.astype(np.float32)


def _resample_linear(
    waveform: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample_rate_must_be_positive")
    if source_rate == target_rate:
        return waveform

    duration = len(waveform) / float(source_rate)
    target_length = max(int(round(duration * target_rate)), 1)
    source_positions = np.linspace(0.0, duration, num=len(waveform), endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, waveform).astype(np.float32)


def _duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    waveform = np.asarray(audio)
    if waveform.ndim == 2:
        length = waveform.shape[0]
    else:
        length = waveform.size
    if sample_rate <= 0:
        return 0.0
    return round(length / float(sample_rate), 6)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("zero_norm_embedding")
    return (vector / norm).astype(np.float32)


def _validate_optional_unit_interval(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in the interval [0, 1]")
    return numeric


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
