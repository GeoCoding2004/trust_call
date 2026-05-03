from pydantic import BaseModel, Field


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
