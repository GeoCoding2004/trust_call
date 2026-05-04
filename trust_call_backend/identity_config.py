from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "iep3_identity.json"
CONFIG_PATH_ENV_VAR = "TRUSTCALL_IEP3_CONFIG_PATH"


@dataclass(frozen=True)
class IdentityAuditorConfig:
    target_sample_rate_hz: int = 16000
    min_speech_seconds: float = 1.5
    min_rms: float = 0.003
    match_threshold: float = 0.82
    review_threshold: float = 0.68
    ema_alpha: float = 0.20
    require_safe_call_confirmation_for_enrollment: bool = True
    require_safe_call_confirmation_for_update: bool = True
    min_match_confidence_for_update: float = 0.90
    max_synthetic_score_for_update: float = 0.35
    max_coercion_score_for_update: float = 0.35
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    model_savedir: str = "trust_call_backend/state/models/ecapa_voxceleb"

    def __post_init__(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        if self.min_speech_seconds <= 0.0:
            raise ValueError("min_speech_seconds must be positive")
        if self.min_rms < 0.0:
            raise ValueError("min_rms must be non-negative")
        if not 0.0 < self.review_threshold < self.match_threshold <= 1.0:
            raise ValueError(
                "review_threshold and match_threshold must satisfy 0 < review < match <= 1"
            )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in the interval (0, 1]")
        if not 0.0 <= self.min_match_confidence_for_update <= 1.0:
            raise ValueError("min_match_confidence_for_update must be in the interval [0, 1]")
        if not 0.0 <= self.max_synthetic_score_for_update <= 1.0:
            raise ValueError("max_synthetic_score_for_update must be in the interval [0, 1]")
        if not 0.0 <= self.max_coercion_score_for_update <= 1.0:
            raise ValueError("max_coercion_score_for_update must be in the interval [0, 1]")
        if not self.model_source.strip():
            raise ValueError("model_source must not be empty")
        if not self.model_savedir.strip():
            raise ValueError("model_savedir must not be empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IdentityAuditorConfig":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_model_savedir(self) -> Path:
        savedir = Path(self.model_savedir)
        if savedir.is_absolute():
            return savedir
        return REPO_ROOT / savedir


def resolve_identity_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv(CONFIG_PATH_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return DEFAULT_CONFIG_PATH


def load_identity_auditor_config(
    config_path: str | Path | None = None,
) -> IdentityAuditorConfig:
    path = resolve_identity_config_path(config_path)
    if not path.is_file():
        if path == DEFAULT_CONFIG_PATH:
            return IdentityAuditorConfig()
        raise FileNotFoundError(f"IEP3 config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("IEP3 config must deserialize to a JSON object")

    return IdentityAuditorConfig.from_dict(payload)
