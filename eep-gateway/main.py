import asyncio
import math
import os
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Trust-Call EEP Gateway", version="1.0")


class AudioPayload(BaseModel):
    caller_id: str = "unknown"
    identity_score: float = 0.0
    scrubbed_text: str = ""
    audio_base64: Optional[str] = None
    base64_audio: Optional[str] = None

    @property
    def resolved_audio_base64(self) -> str:
        return self.audio_base64 or self.base64_audio or ""


RAWNET_URL = os.getenv("RAWNET_URL", "http://127.0.0.1:8000/predict")
DISTILBERT_URL = os.getenv("DISTILBERT_URL", "http://127.0.0.1:8002/predict")

RAWNET_TIMEOUT = float(os.getenv("RAWNET_TIMEOUT", "10.0"))
DISTILBERT_TIMEOUT = float(os.getenv("DISTILBERT_TIMEOUT", "4.0"))


def calculate_late_fusion(signal: float, semantic: float, identity: float) -> float:
    w1, w2, w3 = 0.4, 0.4, 0.2
    raw_score = (w1 * signal) + (w2 * semantic) + (w3 * identity)
    return 1 / (1 + math.exp(-raw_score))


async def call_rawnet(
    client: httpx.AsyncClient, audio_base64: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = await client.post(
            RAWNET_URL,
            json={"base64_audio": audio_base64},
            timeout=RAWNET_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPError as exc:
        return None, f"http_error:{exc}"
    except Exception:
        return None, "exception"


async def call_distilbert(
    client: httpx.AsyncClient, scrubbed_text: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        response = await client.post(
            DISTILBERT_URL,
            json={"scrubbed_text": scrubbed_text},
            timeout=DISTILBERT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None
    except httpx.TimeoutException:
        return None, "timeout"
    except httpx.HTTPError as exc:
        return None, f"http_error:{exc}"
    except Exception:
        return None, "exception"


@app.post("/analyze")
async def analyze_audio(payload: AudioPayload):
    audio_base64 = payload.resolved_audio_base64
    if not audio_base64:
        raise HTTPException(status_code=400, detail="Missing audio payload")

    async with httpx.AsyncClient() as client:
        rawnet_task = call_rawnet(client, audio_base64)
        distilbert_task = call_distilbert(client, payload.scrubbed_text)
        rawnet_result, distilbert_result = await asyncio.gather(rawnet_task, distilbert_task)

    rawnet_data, rawnet_error = rawnet_result
    distilbert_data, distilbert_error = distilbert_result

    warnings = []

    signal_status = "ok"
    signal_score = 0.0
    spoof_prob = 0.0
    real_prob = 0.0
    if rawnet_data is None:
        signal_status = rawnet_error or "unavailable"
        warnings.append("signal_unavailable")
    else:
        spoof_prob = float(rawnet_data.get("spoof_probability_percent", 0.0))
        real_prob = float(rawnet_data.get("real_probability_percent", 0.0))
        signal_score = spoof_prob / 100.0

    semantic_status = "ok"
    semantic_score = 0.0
    semantic_label = "semantic_unavailable"
    flagged_phrases = []
    if distilbert_data is None:
        semantic_status = distilbert_error or "unavailable"
        warnings.append("semantic_unavailable")
    else:
        semantic_score = float(distilbert_data.get("semantic_score", 0.0))
        semantic_label = distilbert_data.get("label", "unknown")
        flagged_phrases = distilbert_data.get("flagged_phrases", [])

    final_risk = calculate_late_fusion(signal_score, semantic_score, payload.identity_score)
    threat = "CRITICAL" if final_risk > 0.75 or spoof_prob > 50.0 else "SAFE"
    action = "DUCK_AUDIO" if threat == "CRITICAL" else "NONE"
    decision = "BLOCK" if threat == "CRITICAL" else "ALLOW"

    return {
        "status": "degraded" if warnings else "success",
        "decision": decision,
        "threat": threat,
        "threat_level": threat,
        "action": action,
        "late_fusion_score": round(final_risk, 3),
        "warnings": warnings,
        "details": {
            "spoof_probability_percent": round(spoof_prob, 4),
            "real_probability_percent": round(real_prob, 4),
        },
        "breakdown": {
            "signal": round(signal_score, 4),
            "semantic": round(semantic_score, 4),
            "identity": round(payload.identity_score, 4),
        },
        "semantic": {
            "semantic_score": round(semantic_score, 4),
            "label": semantic_label,
            "flagged_phrases": flagged_phrases,
            "status": semantic_status,
        },
        "signal": {
            "status": signal_status,
        },
    }
