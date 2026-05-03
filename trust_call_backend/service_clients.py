from __future__ import annotations

from typing import Any

import httpx


async def fetch_rawnet_prediction_details(
    base64_audio: str,
    rawnet_url: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                rawnet_url,
                json={"base64_audio": base64_audio},
                timeout=timeout,
            )

        if response.status_code == 200:
            return {
                "score": float(response.json().get("spoof_probability_percent", 0.0)),
                "status": "success",
                "http_status": 200,
            }

        print(f"RawNet service error: {response.status_code} {response.text}")
        return {
            "score": 0.0,
            "status": "http_error",
            "http_status": response.status_code,
        }

    except httpx.TimeoutException:
        print("RawNet service unavailable: timeout")
        return {"score": 0.0, "status": "timeout", "http_status": None}
    except httpx.HTTPError as exc:
        print(f"RawNet service unavailable: {exc}")
        return {"score": 0.0, "status": "unavailable", "http_status": None}


async def fetch_rawnet_prediction(
    base64_audio: str,
    rawnet_url: str,
    timeout: float = 5.0,
) -> float:
    result = await fetch_rawnet_prediction_details(base64_audio, rawnet_url, timeout=timeout)
    return float(result["score"])


async def fetch_distilbert_prediction_details(
    text: str,
    distilbert_url: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    if not text.strip():
        return {
            "data": {
                "semantic_score": 0.0,
                "label": "insufficient_text",
            },
            "status": "insufficient_text",
            "http_status": None,
        }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                distilbert_url,
                json={"scrubbed_text": text},
                timeout=timeout,
            )

        if response.status_code == 200:
            return {
                "data": response.json(),
                "status": "success",
                "http_status": 200,
            }

        print(f"DistilBERT service error: {response.status_code} {response.text}")
        return {
            "data": {
                "semantic_score": 0.0,
                "label": "semantic_unavailable",
            },
            "status": "http_error",
            "http_status": response.status_code,
        }

    except httpx.TimeoutException:
        print("DistilBERT service unavailable: timeout")
        return {
            "data": {
                "semantic_score": 0.0,
                "label": "semantic_unavailable",
            },
            "status": "timeout",
            "http_status": None,
        }
    except httpx.HTTPError as exc:
        print(f"DistilBERT service unavailable: {exc}")
        return {
            "data": {
                "semantic_score": 0.0,
                "label": "semantic_unavailable",
            },
            "status": "unavailable",
            "http_status": None,
        }


async def fetch_distilbert_prediction(
    text: str,
    distilbert_url: str,
    timeout: float = 5.0,
) -> dict:
    result = await fetch_distilbert_prediction_details(text, distilbert_url, timeout=timeout)
    return dict(result["data"])
