from __future__ import annotations

import httpx


async def fetch_rawnet_prediction(base64_audio: str, rawnet_url: str) -> float:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(rawnet_url, json={"base64_audio": base64_audio}, timeout=5.0)
            if response.status_code == 200:
                return float(response.json().get("spoof_probability_percent", 0.0))
            print(f"RawNet service error: {response.status_code} {response.text}")
    except Exception as exc:
        print(f"RawNet service unavailable: {exc}")
    return 0.0


async def fetch_distilbert_prediction(text: str, distilbert_url: str) -> dict:
    if not text.strip():
        return {"semantic_score": 0.0, "label": "insufficient_text"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(distilbert_url, json={"scrubbed_text": text}, timeout=5.0)
            if response.status_code == 200:
                return response.json()
            print(f"DistilBERT service error: {response.status_code} {response.text}")
    except Exception as exc:
        print(f"DistilBERT service unavailable: {exc}")
    return {"semantic_score": 0.0, "label": "semantic_unavailable"}
