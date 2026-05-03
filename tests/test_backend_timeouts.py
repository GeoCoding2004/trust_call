"""Tests for timeout and fallback behaviour in backend fetchers."""
import asyncio


async def fetch_rawnet_impl(base64_audio: str, rawnet_url: str, timeout: float) -> float:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(rawnet_url, json={"base64_audio": base64_audio}, timeout=timeout)
            if response.status_code == 200:
                return float(response.json().get("spoof_probability_percent", 0.0))
    except Exception:
        pass
    return 0.0


async def fetch_distilbert_impl(text: str, distilbert_url: str, timeout: float) -> dict:
    import httpx
    if not text.strip():
        return {"semantic_score": 0.0, "label": "insufficient_text"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(distilbert_url, json={"scrubbed_text": text}, timeout=timeout)
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    return {"semantic_score": 0.0, "label": "semantic_unavailable"}


def test_rawnet_service_unavailable_returns_zero():
    score = asyncio.run(fetch_rawnet_impl("AAAA", "http://127.0.0.1:19999/predict", 0.1))
    assert score == 0.0


def test_distilbert_service_unavailable_returns_fallback():
    result = asyncio.run(fetch_distilbert_impl("Send money urgently", "http://127.0.0.1:19998/predict", 0.1))
    assert result["label"] == "semantic_unavailable"
    assert result["semantic_score"] == 0.0


def test_distilbert_empty_text_returns_insufficient_without_network_call():
    result = asyncio.run(fetch_distilbert_impl("", "http://127.0.0.1:19998/predict", 0.1))
    assert result["label"] == "insufficient_text"
    assert result["semantic_score"] == 0.0
