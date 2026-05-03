"""Tests for timeout and fallback behaviour in production service clients."""

import asyncio

from trust_call_backend.service_clients import (
    fetch_distilbert_prediction,
    fetch_rawnet_prediction,
)


def test_rawnet_service_unavailable_returns_zero():
    score = asyncio.run(
        fetch_rawnet_prediction(
            base64_audio="AAAA",
            rawnet_url="http://127.0.0.1:19999/predict",
            timeout=0.1,
        )
    )

    assert score == 0.0


def test_distilbert_service_unavailable_returns_fallback():
    result = asyncio.run(
        fetch_distilbert_prediction(
            text="Send money urgently",
            distilbert_url="http://127.0.0.1:19998/predict",
            timeout=0.1,
        )
    )

    assert result == {
        "semantic_score": 0.0,
        "label": "semantic_unavailable",
    }


def test_distilbert_empty_text_returns_insufficient_without_network_call():
    result = asyncio.run(
        fetch_distilbert_prediction(
            text="",
            distilbert_url="http://127.0.0.1:19998/predict",
            timeout=0.1,
        )
    )

    assert result == {
        "semantic_score": 0.0,
        "label": "insufficient_text",
    }


def test_distilbert_whitespace_text_returns_insufficient_without_network_call():
    result = asyncio.run(
        fetch_distilbert_prediction(
            text="   \n\t   ",
            distilbert_url="http://127.0.0.1:19998/predict",
            timeout=0.1,
        )
    )

    assert result == {
        "semantic_score": 0.0,
        "label": "insufficient_text",
    }
