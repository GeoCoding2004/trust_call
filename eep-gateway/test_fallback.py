import asyncio
import base64
import io

import httpx
import numpy as np
import soundfile as sf


def generate_dummy_audio_base64() -> str:
    sample_rate = 16000
    duration_seconds = 1.0
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 220 * t)

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, tone, sample_rate, format="WAV")
    wav_bytes = wav_buffer.getvalue()
    return base64.b64encode(wav_bytes).decode("utf-8")


async def run_test() -> None:
    url = "http://127.0.0.1:8001/analyze"
    payload = {
        "caller_id": "test-user-002",
        "identity_score": 0.35,
        "scrubbed_text": "Please act now and confirm identity.",
        "audio_base64": generate_dummy_audio_base64(),
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=8.0)
            print("=== GATEWAY FALLBACK RESPONSE ===")
            print(f"Status Code: {response.status_code}")
            print(f"Data: {response.json()}")
            print("If DistilBERT is unavailable, expect warnings including 'semantic_unavailable'.")
        except Exception as exc:
            print(f"Connection failed: {exc}. Is the gateway running on port 8001?")


if __name__ == "__main__":
    asyncio.run(run_test())
