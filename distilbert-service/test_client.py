import asyncio
import httpx


SAMPLES = [
    ("benign", "Hey, just checking in about our meeting tomorrow."),
    ("suspicious", "Urgent: verify your account and send money via wire transfer."),
    ("empty", "   "),
]


async def run_tests() -> None:
    url = "http://127.0.0.1:8002/predict"
    async with httpx.AsyncClient() as client:
        for label, text in SAMPLES:
            try:
                response = await client.post(url, json={"scrubbed_text": text}, timeout=5.0)
                print(f"\n=== {label.upper()} SAMPLE ===")
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.json()}")
            except Exception as exc:
                print(f"\n=== {label.upper()} SAMPLE ===")
                print(f"Request failed: {exc}")


if __name__ == "__main__":
    asyncio.run(run_tests())
