import httpx
import base64
import os
import sys

# 1. Point this to your test audio file
test_audio_path = "test_voice_AI.wav"

if not os.path.exists(test_audio_path):
    print(f"❌ Error: '{test_audio_path}' not found. Please copy your test_voice.wav into the rawnet-service folder.")
    sys.exit(1)

print(f"1. Reading real audio from {test_audio_path}...")
with open(test_audio_path, "rb") as audio_file:
    raw_audio_bytes = audio_file.read()

print("2. Converting raw audio to Base64 string...")
# Decode to utf-8 so it becomes a standard string for the JSON payload
base64_audio = base64.b64encode(raw_audio_bytes).decode('utf-8')

payload = {
    "base64_audio": base64_audio
}

print("3. Sending JSON payload to RawNet2 Microservice (http://localhost:8000/predict)...")
try:
    # Send the POST request to your locally running server
    response = httpx.post("http://localhost:8000/predict", json=payload, timeout=10.0)
    
    print("\n=== SERVER RESPONSE ===")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ INFERENCE SUCCESSFUL:")
        print(f"Spoof (AI) Probability:    {data.get('spoof_probability_percent')}%")
        print(f"Real (Human) Probability:  {data.get('real_probability_percent')}%")
    else:
        print(f"❌ Error Details: {response.text}")
        
except httpx.ConnectError:
    print("❌ Connection failed. Make sure your uvicorn server is actively running in another terminal window!")