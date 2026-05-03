# Trust-Call

## For Grading

The main grading map is:
- `readme_correction.md`

Supporting evidence documents:
- `docs/demo_evidence/README.md`
- `docs/production_hardening.md`
- `docs/evaluation_plan.md`
- `docs/evaluation_results_template.md`
- `docs/observability.md`

CI and QA evidence:
- `.github/workflows/test.yml`
- `.github/workflows/docker-build.yml`
- `tests/`

`readme_correction.md` maps every rubric component to the exact file, test, deployment config, or monitoring artifact where it can be verified.

Trust-Call is a real-time, multimodal AI defense system for VoIP-style calls. It analyzes live audio with three internal AI auditors and combines their outputs through a late-fusion decision engine.

Current `dev` includes the full demo pipeline:

- IEP1 Signal Auditor: RawNet2-based synthetic/deepfake voice detection.
- IEP2 Semantic Auditor: Whisper transcription plus DistilBERT scam/coercion detection.
- IEP3 Identity Auditor: ECAPA-TDNN speaker verification with TOFU enrollment.
- EEP Gateway: WebRTC audio ingestion, parallel service orchestration, telemetry, and late fusion.
- React Native Android app: simulated call flow, contact resolution, live telemetry display, and IEP3 voice-profile save/discard flow.
- Monitoring: Prometheus and Grafana for RawNet, DistilBERT, and the IEP3/backend gateway.

Current public EEP deployment:

- Cloud provider: GCP Compute Engine VM.
- Public EEP URL: `http://35.189.221.158:8080`
- Grafana URL: `http://35.189.221.158:3000`
- Prometheus URL: `http://35.189.221.158:9090`
- Mobile app default backend: the public VM EEP URL above.

## Architecture

```text
React Native app
  -> WebRTC audio offer to public EEP /offer
  -> backend receives live microphone audio
  -> 3-second overlapping audio chunks
  -> IEP1 RawNet service
  -> Whisper STT inside backend
  -> IEP2 DistilBERT service
  -> IEP3 ECAPA identity auditor inside backend
  -> EEP late fusion
  -> WebSocket + polling telemetry back to mobile
```

The mobile app is currently a demo control surface. It simulates incoming calls inside the app; it does not intercept native Android phone calls.

## Services And Ports

| Component | Path | Port | Purpose |
| --- | --- | --- | --- |
| RawNet service | `rawnet-service` | `8000` | IEP1 signal/deepfake detection |
| DistilBERT service | `distilbert-service` | `8002` | IEP2 semantic scam/coercion detection |
| Backend gateway | `trust_call_backend` | `8080` | WebRTC, Whisper, IEP3, EEP fusion, telemetry |
| Prometheus | `docker-compose.yml` | `9090` | Scrapes AI service metrics |
| Grafana | `docker-compose.yml` | `3000` | Trust-Call dashboard |
| React Native Metro | `TrustCallApp` | `8081` or selected Metro port | Mobile bundler |

## IEP1: Signal Auditor

The RawNet service loads the fine-tuned RawNet2 model at startup and exposes:

- `POST /predict`: accepts Base64 WAV audio and returns spoof/real probabilities.
- `GET /metrics`: FastAPI/Prometheus metrics.

The backend sends each live audio chunk to this service asynchronously so the WebRTC stream is not blocked by model inference.

## IEP2: Semantic Auditor

The backend transcribes speech with Whisper, then sends text to the DistilBERT service.

The DistilBERT service exposes:

- `POST /predict`: accepts `scrubbed_text` and returns semantic risk, label, flagged phrases, and model name.
- `GET /metrics`: FastAPI/Prometheus metrics.

Note: live semantic transcription requires `faster-whisper` in the backend Python environment. Without it, the backend still runs, but IEP2 live transcription will not produce real semantic telemetry.

## IEP3: Identity Auditor

IEP3 verifies whether the live speaker matches the expected contact. It is speaker verification, not broad speaker identification by default.

Implemented behavior:

- Resolves a caller/contact ID from the mobile app.
- Extracts ECAPA-TDNN speaker embeddings from live audio chunks.
- Compares the live embedding against the stored master vector using cosine similarity.
- Uses configured thresholds from `configs/iep3_identity.json`.
- Supports TOFU candidate collection when no profile exists.
- After a call, the app can save or discard temporary TOFU embeddings.
- Stores local backend speaker profiles under `trust_call_backend/state/identity_profiles`.

Important privacy note for the current demo: profiles are local to the Python backend machine, not yet encrypted on the phone. The final product should move identity vectors to secure on-device storage or a production-grade encrypted profile store.

Useful IEP3 endpoints:

- `GET /identity/config`
- `GET /identity/enrollment/{caller_id}`
- `POST /identity/enroll`
- `POST /identity/verify`
- `POST /identity/identify`
- `POST /identity/live/validate`
- `GET /identity/live/sessions`
- `GET /identity/live/sessions/{session_id}`
- `POST /identity/live/sessions/{session_id}/enroll-candidate`
- `DELETE /identity/live/sessions/{session_id}/candidate`

## EEP Late Fusion

The backend combines IEP1, IEP2, and IEP3 results into a final call status.

Current fusion behavior:

- Synthetic signal or semantic threat can trigger a threat state.
- Identity mismatch alone creates review/caution states rather than immediately blocking.
- Identity mismatch combined with signal or semantic risk becomes stronger evidence.
- TOFU enrollment is treated as learning mode until the user confirms saving a voice profile.

The mobile app displays this as `Late Fusion Status`.

## Mobile App Behavior

The Android app currently supports:

- Loading real Android contacts when `READ_CONTACTS` is granted.
- Falling back to demo contacts when contacts are unavailable.
- Normalizing phone numbers into stable caller IDs.
- Simulating incoming calls by entered phone number.
- Simulating selected-contact and unknown-caller calls.
- Requesting microphone permission.
- Sending microphone audio to the backend through WebRTC.
- Displaying live Signal, Semantic, Identity, confidence, candidate, chunk, frame, buffer, TOFU, reason, and fusion telemetry.
- Saving or discarding TOFU voice profiles after a call.

For the deployed VM demo, `TrustCallApp/src/config/backend.ts` sets `CLOUD_BACKEND_BASE_URL` to the public GCP VM EEP URL. With that value set, the app uses:

- HTTP: `http://35.189.221.158:8080`
- WebSocket: `ws://35.189.221.158:8080`

To return to laptop-local testing, set `CLOUD_BACKEND_BASE_URL` to `null`.

For a physical Android phone connected by USB against a laptop-local backend, keep the backend host override as `127.0.0.1` and run:

```powershell
adb reverse tcp:8080 tcp:8080
```

For a LAN/Wi-Fi phone test without `adb reverse`, update `TrustCallApp/src/config/backend.ts` so `MANUAL_BACKEND_HOST_OVERRIDE` points to the laptop IP address.

## Monitoring

Prometheus and Grafana are configured through Docker Compose.

Prometheus scrapes:

- `rawnet_audio_ai` at `rawnet-service:8000/metrics`
- `distilbert_semantic_ai` at `distilbert-service:8002/metrics`
- `iep3_identity_gateway` at `trust-call-backend:8080/metrics`

Grafana is provisioned automatically with:

- Prometheus datasource.
- `Trust-Call AI Services` dashboard.
- Panels for service health, backend audio chunk rate, IEP3 identity decisions, enrolled profiles, and EEP fusion outcomes.

Start monitoring:

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline
docker compose up
```

Open:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Cloud Deployment

The live demo deployment uses a GCP Compute Engine VM because the app streams live WebRTC audio to an `aiortc` backend. Cloud Run was suitable for HTTP APIs, but it did not reliably carry the WebRTC media path for this project.

| Service | Deployment | Public Port | Role |
| --- | --- | --- | --- |
| EEP | `trust-call-backend` Docker Compose service with host networking | `8080` | Public system boundary, WebRTC offer handling, Whisper, IEP3, EEP fusion |
| IEP1 | `rawnet-service` Docker Compose service | `8000` | RawNet signal/deepfake inference |
| IEP2 | `distilbert-service` Docker Compose service | `8002` | DistilBERT semantic scam/coercion inference |
| Prometheus | Docker Compose service | `9090` | Metrics scraping |
| Grafana | Docker Compose service | `3000` | Dashboard |

The VM backend is configured with:

```text
RAWNET_URL=http://127.0.0.1:8000/predict
DISTILBERT_URL=http://127.0.0.1:8002/predict
TRUST_CALL_WEBRTC_STUN_URL=stun:stun.l.google.com:19302
```

The backend container uses host networking so `aiortc` can expose reachable WebRTC candidates from the VM. RawNet and DistilBERT still run as separate services and are reached through their published localhost ports.

Required VM firewall ports for the demo:

```text
TCP 8080    EEP/backend
TCP 3000    Grafana
TCP 9090    Prometheus
TCP 8000    RawNet debug
TCP 8002    DistilBERT debug
UDP 1024-65535 WebRTC media
```

Kubernetes manifests are included under `deployment/kubernetes/` for portability and orchestration evidence. The working live demo target is the GCP VM because it supports the current WebRTC media path with fewer moving parts before the deadline.

Deployment tradeoff: the public EEP is unauthenticated and the demo firewall is permissive for WebRTC. Production should add authentication, request limits, TURN, tighter firewall rules, encrypted profile storage, secret management, and hardened observability.

Cloud smoke tests:

```powershell
Invoke-WebRequest http://35.189.221.158:8080/docs
Invoke-WebRequest http://35.189.221.158:8080/metrics
```

## Friend Cloud Demo Quickstart

Use this path when testing the deployed VM demo without local Python models.

What your tester needs:

- Latest `dev` branch.
- Node/npm installed.
- Android Studio / Android SDK installed.
- A running Android emulator or USB-connected Android phone with USB debugging enabled.
- Internet access.

What your tester does not need:

- No local Python backend.
- No local RawNet model.
- No local DistilBERT model.
- No local ECAPA/IEP3 model cache.
- No `adb reverse tcp:8080 tcp:8080`.

The mobile app already points to the deployed public VM EEP:

```text
http://35.189.221.158:8080
```

### Path A: React Native Dev Run

Use this when actively debugging the app from a laptop.

```powershell
git checkout dev
git pull origin dev
cd TrustCallApp
npm install
```

Terminal 1:

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline\TrustCallApp

$env:JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"
$env:ANDROID_HOME="C:\Users\JL\AppData\Local\Android\Sdk"
$env:Path="$env:JAVA_HOME\bin;$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\emulator;$env:Path"

adb devices
```

Terminal 2:

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline\TrustCallApp
npm start -- --reset-cache
```

Terminal 3:

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline\TrustCallApp

$env:JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"
$env:ANDROID_HOME="C:\Users\JL\AppData\Local\Android\Sdk"
$env:Path="$env:JAVA_HOME\bin;$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\emulator;$env:Path"

npx react-native run-android --device <device_id> --port 8082
```

Do not run a local backend and do not run `adb reverse` for the VM demo.

### Path B: Standalone APK

Use this when the phone should run without staying plugged into the laptop.

Build the APK:

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline\TrustCallApp

$env:JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"
$env:ANDROID_HOME="C:\Users\JL\AppData\Local\Android\Sdk"
$env:Path="$env:JAVA_HOME\bin;$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\emulator;$env:Path"

cd android
.\gradlew.bat assembleRelease
```

Install while the phone is plugged in:

```powershell
adb install -r C:\Users\JL\Desktop\trust_call_full_pipeline\TrustCallApp\android\app\build\outputs\apk\release\app-release.apk
```

After installation, unplug the phone and open Trust-Call normally. The app still calls the deployed VM EEP over the internet.

If release build fails, build/install debug instead:

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline\TrustCallApp\android
.\gradlew.bat assembleDebug
adb install -r C:\Users\JL\Desktop\trust_call_full_pipeline\TrustCallApp\android\app\build\outputs\apk\debug\app-debug.apk
```

### Expected Result

1. The app opens on Android.
2. Select or simulate a caller.
3. Press `Accept`.
4. Allow microphone permission.
5. Speak for 10-20 seconds.
6. The app should show Signal, Semantic, Identity, and Late Fusion telemetry.

If the app cannot connect, first verify the public backend opens:

```text
http://35.189.221.158:8080/docs
```

If the app connects but telemetry stays at zero frames, check VM logs:

```bash
cd ~/trust_call
docker compose logs -f trust-call-backend
```

Demo limitation: in this deployed demo, IEP3 runs in the hosted backend so testers do not need the ECAPA model locally. The intended production privacy direction is to move IEP3 embeddings/model execution to secure on-device storage.

## Running The Local Demo

Use separate terminals.

### Terminal 1: RawNet

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline\rawnet-service
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Terminal 2: DistilBERT

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline\distilbert-service
python -m uvicorn main:app --host 0.0.0.0 --port 8002
```

### Terminal 3: Backend Gateway

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline
python -m uvicorn trust_call_backend.server:app --host 0.0.0.0 --port 8080
```

### Terminal 4: Monitoring

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline
docker compose up
```

### Terminal 5: Android App

```powershell
cd C:\Users\JL\Desktop\trust_call_full_pipeline\TrustCallApp
npx react-native run-android
```

`adb reverse tcp:8080 tcp:8080` is only required when `CLOUD_BACKEND_BASE_URL` is `null` and the app is targeting a laptop-local backend.

If multiple Android devices are connected:

```powershell
adb devices
npx react-native run-android --device <device_id>
```

## Expected Demo Flow

1. Open the app.
2. Confirm the backend status shows online.
3. Enter or select a caller/contact.
4. Start a simulated call.
5. Press `Accept`.
6. Speak for at least 10-20 seconds.
7. Watch Signal, Semantic, Identity, and Late Fusion telemetry update.
8. If the caller is not enrolled, end the call and choose whether to save the collected TOFU voice profile.
9. Repeat the call for that contact and check whether IEP3 moves from TOFU collection to verification.
10. Open Grafana and confirm the IEP3/backend panels update while calls run.

## Evaluation Tools

IEP3 evaluation scripts are available under `scripts/`:

- `scripts/evaluate_iep3.py`
- `scripts/plot_iep3_metrics.py`

Dataset instructions live in:

- `data/evaluation/iep3/README.md`

The evaluator generates trial scores, summary metrics, and threshold recommendations. The current production threshold policy is stored in `configs/iep3_identity.json`.

## Current Limitations

- The app simulates calls inside Trust-Call; it does not hook into Android's native dialer.
- AI models run through local/backend services, not fully on-device.
- IEP3 profile vectors are stored in the backend local state directory for the demo.
- Live IEP2 requires `faster-whisper` to be installed in the backend environment.
- `start_services.ps1` may need path cleanup before it is reliable across machines.
- CI exists, but production deployment still needs stronger security hardening and real benchmark evidence.
