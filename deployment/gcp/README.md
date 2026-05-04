# GCP Cloud Run Deployment

Trust-Call's live cloud demo deploys the public EEP and the first two internal AI endpoints on GCP Cloud Run.

## Deployed Services

| Service | URL | Role |
| --- | --- | --- |
| `trust-call-backend` | `https://trust-call-backend-uccxv72y5a-ew.a.run.app` | Public EEP, WebRTC offer handling, Whisper, IEP3, late fusion, telemetry |
| `rawnet-service` | `https://rawnet-service-uccxv72y5a-ew.a.run.app` | IEP1 RawNet synthetic/deepfake inference |
| `distilbert-service` | `https://distilbert-service-uccxv72y5a-ew.a.run.app` | IEP2 DistilBERT scam/coercion inference |

IEP3 identity embeddings remain privacy-sensitive demo state. The production target is secure on-device storage or encrypted profile storage. Raw audio and embeddings should not be logged.

## Runtime Wiring

The public EEP is configured with:

```text
RAWNET_URL=https://rawnet-service-uccxv72y5a-ew.a.run.app/predict
DISTILBERT_URL=https://distilbert-service-uccxv72y5a-ew.a.run.app/predict
```

The React Native app points to the public EEP through:

```text
TrustCallApp/src/config/backend.ts
```

Current value:

```text
CLOUD_BACKEND_BASE_URL=https://trust-call-backend-uccxv72y5a-ew.a.run.app
```

## Smoke Tests

```powershell
Invoke-WebRequest https://trust-call-backend-uccxv72y5a-ew.a.run.app/docs
Invoke-WebRequest https://trust-call-backend-uccxv72y5a-ew.a.run.app/metrics
```

Expected result: HTTP 200.

## Mobile Demo Test

1. Build/run the Android app.
2. Select or simulate a caller.
3. Press `Accept`.
4. Speak for at least 10-20 seconds.
5. Confirm the app shows:
   - Signal result from IEP1.
   - Semantic result from IEP2 when enough transcript exists.
   - Identity result/candidate state from IEP3.
   - Late fusion status from the EEP.

## Project Requirement Mapping

- Real cloud deployment: GCP Cloud Run.
- Public EEP: `trust-call-backend`.
- Independent IEPs: `rawnet-service` and `distilbert-service`.
- Docker images: RawNet, DistilBERT, backend.
- Kubernetes manifests: `deployment/kubernetes/trust-call-stack.yaml`.
- Observability: Prometheus/Grafana are configured locally; Cloud Run logs and metrics provide cloud-native runtime visibility.

## Tradeoff

Cloud Run is used as the live deployment target because it is simpler and safer to operate before the deadline than a full Kubernetes cluster. Kubernetes manifests are kept for orchestration portability and rubric evidence.
