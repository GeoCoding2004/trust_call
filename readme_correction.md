# Trust-Call Grading Map

> **For graders:** This file is the single-entry-point map from every rubric item to the exact file, test, or deployment artifact where evidence lives. Read this first, then follow the links.

---

Final submission checklist: `docs/final_submission_checklist.md`

## Project Overview

Trust-Call is a production-oriented, real-time AI application for scam and fraud call detection.
It combines three independent AI components whose outputs are fused by a backend orchestration gateway:

| Component | Technology | Role |
|---|---|---|
| IEP1 – Signal Auditor | RawNet2 (fine-tuned) | Deepfake / synthetic-voice detection from raw audio |
| IEP2 – Semantic Auditor | Whisper STT + fine-tuned DistilBERT | Transcript-level scam / coercion intent classification |
| IEP3 – Identity Auditor | ECAPA-TDNN speaker embeddings | Voiceprint enrollment and per-call identity verification |
| EEP Gateway | FastAPI / aiortc | WebRTC ingestion, parallel orchestration, late-fusion, WebSocket telemetry |
| Mobile frontend | React Native (Android) | Simulated call flow, live telemetry display, TOFU save/discard |
| Deployment | Docker Compose + GCP VM + Kubernetes manifests | Container orchestration, cloud deployment |
| Monitoring | Prometheus + Grafana | Per-service metrics, dashboard, ML-specific counters |

The project is an **Application** project, not a pure research project. Every AI component is a deployed, testable microservice with health checks and Prometheus metrics.

---

## Repository Map

| Path | Purpose |
|---|---|
| `README.md` | Architecture, setup, demo flow, cloud URLs |
| `readme_correction.md` | **This file** – rubric grading map |
| `docker-compose.yml` | Orchestrates all services locally and in the cloud |
| `trust_call_backend/server.py` | EEP Gateway: WebRTC, Whisper, IEP3, late-fusion, `/metrics`, `/offer`, identity endpoints |
| `trust_call_backend/identity_auditor.py` | IEP3 ECAPA-TDNN speaker verification logic |
| `trust_call_backend/identity_config.py` | IEP3 policy configuration loader |
| `trust_call_backend/requirements.txt` | Backend Python dependencies |
| `trust_call_backend/Dockerfile` | Backend container definition |
| `rawnet-service/main.py` | IEP1 FastAPI service: `/predict`, `/metrics` |
| `rawnet-service/model.py` | RawNet2 model architecture |
| `rawnet-service/train_transfer.py` | Transfer-learning fine-tuning script |
| `rawnet-service/evaluate_rawnet.py` | RawNet evaluation script |
| `rawnet-service/requirements.txt` | RawNet service dependencies |
| `rawnet-service/Dockerfile` | RawNet container definition |
| `distilbert-service/main.py` | IEP2 FastAPI service: `/predict`, `/metrics`, fallback heuristic |
| `distilbert-service/test_main.py` | Existing IEP2 unit + integration tests |
| `distilbert-service/requirements.txt` | DistilBERT service dependencies |
| `distilbert-service/Dockerfile` | DistilBERT container definition |
| `TrustCallApp/` | React Native Android mobile app |
| `TrustCallApp/src/config/backend.ts` | Cloud backend URL switch |
| `monitoring/prometheus.yml` | Prometheus scrape targets for all three AI services |
| `monitoring/grafana/` | Grafana provisioning: datasource + Trust-Call AI dashboard |
| `deployment/gcp/` | GCP Compute Engine deployment files |
| `deployment/azure/` | Azure deployment files |
| `deployment/kubernetes/` | Kubernetes manifests (portability evidence) |
| `configs/iep3_identity.json` | IEP3 threshold policy |
| `scripts/evaluate_iep3.py` | IEP3 end-to-end evaluation runner |
| `scripts/plot_iep3_metrics.py` | IEP3 metrics visualization |
| `scripts/collect_demo_evidence.py` | **New** – automated health / prediction checks for demo evidence |
| `semantic_model_training/` | DistilBERT fine-tuning scripts and data pipeline |
| `data/` | Evaluation dataset README and supporting data |
| `tests/` | **New** – pytest test suite (fusion, validation, timeouts, identity, rawnet, semantic) |
| `docs/demo_evidence/README.md` | **New** – evidence collection guide |
| `docs/production_hardening.md` | **New** – production gap analysis and hardening plan |
| `docs/evaluation_plan.md` | **New** – evaluation methodology per component |
| `docs/evaluation_results_template.md` | **New** – fillable results template |
| `docs/observability.md` | **New** – Prometheus metrics reference and Grafana guide |
| `docs/final_submission_checklist.md` | **New** – final pre-submission verification checklist |
| `.env.example` | **New** – environment variable reference |
| `.github/workflows/test.yml` | **New** – CI: pytest on push/PR |
| `.github/workflows/docker-build.yml` | **New** – CI: Docker build validation on push/PR |

---

## Baseline Gates

### GT1 – Demo works end-to-end

- Full demo flow: `README.md` → **Expected Demo Flow** section
- Backend gateway: `trust_call_backend/server.py` (FastAPI app, `/offer` endpoint)
- Mobile frontend: `TrustCallApp/` (React Native, connects to public GCP VM by default)
- How to run locally: `README.md` → **Running The Local Demo** section
- Public cloud backend (no local setup needed): `http://35.189.221.158:8080/docs`

### GT2 – Public cloud API functional

- Deployment docs: `README.md` → **Cloud Deployment** section
- Public EEP endpoint: `http://35.189.221.158:8080`
- Public Grafana: `http://35.189.221.158:3000`
- Public Prometheus: `http://35.189.221.158:9090`
- Health check commands (see also `docs/demo_evidence/README.md`):
  ```bash
  curl http://35.189.221.158:8080/docs
  curl http://35.189.221.158:8080/metrics
  ```
- **Caveat:** Public IP availability requires team to have the VM running. Verify live at demo time.

### GT3 – Architecture minimum met

- Architecture description: `README.md` → **Architecture** section (ASCII diagram + service table)
- Docker Compose with all five services: `docker-compose.yml`
- Service boundaries: separate containers for IEP1 (port 8000), IEP2 (port 8002), EEP/IEP3 (port 8080), Prometheus (9090), Grafana (3000)
- Kubernetes manifests: `deployment/kubernetes/`

### GT4 – Required deliverables complete

| Deliverable | Location |
|---|---|
| README | `README.md` |
| Grading map | `readme_correction.md` (this file) |
| Deployment files | `docker-compose.yml`, `deployment/` |
| Test files | `tests/`, `distilbert-service/test_main.py` |
| Demo evidence guide | `docs/demo_evidence/README.md` |
| CI workflows | `.github/workflows/test.yml`, `.github/workflows/docker-build.yml` |

### GT5 – Type-specific minimum met (Application project)

This is an application project because:
- Three independent AI inference services are deployed as containerized microservices
- Each service exposes production-style HTTP APIs with health checks and Prometheus metrics
- A WebRTC-capable gateway orchestrates them in real time with late fusion
- A mobile frontend consumes the live results
- The system is deployed on a public cloud VM

Production-oriented AI components:
- RawNet2 fine-tuned deepfake detection (`rawnet-service/`)
- Fine-tuned DistilBERT scam classifier with heuristic fallback (`distilbert-service/`)
- ECAPA-TDNN speaker verification with TOFU enrollment (`trust_call_backend/identity_auditor.py`)
- Whisper STT integration inside the backend gateway
- Late-fusion EEP combining all three signals (`trust_call_backend/server.py` → `build_fusion_status`, `orchestrate_late_fusion`)

---

## Rubric Mapping

| Rubric Item | Evidence Location | Status | Notes |
|---|---|---|---|
| **T1** AI depth and non-triviality | `rawnet-service/model.py` (RawNet2), `trust_call_backend/identity_auditor.py` (ECAPA-TDNN), `distilbert-service/main.py` (DistilBERT fine-tuned classifier) | ✅ | Three separate non-trivial AI models, each with distinct architectures |
| **T2** IEP1 independence and value | `rawnet-service/` – standalone FastAPI service, separate Dockerfile, own requirements, own metrics | ✅ | Detects synthetic/deepfake voice independently of IEP2 and IEP3 |
| **T3** IEP2 independence and value | `distilbert-service/` – standalone FastAPI service, separate Dockerfile, heuristic fallback, Prometheus metrics | ✅ | Classifies transcript-level scam intent independently |
| **T4** EEP orchestration logic | `trust_call_backend/server.py` → `orchestrate_late_fusion`, `build_fusion_status` | ✅ | Parallel async calls to IEP1 and IEP2, IEP3 inline, fusion into threat/safe/review states |
| **T5** Tradeoff evidence | `README.md` → Cloud Deployment tradeoff section; `docs/production_hardening.md` | ✅ | GCP VM chosen over Cloud Run for WebRTC media path; documented auth/security gaps |
| **T6** Execution quality and edge cases | `tests/test_backend_fusion.py`, `tests/test_backend_validation.py`, `tests/test_backend_timeouts.py`; fallback handling in `server.py` `fetch_rawnet`/`fetch_distilbert` | ✅ | Tests cover low/high risk, missing fields, timeouts, degraded responses |
| **S1** Service boundaries and contracts | `docker-compose.yml` (named services, explicit ports), `README.md` → Services And Ports table, Pydantic models in each service | ✅ | Clear API contracts via Pydantic; each service has its own network boundary |
| **S2** Validation and request constraints | `distilbert-service/main.py` Pydantic + `RequestValidationError` handler; `trust_call_backend/server.py` Pydantic models; `tests/test_backend_validation.py` | ✅ | Invalid payloads return 400; missing fields caught by Pydantic |
| **S3** Errors, timeouts, retries, fallbacks | `server.py` → `fetch_rawnet`/`fetch_distilbert` catch all exceptions, return safe defaults; `distilbert-service/main.py` heuristic fallback when model not loaded | ✅ | Services unavailable → fallback score 0, not crash |
| **S4** Containerization and orchestration | `docker-compose.yml`, individual Dockerfiles in each service directory | ✅ | All five services containerized; health checks defined |
| **S5** Deployment architecture and secrets | `deployment/gcp/`, `deployment/kubernetes/`, `.env.example`, `docs/production_hardening.md` | ⚠️ | Deployment files present; secrets management is documented as a known gap for prod |
| **P1** Problem clarity | `README.md` introduction + Architecture section | ✅ | Real-time scam/fraud call detection with multimodal AI |
| **P2** Baseline / benchmark rigor | `rawnet-service/evaluate_rawnet.py`, `scripts/evaluate_iep3.py`, `docs/evaluation_plan.md`, `docs/evaluation_results_template.md` | ⚠️ | Evaluation scripts present; fill `docs/evaluation_results_template.md` with real metrics before submission |
| **P3** AI justification / contribution | `README.md` (each IEP section), `docs/evaluation_plan.md` | ✅ | Each model justified; RawNet for raw-waveform spoof detection, DistilBERT for semantic threat, ECAPA for speaker identity |
| **P4** Value or publishability | `README.md` overview; multimodal fusion novelty | ✅ | Novel combination of three AI auditors for real-time fraud prevention |
| **Q1** Test suite breadth | `tests/` (fusion, validation, timeouts, identity, rawnet, semantic), `distilbert-service/test_main.py` | ✅ | Covers happy path, edge cases, fallback, invalid input |
| **Q2** Regression / validation strategy | `tests/` runnable without model weights; CI workflow `.github/workflows/test.yml` | ✅ | Mocked model weights; tests run on every push/PR |
| **G1** Commit history and ownership | GitHub commit history on `dev` branch | ✅ | Check via `git log --oneline` |
| **G2** Branching, review, traceability | `dev` branch, PR-based workflow | ✅ | CI runs on PR; branch protection recommended |
| **M1** Automated lifecycle pipeline | `.github/workflows/test.yml`, `.github/workflows/docker-build.yml` | ✅ | pytest + Docker build on every push and PR |
| **M2** Experiment tracking and thresholds | `rawnet-service/mlruns/` (MLflow runs), `mlflow.db`, `configs/iep3_identity.json` (threshold policy) | ✅ | MLflow experiment tracking present; IEP3 thresholds configurable |
| **M3** Monitoring and ML-specific signals | `monitoring/prometheus.yml`, `monitoring/grafana/`, `docs/observability.md`, `/metrics` endpoints on all three services | ✅ | Spoof probability, semantic score, identity decisions, fusion outcomes, active sessions, enrolled profiles |
| **M4** Documentation completeness | `README.md`, `readme_correction.md`, `docs/` directory | ✅ | Architecture, setup, demo, evaluation plan, observability, production hardening all documented |

---

## Known Limitations

1. **Native phone-call interception:** The mobile app simulates calls inside Trust-Call. It does not hook into Android's native dialer or intercept real phone calls. This is a demo/prototype.
2. **Public endpoints:** The GCP VM public IP (`35.189.221.158`) must be running at demo time. It requires team verification before the presentation.
3. **Authentication:** Public endpoints are currently unauthenticated (demo mode). Production would require API keys or OAuth. See `docs/production_hardening.md`.
4. **Secrets management:** No secret manager is integrated. Credentials are environment variables. `.env.example` documents them.
5. **Semantic model fallback:** If `./custom_scam_model` weights are missing in the DistilBERT container, the service falls back to the heuristic regex scorer. The heuristic still returns valid predictions.
6. **RawNet model weights:** Docker image builds without committed `.pth` files. Real inference requires providing `RAWNET_MODEL_PATH` at runtime; without weights, `/health` reports `model_ready=false` and `/predict` returns `503`.
7. **IEP3 profile storage:** Speaker profiles are stored in the backend's local filesystem, not encrypted on-device. This is a demo-only configuration.
8. **Whisper dependency:** Live IEP2 semantic scoring requires `faster-whisper` in the backend environment. Without it, transcription is skipped but the backend still runs.
9. **CORS:** `CORS_ORIGINS=*` is a demo default. Production should restrict to the frontend origin only.
10. **Rate limiting and request size limits:** Not yet enforced. Documented as P0/P1 items in `docs/production_hardening.md`.

---

## How to Run

### Local Docker Compose startup
```bash
docker compose up --build
```

### Health checks
```bash
curl http://localhost:8080/metrics        # EEP/IEP3 backend
curl http://localhost:8000/metrics        # RawNet IEP1
curl http://localhost:8002/metrics        # DistilBERT IEP2
curl http://localhost:9090/-/ready        # Prometheus
```

### Run tests
```bash
pip install pytest pytest-asyncio httpx fastapi pydantic
pytest tests/ -v
# Also run DistilBERT service tests (requires torch/transformers):
pip install -r distilbert-service/requirements.txt
pytest distilbert-service/test_main.py -v
```

### Automated evidence collection
```bash
python scripts/collect_demo_evidence.py
# Output saved to docs/demo_evidence/generated/
```

### View metrics dashboard
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default credentials `admin`/`admin`)

### Cloud smoke test
```bash
curl http://35.189.221.158:8080/metrics
curl http://35.189.221.158:8080/docs
```
