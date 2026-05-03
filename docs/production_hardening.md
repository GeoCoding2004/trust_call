# Production Hardening Guide

This document describes Trust-Call's current production-oriented features, known gaps, and a prioritised hardening plan.

---

## Current Production-Oriented Features

| Feature | Location |
|---|---|
| Service separation (3 AI microservices + gateway) | `docker-compose.yml`, individual `Dockerfile`s |
| Docker Compose orchestration | `docker-compose.yml` |
| Docker health checks | `docker-compose.yml` (`healthcheck` on rawnet, distilbert) |
| Prometheus metrics on every AI service | `rawnet-service/main.py`, `distilbert-service/main.py`, `trust_call_backend/server.py` (`/metrics`) |
| Grafana dashboard with provisioned datasource | `monitoring/grafana/provisioning/`, `monitoring/grafana/dashboards/` |
| Cloud deployment on GCP VM | `deployment/gcp/`, `README.md` Cloud Deployment section |
| Kubernetes manifests for portability | `deployment/kubernetes/` |
| AI service orchestration with parallel async calls | `server.py` `orchestrate_late_fusion` uses `asyncio.gather` |
| Timeout and fallback handling | `server.py` `fetch_rawnet` / `fetch_distilbert` – catch all exceptions, return safe default scores |
| Heuristic fallback in semantic service | `distilbert-service/main.py` – if model unavailable, uses regex phrase weights |
| Structured Pydantic request validation | All three services use Pydantic models |
| Custom HTTP error handlers | `distilbert-service/main.py` `RequestValidationError` handler |
| CORS middleware | `server.py` `CORSMiddleware` |
| Audio signal quality gating | `server.py` `measure_audio_quality` – skips weak audio before model inference |
| MLflow experiment tracking | `rawnet-service/mlruns/`, `mlflow.db` |
| IEP3 configurable threshold policy | `configs/iep3_identity.json` |
| Environment variable configuration | `docker-compose.yml` environment blocks; `.env.example` |

---

## Known Production Gaps

### Authentication & Authorization
- **Current:** No API key or token check on any endpoint. All endpoints are publicly accessible.
- **Risk:** Anyone can POST audio to `/offer`, enroll arbitrary voice profiles, or query identity data.
- **Status:** Demo-only. See P0 plan below.

### CORS Policy
- **Current:** CORS is configurable through `CORS_ORIGINS`. Demo defaults may still use `*`.
- **Risk:** A malicious web page could invoke the API from a user's browser.
- **Production requirement:** Set `CORS_ORIGINS` to trusted origins only.

### Rate Limiting
- **Current:** No rate limiting on any endpoint.
- **Risk:** Abuse, DoS, and model inference cost if deployed publicly.
- **Mitigation path:** Add `slowapi` middleware or a cloud load-balancer rate limit rule.

### Request Size Limits
- **Current:** Pydantic field-level request limits exist for SDP, caller IDs, session IDs, and base64 audio.
- **Remaining gap:** Add a full reverse-proxy body-size limit in production (nginx/ingress).

### Secret Management
- **Current:** Secrets are plain environment variables. No secret manager.
- **Risk:** Credentials in shell history, CI logs, or accidentally committed `.env` files.
- **Mitigation path:** GCP Secret Manager or AWS Secrets Manager; never commit real secrets.

### Identity/Voiceprint Data Encryption
- **Current:** Speaker embedding vectors stored as plain JSON/pickle files in `trust_call_backend/state/identity_profiles/`.
- **Risk:** Physical access to the server exposes biometric data.
- **Mitigation path:** Encrypt at rest (AES-256), move to secure on-device storage in production.

### PII and Consent
- **Current:** Voice profiles are stored without an explicit consent flow in the backend.
- **Risk:** GDPR / CCPA violation in a real deployment.
- **Mitigation path:** Require explicit user consent before storing a voiceprint; provide deletion endpoint (exists: `DELETE /identity/enrollment/{caller_id}`).

### Native Phone-Call Integration
- **Current:** The mobile app simulates calls inside Trust-Call. It does not hook into Android's native dialer or call screening APIs.
- **Production path:** Android `CallScreeningService` API + on-device audio routing.

### Model Drift Monitoring
- **Current:** No automated drift detection. Prometheus captures prediction counts but not score distributions over time.
- **Mitigation path:** Log prediction distributions to Prometheus histogram; add Grafana alert rules.

### Audit Logging
- **Current:** Python `print()` statements only; no structured JSON logs.
- **Mitigation path:** Replace with `structlog` or Python `logging` in JSON format; ship to cloud logging.

### TLS / HTTPS
- **Current:** Plain HTTP. Public demo VM exposes ports 8080, 3000, 9090 over HTTP.
- **Risk:** Traffic is unencrypted; voice audio is sent in cleartext.
- **Mitigation path:** Add nginx reverse proxy with TLS certificate (Let’s Encrypt).

---

## Recommended Hardening Plan

### P0 – Before final presentation

1. **Demo-mode API key check** – Planned but not implemented: API-key middleware. `ENABLE_DEMO_MODE`/`API_KEY` are documented env values, but middleware wiring is pending.
2. **`.env.example`** – Committed. All real secrets go in `.env` (git-ignored).
3. **Request size limit** – Add 10 MB body size cap to backend and services.
4. **Stricter CORS** – Set `CORS_ORIGINS` env var; restrict from `*` to app origin.
5. **Health/metrics checks** – Verify `/metrics` endpoints respond before demo.
6. **Demo evidence collection** – Run `scripts/collect_demo_evidence.py` and capture screenshots per `docs/demo_evidence/README.md`.

### P1 – After course submission

1. OAuth / auth provider (Auth0, Firebase Auth, or GCP IAP)
2. Encrypted identity profile store
3. Rate limiting (slowapi or cloud load balancer)
4. Structured JSON logs (structlog)
5. Model versioning in MLflow with promoted registry
6. Canary deployment via Kubernetes rollout strategy
7. Real telecom / native call integration (Android `CallScreeningService`)
8. TLS via nginx + Let’s Encrypt
9. GDPR consent flow for voiceprint storage

---

## Demo-Mode Auth Behavior (Planned)

When `ENABLE_DEMO_MODE=true` (default), the backend runs with no auth check – suitable for local testing and course demo.

When `ENABLE_DEMO_MODE=false`, a middleware will require:
```
X-API-Key: <value matching API_KEY env var>
```
Requests without a valid key return `401 Unauthorized`.

This behavior is not yet implemented; it is documented as a P0 item. Do not set `ENABLE_DEMO_MODE=false` in production until the middleware is wired up.


### RawNet Weights Handling (CI-safe)
- RawNet Docker image builds without bundled model weights.
- Real inference requires providing `RAWNET_MODEL_PATH` at runtime.
- If weights are absent, `/health` reports `model_ready=false` and `/predict` returns `503` (no fake inference).
