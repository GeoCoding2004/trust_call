# Observability Guide

Trust-Call monitoring follows a simple production-oriented path:

`/metrics` on each service -> Prometheus scrapes -> Grafana visualizes

This guide covers the current metrics inventory, Prometheus jobs, Grafana dashboard layout, validation steps, and known limitations.

## Monitoring Architecture

Relevant files:

- `monitoring/prometheus.yml`
- `monitoring/grafana/provisioning/datasources/prometheus.yml`
- `monitoring/grafana/provisioning/dashboards/dashboards.yml`
- `monitoring/grafana/dashboards/trust-call-ai-services.json`
- `docker-compose.yml`
- `rawnet-service/main.py`
- `distilbert-service/main.py`
- `trust_call_backend/server.py`
- `trust_call_backend/observability_metrics.py`
- `trust_call_backend/metrics_registry.py`
- `scripts/generate_grafana_dashboard.py`

Current scrape targets:

| Job | Target | Notes |
| --- | --- | --- |
| `prometheus` | `localhost:9090` | Prometheus self-scrape |
| `rawnet_audio_ai` | `rawnet-service:8000/metrics` | FastAPI HTTP metrics plus RawNet custom metrics |
| `distilbert_semantic_ai` | `distilbert-service:8002/metrics` | FastAPI HTTP metrics plus DistilBERT custom metrics |
| `iep3_identity_gateway` | `host.docker.internal:8080/metrics` | Existing custom backend counters plus Prometheus-client metrics |
| `cadvisor` | `cadvisor:8080/metrics` | Optional container resource metrics when the `cadvisor` compose profile is enabled |

Grafana provisioning:

- Datasource UID: `trustcall-prometheus`
- Dashboard UID: `trust-call-ai-services`
- Dashboard title: `Trust-Call AI Services`
- Provisioned from `monitoring/grafana/dashboards/trust-call-ai-services.json`

## Metrics Inventory

The backend keeps two metric surfaces:

- `trust_call_backend/metrics_registry.py` preserves the original custom counters and gauges already used by the demo.
- `trust_call_backend/observability_metrics.py` adds Prometheus-client counters, gauges, and histograms for production-oriented observability.

### Existing backend custom metrics

- `trust_call_gateway_sessions_started_total{source}`
- `trust_call_gateway_audio_frames_total`
- `trust_call_gateway_audio_chunks_total`
- `trust_call_iep3_identity_results_total{status}`
- `trust_call_iep3_candidate_embeddings_total`
- `trust_call_iep3_candidate_enrollments_total{status}`
- `trust_call_gateway_fusion_results_total{status}`
- `trust_call_gateway_errors_total{source}`
- `trust_call_gateway_active_sessions`
- `trust_call_iep3_enrolled_profiles`

### Readiness

- `trust_call_rawnet_model_ready`
- `trust_call_distilbert_model_ready`
- `trust_call_distilbert_model_mode{mode}`
- `trust_call_whisper_model_ready`
- `trust_call_iep3_model_ready`

### Traffic and session flow

- `trust_call_rawnet_predictions_total{decision}`
- `trust_call_distilbert_predictions_total{label,mode}`
- `trust_call_gateway_sessions_completed_total{result}`
- `trust_call_gateway_websocket_connections_total{status}`
- `trust_call_gateway_user_alerts_total{risk_level}`

### Latency

- `trust_call_rawnet_inference_latency_seconds`
- `trust_call_distilbert_inference_latency_seconds`
- `trust_call_gateway_rawnet_downstream_latency_seconds`
- `trust_call_gateway_distilbert_downstream_latency_seconds`
- `trust_call_gateway_fusion_latency_seconds`
- `trust_call_gateway_end_to_end_decision_latency_seconds`
- `trust_call_whisper_transcription_latency_seconds`
- `trust_call_iep3_identity_latency_seconds`

### RawNet

- `trust_call_rawnet_spoof_score_percent`
- `trust_call_rawnet_real_score_percent`
- `trust_call_rawnet_audio_duration_seconds`
- `trust_call_rawnet_audio_rms`
- `trust_call_rawnet_silence_ratio`
- `trust_call_rawnet_audio_samples`
- `trust_call_rawnet_preprocessing_errors_total{reason}`
- `trust_call_rawnet_low_quality_audio_total{reason}`
- `trust_call_rawnet_errors_total{error_type}`
- `trust_call_rawnet_fallback_total{reason}`

### DistilBERT

- `trust_call_distilbert_semantic_score`
- `trust_call_distilbert_text_length_chars`
- `trust_call_distilbert_empty_text_total`
- `trust_call_distilbert_flagged_keyword_groups_total{keyword_group}`
- `trust_call_distilbert_flagged_phrase_count`
- `trust_call_distilbert_errors_total{error_type}`
- `trust_call_distilbert_fallback_total{reason}`

### Whisper

- `trust_call_whisper_transcription_requests_total{result}`
- `trust_call_whisper_transcript_length_chars`
- `trust_call_whisper_empty_transcripts_total`

If Whisper is unavailable at runtime, readiness still reports that state and request metrics only appear after real transcription attempts.

### IEP3

- `trust_call_iep3_similarity_score`
- `trust_call_iep3_suspicious_identity_events_total{reason}`
- `trust_call_iep3_low_quality_voice_total{reason}`
- `trust_call_iep3_profile_updates_total{result}`

### EEP fusion and gateway decisions

- `trust_call_gateway_rawnet_spoof_score_percent`
- `trust_call_gateway_semantic_score`
- `trust_call_gateway_fusion_score`
- `trust_call_gateway_decision_source_total{dominant_signal}`
- `trust_call_gateway_conflict_cases_total{conflict_type}`
- `trust_call_gateway_degraded_decisions_total{missing_component}`
- `trust_call_gateway_component_missing_total{component}`
- `trust_call_gateway_downstream_timeout_total{service}`

### Data quality

- `trust_call_audio_too_short_total`
- `trust_call_audio_decode_failures_total{route}`
- `trust_call_audio_silence_ratio`
- `trust_call_transcript_empty_total`

### Security / abuse and request validation

- `trust_call_gateway_invalid_payload_total{route,reason}`
- `trust_call_gateway_oversized_payload_rejected_total{route}`

### Resource usage

These are provided by cAdvisor when enabled:

- `container_cpu_usage_seconds_total`
- `container_memory_usage_bytes`
- `container_network_receive_bytes_total`
- `container_network_transmit_bytes_total`

## Grafana Dashboard Organization

The dashboard is generated by `scripts/generate_grafana_dashboard.py` and written to `monitoring/grafana/dashboards/trust-call-ai-services.json`.

Rows:

1. `System Overview`
2. `Service Health & Model Readiness`
3. `Traffic & Session Flow`
4. `Latency`
5. `Errors, Timeouts & Fallbacks`
6. `RawNet Audio Spoofing`
7. `DistilBERT Semantic Scam Detection`
8. `Whisper / Transcription`
9. `IEP3 Identity Verification`
10. `EEP Fusion / Final Decision`
11. `Data Quality`
12. `Security / Abuse`
13. `Resource Usage`

Representative PromQL examples:

```promql
up{job=~"rawnet_audio_ai|distilbert_semantic_ai|iep3_identity_gateway"}
sum by (decision) (increase(trust_call_rawnet_predictions_total[15m]))
histogram_quantile(0.95, sum(rate(trust_call_distilbert_inference_latency_seconds_bucket[5m])) by (le))
sum by (component) (increase(trust_call_gateway_component_missing_total[15m]))
sum by (keyword_group) (increase(trust_call_distilbert_flagged_keyword_groups_total[15m]))
histogram_quantile(0.95, sum(rate(trust_call_gateway_fusion_score_bucket[15m])) by (le))
sum by (route, reason) (increase(trust_call_gateway_invalid_payload_total[1h]))
```

## Privacy and Labeling Rules

These metrics intentionally avoid high-cardinality and sensitive labels.

Not exported as labels:

- raw transcripts
- phone numbers
- caller IDs
- user names
- exact flagged phrases
- base64 audio
- full exception messages

Allowed labels stay bounded to values such as:

- `status`
- `service`
- `route`
- `reason`
- `component`
- `mode`
- `decision`
- `risk_level`
- `dominant_signal`

## How To Verify

Start the stack:

```bash
docker compose up
```

Enable cAdvisor if desired:

```bash
docker compose --profile cadvisor up
```

Check metrics endpoints:

```bash
curl http://localhost:8000/metrics
curl http://localhost:8002/metrics
curl http://localhost:8080/metrics
```

Check Prometheus:

```bash
curl http://localhost:9090/api/v1/targets
```

Open Grafana:

- `http://localhost:3000`

Regenerate the dashboard after edits:

```bash
python scripts/generate_grafana_dashboard.py
python -m json.tool monitoring/grafana/dashboards/trust-call-ai-services.json
```

## Known Limitations

- Metrics appear only after traffic or relevant events occur.
- No fake values or benchmark placeholders are exported.
- No raw transcript, raw audio, caller ID, or other PII labels are exported.
- Offline benchmark metrics remain documented separately and are not synthesized into live Prometheus series.
- Full container resource metrics require the optional cAdvisor service profile.
- Live deployments still need redeploy or restart before new metrics and dashboards appear on a VM.
