from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest


LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
PERCENT_BUCKETS = (0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
UNIT_SCORE_BUCKETS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TEXT_LENGTH_BUCKETS = (0, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000)
SILENCE_BUCKETS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


trust_call_whisper_model_ready = Gauge(
    "trust_call_whisper_model_ready",
    "1 if the Whisper transcription model is loaded and available.",
)
trust_call_iep3_model_ready = Gauge(
    "trust_call_iep3_model_ready",
    "1 if the IEP3 identity auditor is initialized and can serve requests.",
)

trust_call_gateway_sessions_completed_total = Counter(
    "trust_call_gateway_sessions_completed_total",
    "Gateway sessions observed as completed by result.",
    labelnames=("result",),
)
trust_call_gateway_websocket_connections_total = Counter(
    "trust_call_gateway_websocket_connections_total",
    "WebSocket connection lifecycle events by status.",
    labelnames=("status",),
)
trust_call_gateway_user_alerts_total = Counter(
    "trust_call_gateway_user_alerts_total",
    "User-facing gateway alerts by risk level.",
    labelnames=("risk_level",),
)

trust_call_gateway_rawnet_downstream_latency_seconds = Histogram(
    "trust_call_gateway_rawnet_downstream_latency_seconds",
    "Latency for gateway calls to the RawNet service.",
    buckets=LATENCY_BUCKETS,
)
trust_call_gateway_distilbert_downstream_latency_seconds = Histogram(
    "trust_call_gateway_distilbert_downstream_latency_seconds",
    "Latency for gateway calls to the DistilBERT service.",
    buckets=LATENCY_BUCKETS,
)
trust_call_gateway_fusion_latency_seconds = Histogram(
    "trust_call_gateway_fusion_latency_seconds",
    "Latency for gateway fusion computation.",
    buckets=LATENCY_BUCKETS,
)
trust_call_gateway_end_to_end_decision_latency_seconds = Histogram(
    "trust_call_gateway_end_to_end_decision_latency_seconds",
    "Latency for end-to-end gateway decision orchestration.",
    buckets=LATENCY_BUCKETS,
)
trust_call_whisper_transcription_latency_seconds = Histogram(
    "trust_call_whisper_transcription_latency_seconds",
    "Latency for Whisper transcription calls.",
    buckets=LATENCY_BUCKETS,
)
trust_call_iep3_identity_latency_seconds = Histogram(
    "trust_call_iep3_identity_latency_seconds",
    "Latency for IEP3 identity operations.",
    buckets=LATENCY_BUCKETS,
)

trust_call_gateway_rawnet_spoof_score_percent = Histogram(
    "trust_call_gateway_rawnet_spoof_score_percent",
    "Gateway-observed RawNet spoof score distribution.",
    buckets=PERCENT_BUCKETS,
)
trust_call_gateway_semantic_score = Histogram(
    "trust_call_gateway_semantic_score",
    "Gateway-observed DistilBERT semantic risk score distribution.",
    buckets=UNIT_SCORE_BUCKETS,
)
trust_call_gateway_fusion_score = Histogram(
    "trust_call_gateway_fusion_score",
    "Normalized fusion score distribution.",
    buckets=UNIT_SCORE_BUCKETS,
)
trust_call_iep3_similarity_score = Histogram(
    "trust_call_iep3_similarity_score",
    "IEP3 similarity and match-confidence score distribution.",
    buckets=UNIT_SCORE_BUCKETS,
)

trust_call_gateway_downstream_timeout_total = Counter(
    "trust_call_gateway_downstream_timeout_total",
    "Downstream service timeouts recorded by the gateway.",
    labelnames=("service",),
)
trust_call_gateway_component_missing_total = Counter(
    "trust_call_gateway_component_missing_total",
    "Gateway component degradations due to unavailable or failed dependencies.",
    labelnames=("component",),
)
trust_call_gateway_degraded_decisions_total = Counter(
    "trust_call_gateway_degraded_decisions_total",
    "Degraded gateway decisions caused by a missing or weak component.",
    labelnames=("missing_component",),
)
trust_call_gateway_decision_source_total = Counter(
    "trust_call_gateway_decision_source_total",
    "Dominant signal contributing to final gateway decisions.",
    labelnames=("dominant_signal",),
)
trust_call_gateway_conflict_cases_total = Counter(
    "trust_call_gateway_conflict_cases_total",
    "Conflicting evidence cases seen by the fusion engine.",
    labelnames=("conflict_type",),
)
trust_call_gateway_invalid_payload_total = Counter(
    "trust_call_gateway_invalid_payload_total",
    "Invalid payloads rejected by route and bounded reason.",
    labelnames=("route", "reason"),
)
trust_call_gateway_oversized_payload_rejected_total = Counter(
    "trust_call_gateway_oversized_payload_rejected_total",
    "Oversized payloads rejected by route.",
    labelnames=("route",),
)

trust_call_whisper_transcription_requests_total = Counter(
    "trust_call_whisper_transcription_requests_total",
    "Whisper transcription requests by result.",
    labelnames=("result",),
)
trust_call_whisper_transcript_length_chars = Histogram(
    "trust_call_whisper_transcript_length_chars",
    "Transcript length distribution produced by Whisper.",
    buckets=TEXT_LENGTH_BUCKETS,
)
trust_call_whisper_empty_transcripts_total = Counter(
    "trust_call_whisper_empty_transcripts_total",
    "Empty transcript results returned by Whisper.",
)
trust_call_audio_too_short_total = Counter(
    "trust_call_audio_too_short_total",
    "Audio payloads rejected or downgraded because they are too short.",
)
trust_call_audio_decode_failures_total = Counter(
    "trust_call_audio_decode_failures_total",
    "Audio decode failures by route.",
    labelnames=("route",),
)
trust_call_audio_silence_ratio = Histogram(
    "trust_call_audio_silence_ratio",
    "Observed silence ratio for backend audio payloads.",
    buckets=SILENCE_BUCKETS,
)
trust_call_transcript_empty_total = Counter(
    "trust_call_transcript_empty_total",
    "Empty or insufficient transcript contexts seen by the backend.",
)

trust_call_iep3_suspicious_identity_events_total = Counter(
    "trust_call_iep3_suspicious_identity_events_total",
    "Suspicious IEP3 identity events by reason.",
    labelnames=("reason",),
)
trust_call_iep3_low_quality_voice_total = Counter(
    "trust_call_iep3_low_quality_voice_total",
    "Low-quality voice conditions surfaced by IEP3.",
    labelnames=("reason",),
)
trust_call_iep3_profile_updates_total = Counter(
    "trust_call_iep3_profile_updates_total",
    "IEP3 profile enrollment and update outcomes.",
    labelnames=("result",),
)


def clamp_unit_interval(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def clamp_percentage(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 100.0))


def render_prometheus_client_metrics() -> str:
    return generate_latest().decode("utf-8")
