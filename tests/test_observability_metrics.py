"""Tests for Prometheus-client metrics exposed by the backend observability module."""

from trust_call_backend import observability_metrics as metrics


def test_render_prometheus_client_metrics_includes_expected_metrics():
    metrics.trust_call_whisper_model_ready.set(1)
    metrics.trust_call_gateway_downstream_timeout_total.labels(service="rawnet").inc()
    metrics.trust_call_gateway_fusion_score.observe(0.72)
    metrics.trust_call_gateway_invalid_payload_total.labels(
        route="/identity/enroll",
        reason="validation_error",
    ).inc()

    rendered = metrics.render_prometheus_client_metrics()

    assert isinstance(rendered, str)
    assert "trust_call_whisper_model_ready" in rendered
    assert "trust_call_gateway_downstream_timeout_total" in rendered
    assert "trust_call_gateway_fusion_score" in rendered
    assert "trust_call_gateway_invalid_payload_total" in rendered
