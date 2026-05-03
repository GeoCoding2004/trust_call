"""Tests for production identity/metrics support utilities."""

from trust_call_backend.metrics_registry import MetricsRegistry


def test_metrics_registry_increment_counter():
    metrics = MetricsRegistry()

    metrics.inc("trust_call_gateway_errors_total")

    rendered = metrics.render()
    assert "trust_call_gateway_errors_total 1.0" in rendered


def test_metrics_registry_increment_by_amount():
    metrics = MetricsRegistry()

    metrics.inc("trust_call_gateway_audio_frames_total", amount=5.0)

    rendered = metrics.render()
    assert "trust_call_gateway_audio_frames_total 5.0" in rendered


def test_metrics_registry_separates_labels():
    metrics = MetricsRegistry()

    metrics.inc("trust_call_iep3_identity_results_total", status="match")
    metrics.inc("trust_call_iep3_identity_results_total", status="mismatch")

    rendered = metrics.render()

    assert 'trust_call_iep3_identity_results_total{status="match"} 1.0' in rendered
    assert 'trust_call_iep3_identity_results_total{status="mismatch"} 1.0' in rendered


def test_metrics_registry_escapes_label_values():
    metrics = MetricsRegistry()

    metrics.inc("trust_call_gateway_errors_total", source='bad"value')

    rendered = metrics.render()

    assert 'source="bad\\"value"' in rendered


def test_metrics_registry_renders_gauges():
    metrics = MetricsRegistry()

    rendered = metrics.render(
        gauges={
            "trust_call_gateway_active_sessions": 2,
            "trust_call_iep3_enrolled_profiles": 3,
        }
    )

    assert "trust_call_gateway_active_sessions 2" in rendered
    assert "trust_call_iep3_enrolled_profiles 3" in rendered
