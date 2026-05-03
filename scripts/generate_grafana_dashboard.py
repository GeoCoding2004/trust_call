from __future__ import annotations

import json
from pathlib import Path


DASHBOARD_PATH = (
    Path(__file__).resolve().parents[1]
    / "monitoring"
    / "grafana"
    / "dashboards"
    / "trust-call-ai-services.json"
)
DATASOURCE = {"type": "prometheus", "uid": "trustcall-prometheus"}

PANEL_ID = 1
CURRENT_Y = 0


def next_id() -> int:
    global PANEL_ID
    value = PANEL_ID
    PANEL_ID += 1
    return value


def row(title: str) -> dict:
    global CURRENT_Y
    panel = {
        "collapsed": False,
        "datasource": DATASOURCE,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": CURRENT_Y},
        "id": next_id(),
        "panels": [],
        "title": title,
        "type": "row",
    }
    CURRENT_Y += 1
    return panel


def base_panel(title: str, expr: str, panel_type: str, unit: str | None = None) -> dict:
    field_defaults = {
        "color": {"mode": "palette-classic"},
        "mappings": [],
        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
    }
    if unit:
        field_defaults["unit"] = unit
    return {
        "datasource": DATASOURCE,
        "fieldConfig": {"defaults": field_defaults, "overrides": []},
        "id": next_id(),
        "options": {},
        "targets": [
            {
                "datasource": DATASOURCE,
                "editorMode": "code",
                "expr": expr,
                "legendFormat": "__auto",
                "range": True,
                "refId": "A",
            }
        ],
        "title": title,
        "type": panel_type,
    }


def stat_panel(title: str, expr: str, unit: str | None = None) -> dict:
    panel = base_panel(title, expr, "stat", unit)
    panel["fieldConfig"]["defaults"]["color"] = {"mode": "thresholds"}
    panel["fieldConfig"]["defaults"]["thresholds"] = {
        "mode": "absolute",
        "steps": [{"color": "red", "value": None}, {"color": "green", "value": 1}],
    }
    panel["options"] = {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "auto",
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        "textMode": "auto",
    }
    return panel


def timeseries_panel(title: str, expr: str, unit: str | None = None) -> dict:
    panel = base_panel(title, expr, "timeseries", unit)
    panel["fieldConfig"]["defaults"]["custom"] = {
        "axisCenteredZero": False,
        "axisColorMode": "text",
        "axisPlacement": "auto",
        "drawStyle": "line",
        "fillOpacity": 20,
        "gradientMode": "none",
        "lineInterpolation": "linear",
        "lineWidth": 2,
        "pointSize": 4,
        "scaleDistribution": {"type": "linear"},
        "showPoints": "auto",
        "spanNulls": False,
        "stacking": {"group": "A", "mode": "none"},
        "thresholdsStyle": {"mode": "off"},
    }
    panel["options"] = {
        "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
        "tooltip": {"mode": "single", "sort": "none"},
    }
    return panel


def bar_panel(title: str, expr: str) -> dict:
    panel = base_panel(title, expr, "bargauge")
    panel["options"] = {
        "displayMode": "gradient",
        "orientation": "horizontal",
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        "showUnfilled": True,
    }
    return panel


def table_panel(title: str, expr: str) -> dict:
    panel = base_panel(title, expr, "table")
    panel["options"] = {"showHeader": True}
    return panel


def layout_row(panels: list[dict], widths: list[int], height: int = 8) -> list[dict]:
    global CURRENT_Y
    x = 0
    for panel, width in zip(panels, widths, strict=True):
        panel["gridPos"] = {"h": height, "w": width, "x": x, "y": CURRENT_Y}
        x += width
    CURRENT_Y += height
    return panels


def build_dashboard() -> dict:
    panels: list[dict] = []

    panels.append(row("System Overview"))
    panels.extend(
        layout_row(
            [
                stat_panel(
                    "Service Up Status",
                    'up{job=~"rawnet_audio_ai|distilbert_semantic_ai|iep3_identity_gateway"}',
                ),
                stat_panel("Active Sessions", "trust_call_gateway_active_sessions"),
                stat_panel("Enrolled Profiles", "trust_call_iep3_enrolled_profiles"),
            ],
            [8, 8, 8],
            height=6,
        )
    )
    panels.extend(
        layout_row(
            [
                bar_panel(
                    "Fusion Outcomes Last 15m",
                    "sum by (status) (increase(trust_call_gateway_fusion_results_total[15m]))",
                ),
                bar_panel(
                    "Gateway Errors Last 15m",
                    "sum by (source) (increase(trust_call_gateway_errors_total[15m]))",
                ),
            ],
            [12, 12],
        )
    )

    panels.append(row("Service Health & Model Readiness"))
    panels.extend(
        layout_row(
            [
                stat_panel("RawNet Model Ready", "trust_call_rawnet_model_ready"),
                stat_panel("DistilBERT Model Ready", "trust_call_distilbert_model_ready"),
                stat_panel("DistilBERT Model Mode", "trust_call_distilbert_model_mode"),
                stat_panel("Whisper Model Ready", "trust_call_whisper_model_ready"),
                stat_panel("IEP3 Model Ready", "trust_call_iep3_model_ready"),
            ],
            [5, 5, 4, 5, 5],
            height=6,
        )
    )

    panels.append(row("Traffic & Session Flow"))
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "Gateway Audio Chunks/sec",
                    "sum(rate(trust_call_gateway_audio_chunks_total[1m]))",
                    "ops",
                ),
                timeseries_panel(
                    "RawNet Predictions/sec",
                    "sum by (decision) (rate(trust_call_rawnet_predictions_total[1m]))",
                    "ops",
                ),
                timeseries_panel(
                    "DistilBERT Predictions/sec",
                    "sum by (label, mode) (rate(trust_call_distilbert_predictions_total[1m]))",
                    "ops",
                ),
                timeseries_panel(
                    "IEP3 Identity Results/sec",
                    "sum by (status) (rate(trust_call_iep3_identity_results_total[1m]))",
                    "ops",
                ),
            ],
            [6, 6, 6, 6],
        )
    )

    panels.append(row("Latency"))
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "RawNet p95 Latency",
                    "histogram_quantile(0.95, sum(rate(trust_call_rawnet_inference_latency_seconds_bucket[5m])) by (le))",
                    "s",
                ),
                timeseries_panel(
                    "DistilBERT p95 Latency",
                    "histogram_quantile(0.95, sum(rate(trust_call_distilbert_inference_latency_seconds_bucket[5m])) by (le))",
                    "s",
                ),
                timeseries_panel(
                    "IEP3 p95 Latency",
                    "histogram_quantile(0.95, sum(rate(trust_call_iep3_identity_latency_seconds_bucket[5m])) by (le))",
                    "s",
                ),
            ],
            [8, 8, 8],
        )
    )
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "EEP End-to-End p95 Latency",
                    "histogram_quantile(0.95, sum(rate(trust_call_gateway_end_to_end_decision_latency_seconds_bucket[5m])) by (le))",
                    "s",
                ),
                timeseries_panel(
                    "Fusion p95 Latency",
                    "histogram_quantile(0.95, sum(rate(trust_call_gateway_fusion_latency_seconds_bucket[5m])) by (le))",
                    "s",
                ),
            ],
            [12, 12],
        )
    )

    panels.append(row("Errors, Timeouts & Fallbacks"))
    panels.extend(
        layout_row(
            [
                bar_panel(
                    "RawNet Errors Last 15m",
                    "sum by (error_type) (increase(trust_call_rawnet_errors_total[15m]))",
                ),
                bar_panel(
                    "DistilBERT Errors Last 15m",
                    "sum by (error_type) (increase(trust_call_distilbert_errors_total[15m]))",
                ),
                bar_panel(
                    "Downstream Timeouts Last 15m",
                    "sum by (service) (increase(trust_call_gateway_downstream_timeout_total[15m]))",
                ),
            ],
            [8, 8, 8],
        )
    )
    panels.extend(
        layout_row(
            [
                bar_panel(
                    "Degraded Decisions Last 15m",
                    "sum by (missing_component) (increase(trust_call_gateway_degraded_decisions_total[15m]))",
                ),
                bar_panel(
                    "Component Missing Last 15m",
                    "sum by (component) (increase(trust_call_gateway_component_missing_total[15m]))",
                ),
            ],
            [12, 12],
        )
    )

    panels.append(row("RawNet Audio Spoofing"))
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "Spoof Score p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_rawnet_spoof_score_percent_bucket[15m])) by (le))",
                ),
                timeseries_panel(
                    "Real Score p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_rawnet_real_score_percent_bucket[15m])) by (le))",
                ),
                bar_panel(
                    "RawNet Decisions Last 15m",
                    "sum by (decision) (increase(trust_call_rawnet_predictions_total[15m]))",
                ),
            ],
            [8, 8, 8],
        )
    )
    panels.extend(
        layout_row(
            [
                bar_panel(
                    "Low-Quality Audio Last 15m",
                    "sum by (reason) (increase(trust_call_rawnet_low_quality_audio_total[15m]))",
                ),
                timeseries_panel(
                    "RawNet Audio Duration p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_rawnet_audio_duration_seconds_bucket[15m])) by (le))",
                    "s",
                ),
                timeseries_panel(
                    "RawNet Silence Ratio p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_rawnet_silence_ratio_bucket[15m])) by (le))",
                ),
            ],
            [8, 8, 8],
        )
    )

    panels.append(row("DistilBERT Semantic Scam Detection"))
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "Semantic Score p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_distilbert_semantic_score_bucket[15m])) by (le))",
                ),
                bar_panel(
                    "Semantic Decisions Last 15m",
                    "sum by (label, mode) (increase(trust_call_distilbert_predictions_total[15m]))",
                ),
                timeseries_panel(
                    "Text Length p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_distilbert_text_length_chars_bucket[15m])) by (le))",
                ),
            ],
            [8, 8, 8],
        )
    )
    panels.extend(
        layout_row(
            [
                stat_panel(
                    "Empty Text Last 15m",
                    "sum(increase(trust_call_distilbert_empty_text_total[15m]))",
                ),
                bar_panel(
                    "Keyword Groups Last 15m",
                    "sum by (keyword_group) (increase(trust_call_distilbert_flagged_keyword_groups_total[15m]))",
                ),
            ],
            [6, 18],
        )
    )

    panels.append(row("Whisper / Transcription"))
    panels.extend(
        layout_row(
            [
                stat_panel("Whisper Model Ready", "trust_call_whisper_model_ready"),
                bar_panel(
                    "Transcription Requests Last 15m",
                    "sum by (result) (increase(trust_call_whisper_transcription_requests_total[15m]))",
                ),
                timeseries_panel(
                    "Transcription p95 Latency",
                    "histogram_quantile(0.95, sum(rate(trust_call_whisper_transcription_latency_seconds_bucket[5m])) by (le))",
                    "s",
                ),
            ],
            [6, 10, 8],
        )
    )
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "Transcript Length p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_whisper_transcript_length_chars_bucket[15m])) by (le))",
                ),
                stat_panel(
                    "Empty Transcripts Last 15m",
                    "sum(increase(trust_call_whisper_empty_transcripts_total[15m]))",
                ),
            ],
            [16, 8],
        )
    )

    panels.append(row("IEP3 Identity Verification"))
    panels.extend(
        layout_row(
            [
                stat_panel("Enrolled Profiles", "trust_call_iep3_enrolled_profiles"),
                bar_panel(
                    "Identity Results Last 15m",
                    "sum by (status) (increase(trust_call_iep3_identity_results_total[15m]))",
                ),
                timeseries_panel(
                    "Identity p95 Latency",
                    "histogram_quantile(0.95, sum(rate(trust_call_iep3_identity_latency_seconds_bucket[5m])) by (le))",
                    "s",
                ),
            ],
            [6, 10, 8],
        )
    )
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "Similarity Score p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_iep3_similarity_score_bucket[15m])) by (le))",
                ),
                bar_panel(
                    "Suspicious Identity Events Last 15m",
                    "sum by (reason) (increase(trust_call_iep3_suspicious_identity_events_total[15m]))",
                ),
            ],
            [12, 12],
        )
    )

    panels.append(row("EEP Fusion / Final Decision"))
    panels.extend(
        layout_row(
            [
                bar_panel(
                    "Fusion Outcomes Last 15m",
                    "sum by (status) (increase(trust_call_gateway_fusion_results_total[15m]))",
                ),
                timeseries_panel(
                    "Fusion Score p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_gateway_fusion_score_bucket[15m])) by (le))",
                ),
                bar_panel(
                    "Dominant Decision Source Last 15m",
                    "sum by (dominant_signal) (increase(trust_call_gateway_decision_source_total[15m]))",
                ),
            ],
            [8, 8, 8],
        )
    )
    panels.extend(
        layout_row(
            [
                bar_panel(
                    "Conflict Cases Last 15m",
                    "sum by (conflict_type) (increase(trust_call_gateway_conflict_cases_total[15m]))",
                ),
                bar_panel(
                    "User Alerts Last 15m",
                    "sum by (risk_level) (increase(trust_call_gateway_user_alerts_total[15m]))",
                ),
            ],
            [12, 12],
        )
    )

    panels.append(row("Data Quality"))
    panels.extend(
        layout_row(
            [
                stat_panel(
                    "Audio Too Short Last 15m",
                    "sum(increase(trust_call_audio_too_short_total[15m]))",
                ),
                bar_panel(
                    "Audio Decode Failures Last 15m",
                    "sum by (route) (increase(trust_call_audio_decode_failures_total[15m]))",
                ),
                timeseries_panel(
                    "Backend Audio Silence p95",
                    "histogram_quantile(0.95, sum(rate(trust_call_audio_silence_ratio_bucket[15m])) by (le))",
                ),
            ],
            [6, 10, 8],
        )
    )
    panels.extend(
        layout_row(
            [
                stat_panel(
                    "Empty Transcripts Last 15m",
                    "sum(increase(trust_call_transcript_empty_total[15m]))",
                ),
                table_panel(
                    "Invalid Payloads Last 15m",
                    "sum by (route, reason) (increase(trust_call_gateway_invalid_payload_total[15m]))",
                ),
                bar_panel(
                    "Oversized Payloads Rejected Last 15m",
                    "sum by (route) (increase(trust_call_gateway_oversized_payload_rejected_total[15m]))",
                ),
            ],
            [6, 10, 8],
        )
    )

    panels.append(row("Security / Abuse"))
    panels.extend(
        layout_row(
            [
                table_panel(
                    "Invalid Payloads by Route",
                    "sum by (route, reason) (increase(trust_call_gateway_invalid_payload_total[1h]))",
                ),
                bar_panel(
                    "Oversized Payloads by Route",
                    "sum by (route) (increase(trust_call_gateway_oversized_payload_rejected_total[1h]))",
                ),
            ],
            [12, 12],
        )
    )

    panels.append(row("Resource Usage"))
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "Container CPU Usage",
                    'sum(rate(container_cpu_usage_seconds_total{name=~".*trust-call.*"}[5m])) by (name)',
                    "percentunit",
                ),
                timeseries_panel(
                    "Container Memory Usage",
                    'container_memory_usage_bytes{name=~".*trust-call.*"}',
                    "bytes",
                ),
            ],
            [12, 12],
        )
    )
    panels.extend(
        layout_row(
            [
                timeseries_panel(
                    "Container Network Receive",
                    'sum(rate(container_network_receive_bytes_total{name=~".*trust-call.*"}[5m])) by (name)',
                    "Bps",
                ),
                timeseries_panel(
                    "Container Network Transmit",
                    'sum(rate(container_network_transmit_bytes_total{name=~".*trust-call.*"}[5m])) by (name)',
                    "Bps",
                ),
            ],
            [12, 12],
        )
    )

    return {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "liveNow": False,
        "panels": panels,
        "refresh": "10s",
        "schemaVersion": 39,
        "style": "dark",
        "tags": ["trust-call", "observability", "ai"],
        "templating": {"list": []},
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "timezone": "",
        "title": "Trust-Call AI Services",
        "uid": "trust-call-ai-services",
        "version": 1,
        "weekStart": "",
    }


def main() -> None:
    dashboard = build_dashboard()
    DASHBOARD_PATH.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
