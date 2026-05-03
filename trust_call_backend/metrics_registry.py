from __future__ import annotations

from collections import defaultdict
from threading import Lock


def _escape_metric_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class MetricsRegistry:
    def __init__(self):
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._lock = Lock()

    def inc(self, name: str, amount: float = 1.0, **labels: str) -> None:
        label_key = tuple(sorted((key, str(label_value)) for key, label_value in labels.items()))
        with self._lock:
            self._counters[(name, label_key)] += amount

    def render(self, gauges: dict[str, float] | None = None) -> str:
        lines = [
            "# HELP trust_call_gateway_sessions_started_total Live call sessions started by source.",
            "# TYPE trust_call_gateway_sessions_started_total counter",
            "# HELP trust_call_gateway_audio_frames_total Audio frames received by the gateway.",
            "# TYPE trust_call_gateway_audio_frames_total counter",
            "# HELP trust_call_gateway_audio_chunks_total Audio chunks processed by the gateway.",
            "# TYPE trust_call_gateway_audio_chunks_total counter",
            "# HELP trust_call_iep3_identity_results_total IEP3 identity decisions by status.",
            "# TYPE trust_call_iep3_identity_results_total counter",
            "# HELP trust_call_iep3_candidate_embeddings_total TOFU candidate embeddings collected.",
            "# TYPE trust_call_iep3_candidate_embeddings_total counter",
            "# HELP trust_call_iep3_candidate_enrollments_total TOFU candidate enrollment outcomes.",
            "# TYPE trust_call_iep3_candidate_enrollments_total counter",
            "# HELP trust_call_gateway_fusion_results_total EEP fusion outcomes.",
            "# TYPE trust_call_gateway_fusion_results_total counter",
            "# HELP trust_call_gateway_errors_total Gateway errors recorded by source.",
            "# TYPE trust_call_gateway_errors_total counter",
        ]
        with self._lock:
            counters = list(self._counters.items())

        for (name, labels), value in sorted(counters):
            label_text = ""
            if labels:
                label_text = "{" + ",".join(
                    f'{key}="{_escape_metric_label(label_value)}"' for key, label_value in labels
                ) + "}"
            lines.append(f"{name}{label_text} {value}")

        if gauges:
            lines.extend([
                "# HELP trust_call_gateway_active_sessions Active live sessions retained in memory.",
                "# TYPE trust_call_gateway_active_sessions gauge",
                "# HELP trust_call_iep3_enrolled_profiles Stored local IEP3 speaker profiles.",
                "# TYPE trust_call_iep3_enrolled_profiles gauge",
            ])
            for name, value in sorted(gauges.items()):
                lines.append(f"{name} {value}")

        return "\n".join(lines) + "\n"
