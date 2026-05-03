from __future__ import annotations


def build_fusion_status(identity_result, synthetic_score: float, semantic_score: float) -> tuple[str, bool]:
    synthetic_threat = synthetic_score > 50.0
    semantic_threat = semantic_score >= 0.6
    identity_mismatch = identity_result.status in {"mismatch", "unknown_speaker"}
    identity_review = identity_result.status in {
        "review",
        "identity_candidate",
        "profile_incompatible",
    }
    identity_learning = identity_result.status == "candidate_collecting"

    if (synthetic_threat or semantic_threat) and (identity_mismatch or identity_review):
        return "THREAT DETECTED", True
    if synthetic_threat or semantic_threat:
        return "THREAT DETECTED", True
    if identity_mismatch:
        return "IDENTITY REVIEW", False
    if identity_review:
        return "IDENTITY CAUTION", False
    if identity_learning:
        return "LEARNING VOICE", False
    return "SAFE", False


def _clamp_unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def normalized_fusion_score(identity_result, synthetic_score: float, semantic_score: float) -> float:
    rawnet_component = _clamp_unit(float(synthetic_score) / 100.0)
    semantic_component = _clamp_unit(semantic_score)
    identity_status = getattr(identity_result, "status", "")
    if identity_status in {"mismatch", "unknown_speaker"}:
        identity_component = 1.0
    elif identity_status in {"review", "identity_candidate", "profile_incompatible"}:
        identity_component = 0.5
    else:
        identity_component = 0.0
    return max(rawnet_component, semantic_component, identity_component)


def dominant_signal(identity_result, synthetic_score: float, semantic_score: float) -> str:
    components = {
        "rawnet": _clamp_unit(float(synthetic_score) / 100.0),
        "semantic": _clamp_unit(semantic_score),
        "identity": normalized_fusion_score(identity_result, 0.0, 0.0),
    }
    strong_components = [name for name, value in components.items() if value >= 0.6]
    if len(strong_components) >= 2:
        return "multi_signal"

    name, value = max(components.items(), key=lambda item: item[1])
    if value <= 0.0:
        return "none"
    return name
