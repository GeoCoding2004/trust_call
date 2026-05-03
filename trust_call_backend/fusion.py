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
