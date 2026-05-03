"""Unit tests for the production fusion logic."""

from trust_call_backend.fusion import build_fusion_status


class _FakeIdentityResult:
    def __init__(self, status: str):
        self.status = status
        self.caller_id = "test_caller"


class TestBuildFusionStatus:
    def test_all_low_risk_returns_safe(self):
        assert build_fusion_status(_FakeIdentityResult("verified"), 10.0, 0.1) == ("SAFE", False)

    def test_high_spoof_alone_is_threat(self):
        assert build_fusion_status(_FakeIdentityResult("verified"), 75.0, 0.1) == ("THREAT DETECTED", True)

    def test_high_semantic_alone_is_threat(self):
        assert build_fusion_status(_FakeIdentityResult("verified"), 10.0, 0.9) == ("THREAT DETECTED", True)

    def test_identity_mismatch_alone_is_review(self):
        assert build_fusion_status(_FakeIdentityResult("mismatch"), 10.0, 0.1) == ("IDENTITY REVIEW", False)

    def test_identity_review_status_gives_caution(self):
        assert build_fusion_status(_FakeIdentityResult("review"), 10.0, 0.1) == ("IDENTITY CAUTION", False)

    def test_candidate_collecting_gives_learning(self):
        assert build_fusion_status(_FakeIdentityResult("candidate_collecting"), 10.0, 0.1) == ("LEARNING VOICE", False)

    def test_boundary_spoof_exactly_50_not_threat(self):
        assert build_fusion_status(_FakeIdentityResult("verified"), 50.0, 0.1) == ("SAFE", False)

    def test_boundary_semantic_exactly_06_is_threat(self):
        assert build_fusion_status(_FakeIdentityResult("verified"), 10.0, 0.6) == ("THREAT DETECTED", True)

    def test_unknown_speaker_identity_gives_review(self):
        assert build_fusion_status(_FakeIdentityResult("unknown_speaker"), 5.0, 0.1) == ("IDENTITY REVIEW", False)
