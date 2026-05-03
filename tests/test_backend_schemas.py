import pytest
from pydantic import ValidationError

from trust_call_backend.schemas import (
    IdentityEnrollmentPayload,
    IdentityIdentificationPayload,
    LiveSessionEnrollmentPayload,
    Offer,
)


def test_valid_offer_passes():
    o = Offer(sdp='v=0', type='offer')
    assert o.type == 'offer'


def test_empty_sdp_fails():
    with pytest.raises(ValidationError):
        Offer(sdp='', type='offer')


def test_sdp_too_long_fails():
    with pytest.raises(ValidationError):
        Offer(sdp='x' * 200001, type='offer')


def test_base64_audio_too_long_fails():
    with pytest.raises(ValidationError):
        IdentityEnrollmentPayload(caller_id='c', base64_audio='a' * 12000001)


def test_caller_id_too_long_fails():
    with pytest.raises(ValidationError):
        IdentityEnrollmentPayload(caller_id='c' * 129, base64_audio='abc')


def test_top_k_bounds():
    with pytest.raises(ValidationError):
        IdentityIdentificationPayload(base64_audio='abc', top_k=0)
    with pytest.raises(ValidationError):
        IdentityIdentificationPayload(base64_audio='abc', top_k=11)


def test_score_bounds():
    with pytest.raises(ValidationError):
        LiveSessionEnrollmentPayload(synthetic_score=-0.1)
    with pytest.raises(ValidationError):
        LiveSessionEnrollmentPayload(synthetic_score=1.1)
    with pytest.raises(ValidationError):
        LiveSessionEnrollmentPayload(coercion_score=-0.1)
    with pytest.raises(ValidationError):
        LiveSessionEnrollmentPayload(coercion_score=1.1)
