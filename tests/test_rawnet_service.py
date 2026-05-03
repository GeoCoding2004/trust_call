import pytest
pytestmark = pytest.mark.requires_models

"""
tests/test_rawnet_service.py

Unit tests for the RawNet service (rawnet-service/main.py).
Model inference is mocked so no GPU or model weights are required.
"""
import io
import base64
import sys
import types
import math
import struct
import wave
import pytest


# ----------------------------------------------------------------
# Stub out torch and torchaudio before importing main
# ----------------------------------------------------------------

def _make_torch_stub():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float32 = "float32"
    torch.no_grad = lambda: __import__("contextlib").nullcontext()

    class FakeTensor:
        def __init__(self, data=None):
            self._d = data if data is not None else []
        def unsqueeze(self, dim): return self
        def to(self, device): return self
        @property
        def shape(self): return (64000,)
        @property
        def ndim(self): return 1

    torch.tensor = lambda *a, **kw: FakeTensor()
    torch.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(pad=lambda t, p: t),
        Module=object,
    )

    class FakeExp:
        def cpu(self): return self
        def squeeze(self): return self
        def numpy(self): return [0.3, 0.7]  # spoof=30%, real=70%

    torch.exp = lambda x: FakeExp()

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional
    return torch


def _make_torchaudio_stub():
    ta = types.ModuleType("torchaudio")
    ta_transforms = types.ModuleType("torchaudio.transforms")

    class FakeResampler:
        def __init__(self, orig_freq, new_freq): pass
        def __call__(self, x): return x

    ta_transforms.Resample = FakeResampler
    ta.transforms = ta_transforms
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.transforms"] = ta_transforms
    return ta


def _make_model_stub():
    model_mod = types.ModuleType("model")

    class FakeRawNet:
        def __init__(self, *a, **kw): pass
        def to(self, device): return self
        def eval(self): return self
        def load_state_dict(self, *a, **kw): pass
        def __call__(self, x):
            import types as _t
            class FakeOut:
                pass
            return FakeOut()

    model_mod.RawNet = FakeRawNet
    sys.modules["model"] = model_mod
    return model_mod


def _make_prom_stub():
    prom = types.ModuleType("prometheus_fastapi_instrumentator")
    class FakeInst:
        def instrument(self, app): return self
        def expose(self, app): return self
    prom.Instrumentator = FakeInst
    sys.modules["prometheus_fastapi_instrumentator"] = prom


if "torch" not in sys.modules:
    _make_torch_stub()
    _make_torchaudio_stub()
    _make_model_stub()
    _make_prom_stub()


# ----------------------------------------------------------------
# Helper: generate a minimal valid WAV file as base64
# ----------------------------------------------------------------

def _make_wav_base64(duration_seconds: float = 1.0, sample_rate: int = 16000) -> str:
    n = int(sample_rate * duration_seconds)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------
# Tests that use real soundfile + mocked torch
# ----------------------------------------------------------------

class TestRawNetValidation:

    def test_valid_base64_wav_decodes(self):
        """A valid WAV base64 payload should decode and be readable by soundfile."""
        import soundfile as sf
        b64 = _make_wav_base64()
        audio_bytes = base64.b64decode(b64)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        assert sr == 16000
        assert len(audio_data) > 0

    def test_invalid_base64_raises_exception(self):
        """Non-audio base64 payload should raise an exception (not return silently)."""
        import soundfile as sf
        with pytest.raises(Exception):
            sf.read(io.BytesIO(b"this is not audio data"))

    def test_generated_wav_is_correct_length(self):
        """A 1-second 16kHz WAV should have exactly 16000 samples."""
        import soundfile as sf
        b64 = _make_wav_base64(1.0, 16000)
        audio_data, sr = sf.read(io.BytesIO(base64.b64decode(b64)))
        assert len(audio_data) == 16000

    def test_4s_wav_meets_rawnet_input_requirement(self):
        """RawNet expects 64000 samples (4 s @ 16kHz)."""
        import soundfile as sf
        b64 = _make_wav_base64(4.0, 16000)
        audio_data, sr = sf.read(io.BytesIO(base64.b64decode(b64)))
        assert len(audio_data) == 64000
