import base64
import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from model import RawNet


d_args = {
    "nb_samp": 64000,
    "first_conv": 1024,
    "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2,
}

SCORE_BUCKETS = (0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)

trust_call_rawnet_model_ready = Gauge(
    "trust_call_rawnet_model_ready",
    "1 if RawNet weights are loaded and prediction can run.",
)
trust_call_rawnet_predictions_total = Counter(
    "trust_call_rawnet_predictions_total",
    "RawNet prediction outcomes by decision.",
    labelnames=("decision",),
)
trust_call_rawnet_errors_total = Counter(
    "trust_call_rawnet_errors_total",
    "RawNet service errors by type.",
    labelnames=("error_type",),
)
trust_call_rawnet_fallback_total = Counter(
    "trust_call_rawnet_fallback_total",
    "Controlled fallback states encountered by RawNet.",
    labelnames=("reason",),
)
trust_call_rawnet_inference_latency_seconds = Histogram(
    "trust_call_rawnet_inference_latency_seconds",
    "Latency of RawNet prediction requests.",
    buckets=LATENCY_BUCKETS,
)
trust_call_rawnet_spoof_score_percent = Histogram(
    "trust_call_rawnet_spoof_score_percent",
    "Distribution of RawNet spoof probabilities.",
    buckets=SCORE_BUCKETS,
)
trust_call_rawnet_real_score_percent = Histogram(
    "trust_call_rawnet_real_score_percent",
    "Distribution of RawNet bonafide probabilities.",
    buckets=SCORE_BUCKETS,
)
trust_call_rawnet_audio_duration_seconds = Histogram(
    "trust_call_rawnet_audio_duration_seconds",
    "Duration of decoded audio inputs before truncation or padding.",
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)
trust_call_rawnet_audio_rms = Histogram(
    "trust_call_rawnet_audio_rms",
    "Loudness distribution of decoded audio inputs.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
trust_call_rawnet_silence_ratio = Histogram(
    "trust_call_rawnet_silence_ratio",
    "Fraction of near-silent samples in decoded audio.",
    buckets=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)
trust_call_rawnet_audio_samples = Histogram(
    "trust_call_rawnet_audio_samples",
    "Sample counts of decoded audio inputs before truncation or padding.",
    buckets=(4000, 8000, 16000, 32000, 64000, 128000, 256000),
)
trust_call_rawnet_preprocessing_errors_total = Counter(
    "trust_call_rawnet_preprocessing_errors_total",
    "Preprocessing failures in the RawNet service.",
    labelnames=("reason",),
)
trust_call_rawnet_low_quality_audio_total = Counter(
    "trust_call_rawnet_low_quality_audio_total",
    "Low-quality audio patterns observed by RawNet.",
    labelnames=("reason",),
)


def _record_quality_metrics(waveform: torch.Tensor, sample_rate: int) -> None:
    samples = waveform.detach().cpu().numpy().astype(np.float32).reshape(-1)
    sample_count = int(samples.size)
    trust_call_rawnet_audio_samples.observe(sample_count)
    if sample_rate > 0:
        duration_seconds = float(sample_count) / float(sample_rate)
        trust_call_rawnet_audio_duration_seconds.observe(duration_seconds)
        if duration_seconds < 0.5:
            trust_call_rawnet_low_quality_audio_total.labels(reason="too_short").inc()
    if sample_count == 0:
        trust_call_rawnet_low_quality_audio_total.labels(reason="empty_audio").inc()
        return

    peak = float(np.max(np.abs(samples)))
    if peak > 1.5:
        samples = samples / 32768.0

    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    silence_ratio = float(np.mean(np.abs(samples) <= 1e-3)) if samples.size else 1.0
    trust_call_rawnet_audio_rms.observe(rms)
    trust_call_rawnet_silence_ratio.observe(silence_ratio)

    if silence_ratio > 0.85:
        trust_call_rawnet_low_quality_audio_total.labels(reason="mostly_silent").inc()
    if peak >= 0.99:
        trust_call_rawnet_low_quality_audio_total.labels(reason="clipped").inc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.getenv("RAWNET_MODEL_PATH", "/models/fine_tuned_DF_model.pth")
    app.state.device = device
    app.state.model_path = model_path
    app.state.model = None
    app.state.model_ready = False
    app.state.model_error = None
    trust_call_rawnet_model_ready.set(0)

    print(f"--- Booting RawNet on {device.upper()} ---")
    if os.path.exists(model_path):
        try:
            model = RawNet(d_args, device)
            model = model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            app.state.model = model
            app.state.model_ready = True
            app.state.model_error = None
            trust_call_rawnet_model_ready.set(1)
            print(f"--- RawNet model loaded from {model_path} ---")
        except Exception as exc:
            app.state.model = None
            app.state.model_ready = False
            app.state.model_error = f"model load failed: {exc}"
            trust_call_rawnet_model_ready.set(0)
            trust_call_rawnet_errors_total.labels(error_type="model_load_failed").inc()
            trust_call_rawnet_fallback_total.labels(reason="model_load_failed").inc()
            print(f"WARNING: {app.state.model_error}")
    else:
        app.state.model = None
        app.state.model_ready = False
        app.state.model_error = f"model file not found: {model_path}"
        trust_call_rawnet_model_ready.set(0)
        trust_call_rawnet_fallback_total.labels(reason="model_file_missing").inc()
        print(f"WARNING: {app.state.model_error}")

    yield
    print("--- Shutting down RawNet service ---")
    app.state.model = None


app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


class AudioPayload(BaseModel):
    base64_audio: str = Field(..., min_length=1, max_length=12_000_000)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": bool(app.state.model_ready),
        "model_error": app.state.model_error,
        "model_path": app.state.model_path,
    }


@app.post("/predict")
async def predict(payload: AudioPayload):
    started = time.perf_counter()
    if not getattr(app.state, "model_ready", False):
        trust_call_rawnet_predictions_total.labels(decision="model_unavailable").inc()
        trust_call_rawnet_fallback_total.labels(reason="model_not_ready").inc()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "rawnet_model_unavailable",
                "message": "RawNet model weights are not loaded.",
                "model_path": getattr(app.state, "model_path", None),
            },
        )

    try:
        try:
            audio_bytes = base64.b64decode(payload.base64_audio)
        except Exception as exc:
            trust_call_rawnet_preprocessing_errors_total.labels(
                reason="base64_decode_failed"
            ).inc()
            trust_call_rawnet_errors_total.labels(error_type="decode_failed").inc()
            raise HTTPException(status_code=400, detail="invalid audio payload") from exc

        try:
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        except Exception as exc:
            trust_call_rawnet_preprocessing_errors_total.labels(
                reason="audio_decode_failed"
            ).inc()
            trust_call_rawnet_errors_total.labels(error_type="decode_failed").inc()
            raise HTTPException(status_code=400, detail="invalid audio payload") from exc

        waveform = torch.tensor(audio_data, dtype=torch.float32)

        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=1)

        _record_quality_metrics(waveform, sample_rate)

        if sample_rate != 16000:
            try:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            except Exception as exc:
                trust_call_rawnet_preprocessing_errors_total.labels(reason="resample_failed").inc()
                trust_call_rawnet_errors_total.labels(error_type="validation_failed").inc()
                raise HTTPException(status_code=400, detail="invalid audio payload") from exc

        target_length = 64000
        current_length = waveform.shape[0]
        if current_length <= 0:
            trust_call_rawnet_preprocessing_errors_total.labels(reason="invalid_audio").inc()
            trust_call_rawnet_errors_total.labels(error_type="validation_failed").inc()
            raise HTTPException(status_code=400, detail="invalid audio payload")
        if current_length > target_length:
            waveform = waveform[:target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        input_tensor = waveform.unsqueeze(0).to(app.state.device)
        with torch.no_grad():
            output = app.state.model(input_tensor)

        probabilities = torch.exp(output).cpu().squeeze().numpy()
        spoof_prob = float(probabilities[0] * 100)
        real_prob = float(probabilities[1] * 100)
        trust_call_rawnet_spoof_score_percent.observe(spoof_prob)
        trust_call_rawnet_real_score_percent.observe(real_prob)
        decision = "spoof" if spoof_prob >= 50.0 else "real"
        trust_call_rawnet_predictions_total.labels(decision=decision).inc()
        return {
            "status": "success",
            "spoof_probability_percent": round(spoof_prob, 2),
            "real_probability_percent": round(real_prob, 2),
        }
    except HTTPException:
        raise
    except Exception as exc:
        trust_call_rawnet_predictions_total.labels(decision="error").inc()
        trust_call_rawnet_errors_total.labels(error_type="inference_failed").inc()
        raise HTTPException(status_code=500, detail="rawnet inference failed") from exc
    finally:
        trust_call_rawnet_inference_latency_seconds.observe(time.perf_counter() - started)
