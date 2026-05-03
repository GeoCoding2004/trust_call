import base64
import io
import os
from contextlib import asynccontextmanager

import soundfile as sf
import torch
import torchaudio.transforms as T
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.getenv("RAWNET_MODEL_PATH", "/models/fine_tuned_DF_model.pth")
    app.state.device = device
    app.state.model_path = model_path
    app.state.model = None
    app.state.model_ready = False
    app.state.model_error = None

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
            print(f"--- RawNet model loaded from {model_path} ---")
        except Exception as exc:
            app.state.model = None
            app.state.model_ready = False
            app.state.model_error = f"model load failed: {exc}"
            print(f"WARNING: {app.state.model_error}")
    else:
        app.state.model = None
        app.state.model_ready = False
        app.state.model_error = f"model file not found: {model_path}"
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
    if not getattr(app.state, "model_ready", False):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "rawnet_model_unavailable",
                "message": "RawNet model weights are not loaded.",
                "model_path": getattr(app.state, "model_path", None),
            },
        )

    try:
        audio_bytes = base64.b64decode(payload.base64_audio)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        waveform = torch.tensor(audio_data, dtype=torch.float32)

        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=1)

        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        target_length = 64000
        current_length = waveform.shape[0]
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
        return {
            "status": "success",
            "spoof_probability_percent": round(spoof_prob, 2),
            "real_probability_percent": round(real_prob, 2),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
