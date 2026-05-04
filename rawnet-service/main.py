import base64
import io
import torch
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from model import RawNet

from prometheus_fastapi_instrumentator import Instrumentator

# 1. Define RawNet2 Architecture Configuration
d_args = {
    "nb_samp": 64000,
    "first_conv": 1024,
    "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2
}

# 2. The "Warm Start" Lifespan Event
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Booting AI Engine on {device.upper()} ---")
    
    model = RawNet(d_args, device)
    model = model.to(device)
    #model.load_state_dict(torch.load("pre_trained_DF_model.pth", map_location=device, weights_only=True))
    model.load_state_dict(torch.load("fine_tuned_DF_model.pth", map_location=device, weights_only=True))
    model.eval()
    
    app.state.model = model
    app.state.device = device
    print("--- Model successfully loaded and locked into RAM! ---")
    
    yield
    print("--- Shutting down and clearing RAM ---")
    app.state.model = None

app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)




class AudioPayload(BaseModel):
    base64_audio: str

# 4. The Inference Endpoint
@app.post("/predict")
async def predict(payload: AudioPayload):
    try:
        # Decode Base64 straight into volatile RAM
        audio_bytes = base64.b64decode(payload.base64_audio)
        
        # BYPASS FFMPEG: Use soundfile to read the RAM bytes
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Convert numpy array to PyTorch float32 tensor
        waveform = torch.tensor(audio_data, dtype=torch.float32)
        
        # Force Mono channel
        if waveform.ndim > 1:
            waveform = torch.mean(waveform, dim=1)
            
        # Force exactly 16kHz sample rate
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            
        # Pad or Truncate to exactly 64,000 samples (4 seconds)
        target_length = 64000
        current_length = waveform.shape[0]
        
        if current_length > target_length:
            waveform = waveform[:target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Add the batch dimension [1, 64000] and send to device
        input_tensor = waveform.unsqueeze(0).to(app.state.device)
        
        # Run lightning-fast inference using the warm model
        with torch.no_grad():
            output = app.state.model(input_tensor)
            
        # Convert log-softmax output to clean percentages
        probabilities = torch.exp(output).cpu().squeeze().numpy()
        spoof_prob = float(probabilities[0] * 100)
        real_prob = float(probabilities[1] * 100)
        
        return {
            "status": "success",
            "spoof_probability_percent": round(spoof_prob, 2),
            "real_probability_percent": round(real_prob, 2)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc() # Prints exact error to your server terminal for debugging
        raise HTTPException(status_code=500, detail=str(e))