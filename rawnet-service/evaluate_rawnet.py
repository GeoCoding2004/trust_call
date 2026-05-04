import os
import glob
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, roc_auc_score

# Import the RawNet architecture
from model import RawNet 

# ==========================================
# 1. SETUP MLFLOW TRACKING
# ==========================================
mlflow.set_tracking_uri("sqlite:///E:/trust_call/mlflow.db")
mlflow.set_experiment("RawNet2_Deepfake_Audio_Evaluation")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Using device: {device}")

# ==========================================
# 2. LOAD THE RAWNET MODEL
# ==========================================
def load_model(weights_path):
    print(f"\n🧠 Loading RawNet2 Model from {weights_path}...")
    
    # ⚠️ THE FIX: Define d_args FRESH inside the function every time!
    d_args = {
        "nb_samp": 64000, "first_conv": 1024, "in_channels": 1, 
        "filts": [20, [20, 20], [20, 128], [128, 128]], "blocks": [2, 4], 
        "nb_fc_node": 1024, "gru_node": 1024, "nb_gru_layer": 3, "nb_classes": 2
    }
    
    model = RawNet(d_args=d_args, device=device)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        print(f"❌ ERROR: {weights_path} not found!")
        exit()
        
    model.to(device)
    model.eval()
    return model

# ==========================================
# 3. DYNAMIC DATA LOADING (FFmpeg Bypass)
# ==========================================
def get_test_data(data_folder="training_data", limit_per_class=100):
    """Scans the training_data folder for real and fake audio files."""
    print(f"📂 Scanning {data_folder} for test files...")
    test_cases = []
    
    # 0 = Fake/Spoof, 1 = Real (Matching your train_transfer.py logic)
    real_files = glob.glob(f"{data_folder}/real/*.wav")[:limit_per_class]
    fake_files = glob.glob(f"{data_folder}/fake/*.wav")[:limit_per_class]
    
    for f in real_files: test_cases.append({"path": f, "label": 1})
    for f in fake_files: test_cases.append({"path": f, "label": 0})
        
    print(f"✅ Loaded {len(real_files)} Real and {len(fake_files)} Fake files for the Final Exam.")
    return test_cases

def process_audio(file_path):
    """Loads audio safely using soundfile to prevent Windows FFmpeg crashes."""
    audio_data, sample_rate = sf.read(file_path)
    waveform = torch.tensor(audio_data, dtype=torch.float32)
    
    if waveform.ndim > 1:
        waveform = waveform.transpose(0, 1) 
        waveform = torch.mean(waveform, dim=0, keepdim=True) 
    else:
        waveform = waveform.unsqueeze(0) 
        
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        
    waveform = waveform.squeeze()

    target_length = 64000
    current_length = waveform.shape[0]
    if current_length > target_length:
        waveform = waveform[:target_length]
    elif current_length < target_length:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        
    return waveform.unsqueeze(0).to(device) # Shape: [1, 64000]

# ==========================================
# 4. EVALUATION & LOGGING ENGINE
# ==========================================
def evaluate_model(run_name, test_data, model):
    print(f"🚀 Running {run_name} evaluation...")
    
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for item in test_data:
            X_tensor = process_audio(item["path"])
            output = model(X_tensor) 
            probs = torch.exp(output) # Convert LogSoftmax to clean percentages
            
            score = probs[0][1].item() # Probability of being Real (1)
            pred = torch.argmax(probs, dim=1).item()
            
            y_true.append(item["label"])
            y_pred.append(pred)
            y_scores.append(score)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try: auc = roc_auc_score(y_true, y_scores)
    except: auc = 0.0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset_size", len(y_true))
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=["Fake/Spoof (0)", "Real (1)"], cmap="Oranges"
        )
        plt.title(f"{run_name}\nAccuracy: {acc*100:.1f}%")
        filename = f"{run_name}_cm.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
        
    print(f"✅ Logged {run_name}! Score: {acc*100:.1f}%\n")

# ==========================================
# 5. THE FINAL EXAM EXECUTION
# ==========================================
if __name__ == "__main__":
    test_dataset = get_test_data()
    
    # Test 1: The old baseline model
    pretrained_model = load_model("pre_trained_DF_model.pth")
    evaluate_model("ASVspoof_Baseline", test_dataset, pretrained_model)
    
    # Test 2: Your newly trained smart model
    finetuned_model = load_model("fine_tuned_DF_model.pth")
    evaluate_model("Fine_Tuned_HuggingFace", test_dataset, finetuned_model)
    
    print("🎉 Both models evaluated! Check http://127.0.0.1:5000 to compare them.")