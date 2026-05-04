import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import soundfile as sf
import mlflow  # <--- NEW: Import MLflow
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
    "nb_classes": 2
}

# ==========================================
# NEW: SETUP MLFLOW
# ==========================================
mlflow.set_tracking_uri("sqlite:///E:/trust_call/mlflow.db")
mlflow.set_experiment("RawNet2_Deepfake_Training")

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.filepaths = glob.glob(f"{real_dir}/*.wav") + glob.glob(f"{fake_dir}/*.wav")
        self.labels = [1 if "real" in path else 0 for path in self.filepaths]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        audio_data, sample_rate = sf.read(path)
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
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return waveform, label

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Initializing Transfer Learning on {device.upper()}...")

    dataset = DeepfakeDataset("training_data/real", "training_data/fake")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = RawNet(d_args, device).to(device)
    
    # Ensure this file exists before running!
    model.load_state_dict(torch.load("pre_trained_DF_model.pth", map_location=device, weights_only=True))
    
    print("❄️ Freezing feature extraction layers...")
    for name, param in model.named_parameters():
        if "gru" in name or "fc1_gru" in name or "fc2_gru" in name:
            param.requires_grad = True  
        else:
            param.requires_grad = False 

    learning_rate = 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.NLLLoss() 

    epochs = 20

    # ==========================================
    # NEW: START MLFLOW RUN
    # ==========================================
    with mlflow.start_run(run_name="RawNet2_Transfer_Learning_Run"):
        # Log the settings we used
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("dataset_size", len(dataset))

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for batch_idx, (waveforms, labels) in enumerate(dataloader):
                waveforms, labels = waveforms.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                
            accuracy = 100 * correct / len(dataset)
            avg_loss = total_loss/len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

            # ==========================================
            # NEW: BEAM DATA TO DASHBOARD AFTER EVERY EPOCH
            # ==========================================
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", accuracy, step=epoch)

        # Save local file
        torch.save(model.state_dict(), "fine_tuned_DF_model.pth")
        print("🎉 Training Complete! Saved new weights as 'fine_tuned_DF_model.pth'")
        
        # ==========================================
        # NEW: UPLOAD FINAL MODEL TO MLFLOW REGISTRY
        # ==========================================
        mlflow.log_artifact("fine_tuned_DF_model.pth")

if __name__ == "__main__":
    train()