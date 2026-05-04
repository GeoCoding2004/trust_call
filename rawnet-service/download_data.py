import os
from datasets import load_dataset, Audio

# 1. Set a custom cache directory on your E: drive so it doesn't fill up your C: drive
cache_dir = "E:/trust_call/hf_cache"

os.makedirs("training_data/real", exist_ok=True)
os.makedirs("training_data/fake", exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

print("⏳ Reaching out to Hugging Face servers...")
# 2. Add the cache_dir parameter
dataset = load_dataset("garystafford/deepfake-audio-detection", split="train", cache_dir=cache_dir)
print(f"✅ Successfully found {len(dataset)} total audio files!")

# 3. THE FIX: Tell Hugging Face NOT to decode the audio. Just give us the raw bytes!
dataset = dataset.cast_column("audio", Audio(decode=False))

LIMIT = 500 
real_count = 0
fake_count = 0

print(f"💾 Extracting and saving {LIMIT} real and {LIMIT} fake files to your folders...")

# 4. Save the raw bytes directly to .wav files
for item in dataset:
    label = item["label"]
    audio_bytes = item["audio"]["bytes"] # Grabbing the raw file data
    
    if label == 0 and real_count < LIMIT:
        with open(f"training_data/real/real_{real_count}.wav", "wb") as f:
            f.write(audio_bytes)
        real_count += 1
        
    elif label == 1 and fake_count < LIMIT:
        with open(f"training_data/fake/fake_{fake_count}.wav", "wb") as f:
            f.write(audio_bytes)
        fake_count += 1
        
    if real_count >= LIMIT and fake_count >= LIMIT:
        break

print("🎉 Data processing complete!")
print(f"📁 Check your 'training_data' folder. You now have {real_count} real and {fake_count} fake audio clips ready for RawNet2!")