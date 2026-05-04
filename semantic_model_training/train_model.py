import pandas as pd
import torch
import os        # NEW
import mlflow    # NEW
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# NEW: MLFLOW ENVIRONMENT SETUP
# ==========================================
# We use environment variables so Hugging Face automatically detects your local server
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///E:/trust_call/mlflow.db"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Semantic_Auditor_Training"

# ==========================================
# 1. DATA PREPARATION & SPLITTING
# ==========================================
print("🚀 Loading dataset...")
df = pd.read_csv("final_training_dataset.csv")

# We hold back 20% of the data to "test" the AI after it studies
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42
)

# ==========================================
# 2. TOKENIZATION (TEXT -> NUMBERS)
# ==========================================
print("⚙️ Tokenizing text data...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class ScamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ScamDataset(train_encodings, train_labels)
val_dataset = ScamDataset(val_encodings, val_labels)

# ==========================================
# 3. AI METRICS (HOW WE GRADE IT)
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ==========================================
# 4. MODEL ARCHITECTURE & TRAINING LOOP
# ==========================================
print("🧠 Loading Base Model into RAM...")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",       
    save_strategy="epoch",
    load_best_model_at_end=True,     
    report_to="mlflow"               # NEW: Tells Hugging Face to beam stats directly to MLflow!
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("🔥 STARTING TRAINING LOOP...")
trainer.train()

# ==========================================
# 5. EXPORT THE FINAL BRAIN
# ==========================================
print("💾 Saving the fine-tuned model...")
model.save_pretrained("./custom_scam_model")
tokenizer.save_pretrained("./custom_scam_model")
print("✅ Training complete! Model saved to ./custom_scam_model")