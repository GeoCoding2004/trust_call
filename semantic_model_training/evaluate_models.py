import pandas as pd
import re
import mlflow
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
# ==========================================
# 1. SETUP MLFLOW TRACKING
# ==========================================
mlflow.set_tracking_uri("sqlite:///E:/trust_call/mlflow.db")
mlflow.set_experiment("Semantic_Auditor_Evaluation_V2.1")

# ==========================================
# 2. LOAD THE EXACT 20% HOLDOUT EXAM DATA
# ==========================================
print("📂 Loading test dataset...")
df = pd.read_csv("final_training_dataset.csv")

_, val_texts, _, val_labels = train_test_split(
    df['text'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42
)

# ==========================================
# 3. DEFINE THE MODELS
# ==========================================
def regex_model(text):
    keywords = [r"bank", r"money", r"code", r"urgent", r"gift card", r"crypto", r"refund"]
    for word in keywords:
        if re.search(word, text.lower()): return 1
    return 0

print("🧠 Loading Untrained Base DistilBERT (Exact original state)...")
base_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
base_model_raw = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
base_pipeline = pipeline("text-classification", model=base_model_raw, tokenizer=base_tokenizer)

print("🧠 Loading Fine-Tuned DistilBERT...")
ai_pipeline = pipeline("text-classification", model="./custom_scam_model", tokenizer="./custom_scam_model")


# ==========================================
# 4. EVALUATION & LOGGING ENGINE
# ==========================================
def evaluate_and_log(run_name, texts, labels, model_func, is_ai=False):
    print(f"\n🚀 Running {run_name} on {len(texts)} real samples...")
    preds = []
    
    for t in texts:
        if is_ai:
            out = model_func(t, truncation=True, max_length=128)[0]
            # Convert string labels to binary integers
            preds.append(1 if out['label'] in ['LABEL_1', 'POSITIVE'] else 0)
        else:
            preds.append(model_func(t))
    
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset_size", len(texts))
        mlflow.log_param("model_type", "Neural Network" if is_ai else "Regex Heuristic")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        disp = ConfusionMatrixDisplay.from_predictions(
            labels, preds, display_labels=["Safe (0)", "Scam (1)"], cmap="Blues"
        )
        plt.title(f"{run_name}\nAccuracy: {acc*100:.1f}%")
        filename = f"{run_name}_confusion_matrix.png"
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.close()
        
    print(f"✅ Logged {run_name} to MLflow!")

# ==========================================
# 5. EXECUTE THE TESTS
# ==========================================
evaluate_and_log("Legacy_Regex_Heuristic", val_texts, val_labels, regex_model, is_ai=False)
evaluate_and_log("Untrained_Base_DistilBERT", val_texts, val_labels, base_pipeline, is_ai=True)
evaluate_and_log("FineTuned_DistilBERT_v1", val_texts, val_labels, ai_pipeline, is_ai=True)

print("\n🎉 All tests complete! Go check your browser at http://127.0.0.1:5000")