import pandas as pd
import re
from transformers import pipeline

# ==========================================
# 1. THE OLD MODEL (Heuristic/Regex Baseline)
# ==========================================
def old_regex_model(text):
    """
    Simulates the old keyword-based approach.
    If it sees certain words, it panics and flags a scam.
    """
    keywords = [r"bank", r"money", r"code", r"urgent", r"gift card", r"crypto", r"refund"]
    text_lower = text.lower()
    
    for word in keywords:
        if re.search(word, text_lower):
            return 1 # Scam
    return 0 # Safe

# ==========================================
# 2. THE NEW MODEL (Fine-Tuned Neural Network)
# ==========================================
def load_new_model():
    print("🧠 Loading Fine-Tuned Semantic Auditor...")
    # This pipeline automatically handles the tokenization and prediction math
    semantic_pipeline = pipeline(
        "text-classification", 
        model="./custom_scam_model", 
        tokenizer="./custom_scam_model"
    )
    return semantic_pipeline

# ==========================================
# 3. THE EDGE CASE TEST SUITE
# ==========================================
# A mix of obvious scams, obvious safe texts, and tricky edge cases
test_cases = [
    {"text": "Hey mom, I'm at the bank right now getting money for the pizza.", "expected": 0, "type": "Tricky Safe"},
    {"text": "This is Amazon. We are issuing a refund for your purchase.", "expected": 1, "type": "Obvious Scam"},
    {"text": "Listen, I am stranded and the driver won't let me leave until I pay him, please help me out.", "expected": 1, "type": "Tricky Scam"},
    {"text": "Can you text me the Netflix verification code? It just locked me out.", "expected": 0, "type": "Tricky Safe"},
    {"text": "Your account is compromised. Transfer your crypto to the secure wallet immediately.", "expected": 1, "type": "Obvious Scam"},
    {"text": "Happy birthday! I got you an Apple gift card, check your email.", "expected": 0, "type": "Tricky Safe"}
]

# ==========================================
# 4. RUN THE COMPARISON
# ==========================================
def run_qa_test():
    new_model = load_new_model()
    results = []

    print("\n🚀 Running QA Model Comparison...\n")

    for test in test_cases:
        text = test["text"]
        expected = test["expected"]
        category = test["type"]

        # Run Old Model
        old_pred = old_regex_model(text)

        # Run New Model
        # The pipeline returns a list like: [{'label': 'LABEL_1', 'score': 0.98}]
        new_output = new_model(text)[0]
        # Convert 'LABEL_1' to 1, and 'LABEL_0' to 0
        new_pred = 1 if new_output['label'] == 'LABEL_1' else 0

        results.append({
            "Type": category,
            "Text": text[:40] + "...", # Truncate for clean table printing
            "Expected": expected,
            "Old Model": old_pred,
            "New Model": new_pred,
            "Old Correct?": "✅" if old_pred == expected else "❌",
            "New Correct?": "✅" if new_pred == expected else "❌"
        })

    # ==========================================
    # 5. PRINT THE RESULTS REPORT
    # ==========================================
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    old_acc = (df["Old Model"] == df["Expected"]).mean() * 100
    new_acc = (df["New Model"] == df["Expected"]).mean() * 100

    print("\n" + "="*40)
    print(f"📊 FINAL ACCURACY SCORE")
    print(f"Old Regex Model:   {old_acc:.1f}%")
    print(f"New Neural Model:  {new_acc:.1f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_qa_test()