import pandas as pd
from datasets import load_dataset

def build_training_matrix():
    print("🚀 Starting dataset construction...")

    # 1. Load your untouched synthetic data (Class 1)
    df_synthetic = pd.read_csv('synthetic_scams.csv')
    df_synthetic['label'] = 1
    num_synthetic = len(df_synthetic)

    # 2. Load the SMS dataset (contains BOTH Scams and Safe messages)
    print("Fetching ucirvine/sms_spam...")
    # Note: Because you downgraded datasets, we don't need trust_remote_code anymore
    sms = load_dataset('ucirvine/sms_spam', split='train').to_pandas()
    sms = sms.rename(columns={'sms': 'text'})

    # 3. Separate into Scams (Class 1) and Safe (Class 0)
    df_public_scams = sms[sms['label'] == 1][['text', 'label']]
    df_public_safe = sms[sms['label'] == 0][['text', 'label']]

    # --- SAVE PUBLIC DATA FOR INSPECTION ---
    df_public_scams.to_csv('public_scams_only.csv', index=False)
    df_public_safe.to_csv('public_safe_only.csv', index=False)
    # ---------------------------------------

    # 4. Math & Balancing
    total_scams = num_synthetic + len(df_public_scams)
    
    # Sample exactly enough safe messages to perfectly match the total scams
    df_safe_sampled = df_public_safe.sample(n=total_scams, random_state=42)

    # 5. Merge, shuffle, and save
    df_final = pd.concat([df_synthetic, df_public_scams, df_safe_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_final.to_csv('final_training_dataset.csv', index=False)
    print("\n✅ Final dataset built successfully!")
    print("=====================================")
    print(f"Total Rows: {len(df_final)}")
    print(f"-> {num_synthetic} Synthetic Scams")
    print(f"-> {len(df_public_scams)} Public SMS Scams")
    print(f"-> {len(df_safe_sampled)} Public Safe Messages")
    print("=====================================")

if __name__ == '__main__':
    build_training_matrix()