import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def run_error_analysis():
    # Load features
    df = pd.read_csv("phase3_results/train_features.csv")
    df['target'] = (df['label'] == 'hallucinated').astype(int)
    df = df.fillna(0.0)
    
    h0_cols = [c for c in df.columns if 'h0' in c]
    h1_cols = [c for c in df.columns if 'h1' in c]
    X_Combined = df[h0_cols + h1_cols]
    y = df['target']
    
    # Train a single LR model on all data to get errors for analysis
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_Combined, y)
    preds = lr.predict(X_Combined)
    
    df['pred'] = preds
    
    # False Positives: Model predicted 1 (hallucinated), True label 0 (grounded)
    fp = df[(df['target'] == 0) & (df['pred'] == 1)]
    # False Negatives: Model predicted 0 (grounded), True label 1 (hallucinated)
    fn = df[(df['target'] == 1) & (df['pred'] == 0)]
    
    # Get original dataset to map back to text
    from datasets import load_dataset
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:2000]")
    raw_data = pd.DataFrame(dataset)
    
    with open("error_analysis_report.txt", "w") as f:
        f.write("--- False Positives (Predicted Hallucinated, Actually Grounded) ---\n\n")
        for idx in fp['example_id'].head(5).values:
            row = raw_data.iloc[idx]
            f.write(f"Knowledge: {row['knowledge'][:200]}...\n")
            f.write(f"Question: {row['question']}\n")
            f.write(f"Answer: {row['right_answer']}\n")
            f.write("-" * 50 + "\n")
            
        f.write("\n\n--- False Negatives (Predicted Grounded, Actually Hallucinated) ---\n\n")
        for idx in fn['example_id'].head(5).values:
            row = raw_data.iloc[idx]
            f.write(f"Knowledge: {row['knowledge'][:200]}...\n")
            f.write(f"Question: {row['question']}\n")
            f.write(f"Answer: {row['hallucinated_answer']}\n")
            f.write("-" * 50 + "\n")
            
if __name__ == "__main__":
    run_error_analysis()
