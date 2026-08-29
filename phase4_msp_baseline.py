import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    ).to(device)
    return model, tokenizer, device

def compute_msp_baseline():
    model, tokenizer, device = load_model_and_tokenizer()
    
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:2000]")
    df = pd.DataFrame(dataset)
    
    results = []
    
    print("Extracting MSP for 2000 examples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for ans_type in ['right_answer', 'hallucinated_answer']:
            knowledge = row['knowledge']
            question = row['question']
            answer = row[ans_type]
            
            prompt_text = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer:"
            full_text = f"{prompt_text} {answer}"
            
            prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            full_tokens = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
            
            prompt_len = prompt_tokens.shape[1]
            seq_len = full_tokens.shape[1]
            
            if seq_len > 1024:
                continue
                
            with torch.no_grad():
                outputs = model(full_tokens)
                logits = outputs.logits # (1, seq_len, vocab_size)
                
            # We want the probabilities of the answer tokens.
            # logits[:, :-1, :] predict the next token. 
            # We need the logits that predict the answer tokens: from prompt_len-1 to seq_len-2
            
            shift_logits = logits[0, prompt_len-1:-1, :]
            shift_labels = full_tokens[0, prompt_len:]
            
            if shift_logits.shape[0] == 0:
                continue
                
            probs = torch.softmax(shift_logits, dim=-1)
            
            # 1. Maximum Softmax Probability of the chosen tokens
            token_probs = probs[torch.arange(shift_labels.shape[0]), shift_labels]
            
            mean_prob = token_probs.mean().item()
            min_prob = token_probs.min().item()
            
            # Standard MSP metric often uses the mean token probability or the min
            results.append({
                "example_id": idx,
                "label": "grounded" if ans_type == 'right_answer' else "hallucinated",
                "mean_prob": mean_prob,
                "min_prob": min_prob
            })
            
    res_df = pd.DataFrame(results)
    res_df.to_csv("phase4_msp_features.csv", index=False)
    print("Saved MSP features to phase4_msp_features.csv")
    
    return res_df

def evaluate_msp(df):
    # 0 is grounded, 1 is hallucinated
    df['target'] = (df['label'] == 'hallucinated').astype(int)
    
    # We will use mean_prob as a feature. For logistic regression, it will find the optimal threshold.
    # Actually, lower probability -> higher chance of hallucination.
    # So we can just use -mean_prob or let the model figure it out.
    
    X = df[['mean_prob', 'min_prob']]
    y = df['target']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'lr_roc_auc': [], 'lr_pr_auc': [], 'lr_f1': []}
               
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict_proba(X_test)[:, 1]
        lr_preds_bin = lr.predict(X_test)
        
        metrics['lr_roc_auc'].append(roc_auc_score(y_test, lr_preds))
        metrics['lr_pr_auc'].append(average_precision_score(y_test, lr_preds))
        metrics['lr_f1'].append(f1_score(y_test, lr_preds_bin))
        
    print(f"\n--- MSP Baseline ---")
    print(f"LR  - ROC-AUC: {np.mean(metrics['lr_roc_auc']):.4f} \u00B1 {np.std(metrics['lr_roc_auc']):.4f} | "
          f"PR-AUC: {np.mean(metrics['lr_pr_auc']):.4f} \u00B1 {np.std(metrics['lr_pr_auc']):.4f} | "
          f"F1: {np.mean(metrics['lr_f1']):.4f} \u00B1 {np.std(metrics['lr_f1']):.4f}")
    
    with open("phase4_msp_results.txt", "w") as f:
        f.write(f"MSP ROC-AUC: {np.mean(metrics['lr_roc_auc']):.4f}\n")
        f.write(f"MSP F1: {np.mean(metrics['lr_f1']):.4f}\n")

if __name__ == "__main__":
    if not os.path.exists("phase4_msp_features.csv"):
        df = compute_msp_baseline()
    else:
        df = pd.read_csv("phase4_msp_features.csv")
    evaluate_msp(df)
