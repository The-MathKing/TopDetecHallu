import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import ripser
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def load_smollm():
    model_name = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    print(f"Loading {model_name}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    ).to(device)
    return model, tokenizer, device

def extract_features(diagrams):
    features = {}
    
    # H0 features
    h0 = diagrams[0]
    h0_finite = h0[h0[:, 1] != np.inf] if len(h0) > 0 else np.array([])
    if len(h0_finite) > 0:
        h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
        features['h0_max_lifetime'] = np.max(h0_lifetimes)
        features['h0_total_persistence'] = np.sum(h0_lifetimes)
    else:
        features['h0_max_lifetime'] = 0.0
        features['h0_total_persistence'] = 0.0
        
    # H1 features
    h1 = diagrams[1]
    h1_finite = h1[h1[:, 1] != np.inf] if len(h1) > 0 else np.array([])
    if len(h1_finite) > 0:
        h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
        max_idx = np.argmax(h1_lifetimes)
        
        features['h1_max_lifetime'] = np.max(h1_lifetimes)
        features['h1_total_persistence'] = np.sum(h1_lifetimes)
        features['h1_max_birth'] = h1_finite[max_idx, 0]
        features['h1_max_death'] = h1_finite[max_idx, 1]
    else:
        features['h1_max_lifetime'] = 0.0
        features['h1_total_persistence'] = 0.0
        features['h1_max_birth'] = 0.0
        features['h1_max_death'] = 0.0
        
    return features

def run_extraction():
    out_csv = "phase7_results/truthfulqa_smollm_features.csv"
    if os.path.exists(out_csv):
        print("Features already extracted.")
        return pd.read_csv(out_csv)
        
    os.makedirs("phase7_results", exist_ok=True)
    model, tokenizer, device = load_smollm()
    
    print("Loading TruthfulQA...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = pd.DataFrame(dataset)
    
    df = df.head(500)
    
    results = []
    
    print("Extracting features on TruthfulQA with SmolLM...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        best_ans = row['best_answer']
        bad_ans = row['incorrect_answers'][0] if len(row['incorrect_answers']) > 0 else "I don't know."
        
        for ans_type, ans_text in [('grounded', best_ans), ('hallucinated', bad_ans)]:
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{ans_text}"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = inputs.input_ids.shape[1]
            
            if seq_len > 1024:
                continue
                
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                
            attentions = outputs.attentions
            num_layers = len(attentions)
            
            layer_features = {}
            for layer_idx in range(num_layers):
                attn = attentions[layer_idx].float().cpu().numpy()[0]
                attn = np.nan_to_num(attn, nan=0.0)
                
                avg_attn = np.mean(attn, axis=0)
                W = np.maximum(avg_attn, avg_attn.T)
                D = 1.0 - W
                np.fill_diagonal(D, 0.0)
                
                res = ripser.ripser(D, distance_matrix=True, maxdim=1)
                features = extract_features(res['dgms'])
                
                for k, v in features.items():
                    layer_features[f"layer_{layer_idx}_{k}"] = v
                    
            logits = outputs.logits[0, :-1, :]
            probs = torch.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            msp_score = max_probs.mean().item()
            
            row_data = {
                "example_id": idx,
                "label": ans_type,
                "seq_len": seq_len,
                "msp_score": msp_score
            }
            row_data.update(layer_features)
            results.append(row_data)
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)
    return results_df

def run_evaluation(df):
    print("\n--- Evaluating 10-fold CV on TruthfulQA + SmolLM ---")
    df['target'] = (df['label'] == 'hallucinated').astype(int)
    y = df['target']
    
    X_0d = df[[c for c in df.columns if 'h0' in c]]
    X_1d = df[[c for c in df.columns if 'h1' in c]]
    X_comb = df[[c for c in df.columns if 'h0' in c or 'h1' in c]]
    msp = df['msp_score'].values
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    auc_msp = []
    auc_0d_lr, auc_0d_xgb = [], []
    auc_1d_lr, auc_1d_xgb = [], []
    auc_comb_lr, auc_comb_xgb = [], []
    
    for train_idx, test_idx in skf.split(X_0d, y):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        lr = LogisticRegression()
        lr.fit(msp[train_idx].reshape(-1, 1), y_train)
        preds = lr.predict_proba(msp[test_idx].reshape(-1, 1))[:, 1]
        auc_msp.append(roc_auc_score(y_test, preds))
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_0d.iloc[train_idx], y_train)
        auc_0d_lr.append(roc_auc_score(y_test, lr.predict_proba(X_0d.iloc[test_idx])[:, 1]))
        
        xgb = XGBClassifier(eval_metric='logloss')
        xgb.fit(X_0d.iloc[train_idx], y_train)
        auc_0d_xgb.append(roc_auc_score(y_test, xgb.predict_proba(X_0d.iloc[test_idx])[:, 1]))
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_1d.iloc[train_idx], y_train)
        auc_1d_lr.append(roc_auc_score(y_test, lr.predict_proba(X_1d.iloc[test_idx])[:, 1]))
        
        xgb = XGBClassifier(eval_metric='logloss')
        xgb.fit(X_1d.iloc[train_idx], y_train)
        auc_1d_xgb.append(roc_auc_score(y_test, xgb.predict_proba(X_1d.iloc[test_idx])[:, 1]))
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_comb.iloc[train_idx], y_train)
        auc_comb_lr.append(roc_auc_score(y_test, lr.predict_proba(X_comb.iloc[test_idx])[:, 1]))
        
        xgb = XGBClassifier(eval_metric='logloss')
        xgb.fit(X_comb.iloc[train_idx], y_train)
        auc_comb_xgb.append(roc_auc_score(y_test, xgb.predict_proba(X_comb.iloc[test_idx])[:, 1]))
        
    print(f"MSP Baseline (LR): {np.mean(auc_msp):.3f} +/- {np.std(auc_msp):.3f}")
    print(f"0D Only (LR):      {np.mean(auc_0d_lr):.3f} +/- {np.std(auc_0d_lr):.3f}")
    print(f"0D Only (XGB):     {np.mean(auc_0d_xgb):.3f} +/- {np.std(auc_0d_xgb):.3f}")
    print(f"1D Only (LR):      {np.mean(auc_1d_lr):.3f} +/- {np.std(auc_1d_lr):.3f}")
    print(f"1D Only (XGB):     {np.mean(auc_1d_xgb):.3f} +/- {np.std(auc_1d_xgb):.3f}")
    print(f"Combined (LR):     {np.mean(auc_comb_lr):.3f} +/- {np.std(auc_comb_lr):.3f}")
    print(f"Combined (XGB):    {np.mean(auc_comb_xgb):.3f} +/- {np.std(auc_comb_xgb):.3f}")

if __name__ == "__main__":
    df = run_extraction()
    run_evaluation(df)
