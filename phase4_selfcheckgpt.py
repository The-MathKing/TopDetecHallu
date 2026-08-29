import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct"):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    return model, tokenizer, device

def run_selfcheck():
    print("Loading LLM for generation and evaluation...")
    model, tokenizer, device = load_model_and_tokenizer()
    
    # Use SelfCheckLLMPrompt with local Qwen model instead of NLI
    # We pass the Qwen model as the client
    # selfcheckgpt uses `client.chat.completions.create` if using OpenAI API,
    # but we can implement a simple wrapper for local HF models.
    
    class LocalHFClient:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            
        def evaluate_prompt(self, prompt):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            gen_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
            return gen_text
            
    # The actual selfcheckgpt LLM prompt class expects OpenAI API client by default,
    # so we will just implement the prompt logic directly to avoid API mismatches.
    # The prompt checks if a sentence is supported by a passage.
    
    def llm_evaluate(sentence, passage, client):
        prompt = f"Context: {passage}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No:"
        ans = client.evaluate_prompt(prompt)
        # If it answers 'no', it means contradiction -> score = 1.0 (hallucinated)
        if 'no' in ans or 'contradict' in ans:
            return 1.0
        return 0.0
        
    client = LocalHFClient(model, tokenizer, device)
    nlp = spacy.load("en_core_web_sm")
    
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:200]")
    df = pd.DataFrame(dataset)
    
    results = []
    
    print("Running SelfCheckGPT (LLM Prompt) over 200 examples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        knowledge = row['knowledge']
        question = row['question']
        
        prompt_text = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        
        if prompt_len > 1024:
            continue
            
        # Generate 3 stochastic samples
        with torch.no_grad():
            sample_outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                num_return_sequences=3,
                pad_token_id=tokenizer.eos_token_id
            )
            
        sampled_passages = []
        for out in sample_outputs:
            gen_text = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            sampled_passages.append(gen_text.strip())
            
        for ans_type in ['right_answer', 'hallucinated_answer']:
            answer = row[ans_type]
            doc = nlp(answer)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
            
            if not sentences:
                continue
                
            doc_scores = []
            for sentence in sentences:
                sent_scores = []
                for passage in sampled_passages:
                    score = llm_evaluate(sentence, passage, client)
                    sent_scores.append(score)
                doc_scores.append(np.mean(sent_scores))
                
            final_doc_score = np.mean(doc_scores)
            
            results.append({
                "example_id": idx,
                "label": "grounded" if ans_type == 'right_answer' else "hallucinated",
                "selfcheck_score": final_doc_score
            })
                
    res_df = pd.DataFrame(results)
    res_df.to_csv("phase4_selfcheck_features.csv", index=False)
    print("Saved SelfCheckGPT features to phase4_selfcheck_features.csv")
    return res_df

def evaluate_selfcheck(df):
    df['target'] = (df['label'] == 'hallucinated').astype(int)
    
    X = df[['selfcheck_score']]
    y = df['target']
    
    # 5-fold CV to match others
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
        
    print(f"\n--- SelfCheckGPT (LLM Prompt) Baseline ---")
    print(f"LR  - ROC-AUC: {np.mean(metrics['lr_roc_auc']):.4f} \u00B1 {np.std(metrics['lr_roc_auc']):.4f} | "
          f"PR-AUC: {np.mean(metrics['lr_pr_auc']):.4f} \u00B1 {np.std(metrics['lr_pr_auc']):.4f} | "
          f"F1: {np.mean(metrics['lr_f1']):.4f} \u00B1 {np.std(metrics['lr_f1']):.4f}")

if __name__ == "__main__":
    if not os.path.exists("phase4_selfcheck_features.csv"):
        df = run_selfcheck()
    else:
        df = pd.read_csv("phase4_selfcheck_features.csv")
        
    evaluate_selfcheck(df)
