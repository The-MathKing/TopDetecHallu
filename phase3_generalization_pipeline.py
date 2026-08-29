import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import ripser
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
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
    
    h0 = diagrams[0]
    h0_finite = h0[h0[:, 1] != np.inf] if len(h0) > 0 else np.array([])
    if len(h0_finite) > 0:
        h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
        features['h0_max_lifetime'] = np.max(h0_lifetimes)
        features['h0_total_persistence'] = np.sum(h0_lifetimes)
    else:
        features['h0_max_lifetime'] = 0.0
        features['h0_total_persistence'] = 0.0
        
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

def run_phase3():
    model, tokenizer, device = load_model_and_tokenizer()
    
    # We will run this on a smaller subset for generalization (500 examples) to speed it up
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:500]")
    df = pd.DataFrame(dataset)
    
    out_dir = "phase3_results"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = f"{out_dir}/train_features_qwen1_5b.csv"
    
    with open(out_csv, 'w') as f:
        header_written = False
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            for ans_type in ['right_answer', 'hallucinated_answer']:
                prompt = f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}\nAnswer: {row[ans_type]}"
                
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
                    avg_attn = np.mean(attn, axis=0)
                    avg_attn = np.nan_to_num(avg_attn, nan=0.0)
                    W = np.maximum(avg_attn, avg_attn.T)
                    D = 1.0 - W
                    np.fill_diagonal(D, 0.0)
                    
                    res = ripser.ripser(D, distance_matrix=True, maxdim=1)
                    features = extract_features(res['dgms'])
                    
                    for k, v in features.items():
                        layer_features[f"layer_{layer_idx}_{k}"] = v
                        
                row_data = {
                    "example_id": idx,
                    "label": "grounded" if ans_type == 'right_answer' else "hallucinated",
                    "seq_len": seq_len
                }
                row_data.update(layer_features)
                
                res_df = pd.DataFrame([row_data])
                if not header_written:
                    res_df.to_csv(f, index=False, header=True)
                    header_written = True
                else:
                    res_df.to_csv(f, index=False, header=False)

if __name__ == "__main__":
    run_phase3()
