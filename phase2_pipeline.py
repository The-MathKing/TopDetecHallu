import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import ripser
import persim
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct"):
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
    """
    Extracts topological descriptors from Betti-0 and Betti-1 diagrams.
    Returns a dict of features.
    """
    features = {}
    
    # H0 features
    h0 = diagrams[0]
    # Filter out infinite death
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

def run_pipeline():
    model, tokenizer, device = load_model_and_tokenizer()
    
    print("Loading HaluEval subset...")
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:30]")
    df = pd.DataFrame(dataset)
    
    out_dir = "phase2_results"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    print("Running pipeline over 30 examples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # We will run both the grounded and hallucinated answers
        # to compare the topological features.
        
        for ans_type in ['right_answer', 'hallucinated_answer']:
            prompt = f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}\nAnswer: {row[ans_type]}"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = inputs.input_ids.shape[1]
            
            # To avoid OOM and keep it fast, we skip very long sequences in this test
            if seq_len > 1024:
                print(f"Skipping example {idx} ({ans_type}) due to length {seq_len}")
                continue
                
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                
            attentions = outputs.attentions
            
            # We will just analyze the last layer for plotting, but we can extract features for all
            num_layers = len(attentions)
            
            layer_features = {}
            last_layer_diagrams = None
            
            for layer_idx in range(num_layers):
                # Shape: (batch, num_heads, seq_len, seq_len)
                attn = attentions[layer_idx].float().cpu().numpy()[0]
                
                # Average across heads
                avg_attn = np.mean(attn, axis=0)
                
                # Symmetrize
                W = np.maximum(avg_attn, avg_attn.T)
                
                # Distance matrix
                D = 1.0 - W
                np.fill_diagonal(D, 0.0)
                
                # Ripser
                res = ripser.ripser(D, distance_matrix=True, maxdim=1)
                diagrams = res['dgms']
                
                features = extract_features(diagrams)
                
                for k, v in features.items():
                    layer_features[f"layer_{layer_idx}_{k}"] = v
                    
                if layer_idx == num_layers - 1:
                    last_layer_diagrams = diagrams
                    
            # Save plot for the first few examples to eyeball
            if idx < 5:
                plt.figure()
                persim.plot_diagrams(last_layer_diagrams, show=False)
                plt.title(f"Example {idx} - {ans_type} (Last Layer)")
                plt.savefig(f"{out_dir}/diagram_ex{idx}_{ans_type}.png")
                plt.close()
                
            # Add metadata
            row_data = {
                "example_id": idx,
                "label": "grounded" if ans_type == 'right_answer' else "hallucinated",
                "seq_len": seq_len
            }
            row_data.update(layer_features)
            results.append(row_data)
            
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_dir}/phase2_features.csv", index=False)
    print(f"Pipeline complete. Extracted features saved to {out_dir}/phase2_features.csv")
    print(f"Check {out_dir} for persistence diagram plots.")

if __name__ == "__main__":
    run_pipeline()
