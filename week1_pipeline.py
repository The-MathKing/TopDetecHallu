import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import ripser
import networkx as nx
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd

def run_synthetic_sanity_check():
    """
    Synthetic sanity check for persistent homology computation.
    Creates a 4-cycle graph with a pendant edge (a tail).
    Expected Betti-0 is 1 (one connected component).
    Expected Betti-1 is 1 (one cycle).
    """
    print("--- Running Synthetic Sanity Check ---")
    # Adjacency matrix for 5 nodes
    # Nodes 0, 1, 2, 3 form a cycle. Node 4 is attached to node 3.
    # Edges: (0,1), (1,2), (2,3), (3,0), (3,4)
    A = np.zeros((5, 5))
    edges = [(0,1), (1,2), (2,3), (3,0), (3,4)]
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    
    # Distance matrix: D_ij = 1 - A_ij for i != j. D_ii = 0.
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    
    print("Adjacency Matrix:")
    print(A)
    
    # Run Ripser (distance_matrix=True specifies that the input is a distance matrix)
    result = ripser.ripser(D, distance_matrix=True, maxdim=1)
    diagrams = result['dgms']
    
    # diagrams[0] is H0 (Betti-0), diagrams[1] is H1 (Betti-1)
    # Filter out features with infinite death time for Betti-0 (the global connected component)
    # Actually, Ripser returns one feature in H0 that dies at infinity.
    # The number of connected components at threshold 0 is the number of features in H0 that are born at 0.
    # For H1, we expect 1 feature born at 0 (or some small distance) and dying at distance > 0.
    
    h0_features = len(diagrams[0])
    h1_features = len(diagrams[1])
    
    print(f"H0 diagram (Betti-0 features): {diagrams[0]}")
    print(f"H1 diagram (Betti-1 features): {diagrams[1]}")
    
    # For a graph at threshold < 1, the 4-cycle appears at distance 0.
    # It gets filled in at distance 1.
    # So we expect an H1 feature born at 0, dying at 1.
    has_1_cycle = any((b <= 0) and (d >= 1) for b, d in diagrams[1] if d != float('inf'))
    
    assert has_1_cycle, "Sanity check failed: Expected Betti-1=1 for a 4-cycle graph."
    print("SUCCESS: Synthetic sanity check passed! The 4-cycle was correctly detected.\n")


def load_model_and_tokenizer(model_name="Qwen/Qwen2.5-3B-Instruct"):
    """
    Loads the model on MPS (if available) with eager attention for weight extraction.
    """
    print(f"--- Loading Model: {model_name} ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # IMPORTANT: attn_implementation="eager" is mandatory to get output_attentions=True to work
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    ).to(device)
    
    return model, tokenizer, device


def extract_attention_and_compute_homology(model, tokenizer, device, text):
    """
    Runs a single forward pass, extracts attention, symmetrizes it, and computes homology.
    """
    print("\n--- Extracting Attention ---")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Teacher-forced single pass
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions
    assert attentions is not None, "Failed to extract attentions! Check attn_implementation='eager'."
    print(f"Successfully extracted attentions: {len(attentions)} layers.")
    
    # Let's just look at the last layer for this example
    last_layer_attn = attentions[-1].float().cpu().numpy()[0] # Shape: (num_heads, seq_len, seq_len)
    print(f"Last layer attention shape: {last_layer_attn.shape}")
    
    # Average across heads
    avg_attn = np.mean(last_layer_attn, axis=0) # Shape: (seq_len, seq_len)
    
    # Symmetrize: W = max(A, A^T)
    W = np.maximum(avg_attn, avg_attn.T)
    
    # Convert to distance matrix (larger weight -> smaller distance)
    # We add a small epsilon to avoid exactly 0 distance between distinct tokens, except on diagonal
    D = 1.0 - W
    np.fill_diagonal(D, 0.0)
    
    print("Computing persistent homology on attention graph...")
    # Compute persistent homology
    result = ripser.ripser(D, distance_matrix=True, maxdim=1)
    diagrams = result['dgms']
    
    num_h0 = len(diagrams[0])
    num_h1 = len(diagrams[1])
    
    print(f"Found {num_h0} H0 features and {num_h1} H1 features in the last layer.")
    
    # Just printing the most persistent H1 feature if it exists
    if num_h1 > 0:
        lifetimes = [d - b for b, d in diagrams[1] if d != float('inf')]
        if lifetimes:
            max_lifetime = max(lifetimes)
            print(f"Maximum H1 persistence (lifetime): {max_lifetime:.4f}")
    
    return diagrams


def load_halueval_subset():
    """
    Downloads and inspects a subset of HaluEval.
    """
    print("\n--- Loading HaluEval Dataset ---")
    # For demonstration, let's load a subset from huggingface datasets. 
    # HaluEval is available on HF as 'pminervini/HaluEval' or we can load a specific JSON/CSV.
    # Note: Using a general placeholder name here, adjust if you are using a specific file.
    try:
        # Load a small slice to verify it works
        dataset = load_dataset("pminervini/HaluEval", "qa", split="data[:50]")
        df = pd.DataFrame(dataset)
        print(f"Successfully loaded HaluEval subset: {len(df)} examples.")
        print("Columns:", df.columns.tolist())
        
        # Display the first example
        print("\nFirst example snippet:")
        row = df.iloc[0]
        for col in ['knowledge', 'question', 'right_answer', 'hallucinated_answer']:
            if col in df.columns:
                print(f"  {col}: {str(row[col])[:100]}...")
                
    except Exception as e:
        print(f"Could not load HaluEval from HF Hub automatically. Error: {e}")
        print("You may need to download the JSON files manually from the HaluEval GitHub repo and load them using pandas.")

if __name__ == "__main__":
    # 1. Synthetic sanity check (Topological graph math)
    run_synthetic_sanity_check()
    
    # 2. HaluEval Dataset test
    load_halueval_subset()
    
    # 3. Model setup and Attention extraction
    try:
        model, tokenizer, device = load_model_and_tokenizer()
        test_text = "The capital of France is Paris. The capital of Germany is Berlin."
        extract_attention_and_compute_homology(model, tokenizer, device, test_text)
    except Exception as e:
        print(f"\nModel loading/extraction failed: {e}")
        print("Note: If the model download fails, ensure you have internet access and HF token if needed.")
