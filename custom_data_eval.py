import argparse
from collections import Counter
import numpy as np
from scipy.stats import entropy
import os

# ---------- Helper functions ----------

def ngram_distribution(text, n=3):
    """Return normalized n-gram frequency distribution."""
    counts = Counter(text[i:i+n] for i in range(len(text)-n+1))
    total = sum(counts.values())
    return counts, np.array([v/total for v in counts.values()]), list(counts.keys())

def align_probs(counts, all_keys):
    """Aligns probability vectors across same n-gram vocabulary."""
    total = sum(counts.values())
    return np.array([counts.get(k, 0) / total for k in all_keys])

def compute_kl(train_path, gen_path, n=3):
    """Compute KL Divergence between train text and generated samples."""
    if not os.path.exists(train_path) or not os.path.exists(gen_path):
        raise FileNotFoundError("Check that both files exist.")

    with open(train_path, encoding="utf-8") as f:
        train_text = f.read()
    with open(gen_path, encoding="utf-8") as f:
        gen_text = f.read()

    # Compute n-gram counts
    train_counts, _, train_keys = ngram_distribution(train_text, n=n)
    gen_counts, _, gen_keys = ngram_distribution(gen_text, n=n)

    # Align both vocabularies
    all_keys = sorted(set(train_keys) | set(gen_keys))
    P = align_probs(train_counts, all_keys)
    Q = align_probs(gen_counts, all_keys)

    # Numerical stability
    eps = 1e-12
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)

    # Compute KL Divergence
    D_kl = entropy(P, Q, base=np.e)
    return D_kl


# ---------- General Metrics ----------

def distinct_n(text, n=1):
    """Compute distinct-n (type-token ratio)."""
    if len(text) < n:
        return 0.0
    tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    return len(set(tokens)) / len(tokens)

def compute_general_metrics(gen_path):
    """Compute distinct-1 and distinct-2 diversity metrics."""
    with open(gen_path, encoding="utf-8") as f:
        gen_text = f.read()
    d1 = distinct_n(gen_text, n=1)
    d2 = distinct_n(gen_text, n=2)
    return d1, d2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute KL Divergence and general text diversity metrics.")
    parser.add_argument("--train", required=True, help="Path to training text file (e.g., data/shakespeare_char/input.txt)")
    parser.add_argument("--gen", required=True, help="Path to generated samples file (e.g., samples.txt)")
    parser.add_argument("--n", type=int, default=3, help="N-gram size for KL divergence (default: 3)")
    args = parser.parse_args()

    kl_value = compute_kl(args.train, args.gen, n=args.n)
    d1, d2 = compute_general_metrics(args.gen)

    print("\n===== Evaluation Metrics =====")
    print(f"KL Divergence ({args.n}-gram): {kl_value:.6f}")
    print(f"Distinct-1: {d1:.4f}")
    print(f"Distinct-2: {d2:.4f}")
    print("==============================\n")
