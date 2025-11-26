"""
Evaluate GRPO-trained model and compare with baseline.
"""

import os
import pickle
import torch
import numpy as np
from contextlib import nullcontext
import matplotlib.pyplot as plt
from verifier import verify_batch

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
base_out_dir = 'out'
grpo_out_dir = 'out-grpo'
num_eval_samples = 50
num_samples_per_prompt = 4
max_new_tokens = 50
temperature = 0.8
top_k = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_model(ckpt_path, device):
    """Load a model from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = checkpoint['model_args']
    
    from model import GPTConfig, GPT
    config = GPTConfig(**gptconf)
    model = GPT(config)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model

def evaluate_model(model, prompts, encode, decode, device, ctx):
    """Evaluate model on prompts."""
    all_samples = []
    all_rewards = []
    
    print(f"Evaluating on {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(prompts)}...")
        
        prompt_samples = []
        
        for _ in range(num_samples_per_prompt):
            start_ids = encode(prompt)
            x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
            
            with torch.no_grad():
                with ctx:
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            
            completion = decode(y[0].tolist())
            generated = completion[len(prompt):]
            prompt_samples.append(generated)
        
        prompt_rewards = verify_batch(prompt_samples)
        all_samples.append(prompt_samples)
        all_rewards.extend(prompt_rewards)
    
    return all_samples, all_rewards

# -----------------------------------------------------------------------------
# Load models and data
# -----------------------------------------------------------------------------
print("="*60)
print("GRPO EVALUATION")
print("="*60)

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load encoder/decoder
meta_path = os.path.join(base_out_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)

# Load baseline results
with open('baseline_results.pkl', 'rb') as f:
    baseline = pickle.load(f)

# Use subset of prompts for evaluation
eval_prompts = baseline['prompts'][:num_eval_samples]

print(f"\nLoading base model from {base_out_dir}...")
base_model = load_model(os.path.join(base_out_dir, 'ckpt.pt'), device)

print(f"Loading GRPO model from {grpo_out_dir}...")
grpo_model = load_model(os.path.join(grpo_out_dir, 'ckpt_final.pt'), device)

# -----------------------------------------------------------------------------
# Evaluate both models
# -----------------------------------------------------------------------------
print("\n" + "-"*60)
print("Evaluating BASE model...")
print("-"*60)
base_samples, base_rewards = evaluate_model(base_model, eval_prompts, encode, decode, device, ctx)

print("\n" + "-"*60)
print("Evaluating GRPO model...")
print("-"*60)
grpo_samples, grpo_rewards = evaluate_model(grpo_model, eval_prompts, encode, decode, device, ctx)

# -----------------------------------------------------------------------------
# Quantitative comparison
# -----------------------------------------------------------------------------
base_mean = np.mean(base_rewards)
base_std = np.std(base_rewards)
grpo_mean = np.mean(grpo_rewards)
grpo_std = np.std(grpo_rewards)

print("\n" + "="*60)
print("QUANTITATIVE RESULTS")
print("="*60)
print(f"\nBASE MODEL:")
print(f"  Mean reward: {base_mean:.3f} ± {base_std:.3f}")
print(f"  Min/Max: {min(base_rewards):.3f} / {max(base_rewards):.3f}")

print(f"\nGRPO MODEL:")
print(f"  Mean reward: {grpo_mean:.3f} ± {grpo_std:.3f}")
print(f"  Min/Max: {min(grpo_rewards):.3f} / {max(grpo_rewards):.3f}")

print(f"\nIMPROVEMENT:")
print(f"  Absolute: +{grpo_mean - base_mean:.3f}")
print(f"  Relative: +{100 * (grpo_mean - base_mean) / base_mean:.1f}%")

# -----------------------------------------------------------------------------
# Qualitative comparison - show examples
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("QUALITATIVE COMPARISON - SAMPLE OUTPUTS")
print("="*60)

for i in range(min(5, len(eval_prompts))):
    print(f"\n{'='*60}")
    print(f"PROMPT {i+1}: '{eval_prompts[i]}'")
    print(f"{'='*60}")
    
    print("\nBASE MODEL OUTPUTS:")
    for j in range(num_samples_per_prompt):
        reward = verify_batch([base_samples[i][j]])[0]
        sample_preview = base_samples[i][j][:100].replace('\n', '\\n')
        print(f"  [{reward:.1f}] {sample_preview}")
    
    print("\nGRPO MODEL OUTPUTS:")
    for j in range(num_samples_per_prompt):
        reward = verify_batch([grpo_samples[i][j]])[0]
        sample_preview = grpo_samples[i][j][:100].replace('\n', '\\n')
        print(f"  [{reward:.1f}] {sample_preview}")

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram comparison
axes[0].hist(base_rewards, bins=20, alpha=0.5, label='Base', color='blue')
axes[0].hist(grpo_rewards, bins=20, alpha=0.5, label='GRPO', color='green')
axes[0].axvline(base_mean, color='blue', linestyle='--', linewidth=2, label=f'Base mean: {base_mean:.2f}')
axes[0].axvline(grpo_mean, color='green', linestyle='--', linewidth=2, label=f'GRPO mean: {grpo_mean:.2f}')
axes[0].set_xlabel('Verifier Reward')
axes[0].set_ylabel('Count')
axes[0].set_title('Reward Distribution Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Load and plot training curve
with open(os.path.join(grpo_out_dir, 'training_history.pkl'), 'rb') as f:
    history = pickle.load(f)

axes[1].plot(history['reward_history'], linewidth=2, color='green')
axes[1].axhline(y=base_mean, color='blue', linestyle='--', linewidth=2, label=f'Base: {base_mean:.2f}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Verifier Reward')
axes[1].set_title('GRPO Training Progress')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(grpo_out_dir, 'evaluation_comparison.png'), dpi=300)
print(f"\n✓ Comparison plots saved to '{grpo_out_dir}/evaluation_comparison.png'")

# -----------------------------------------------------------------------------
# Pattern analysis
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("PATTERN ANALYSIS")
print("="*60)

# Analyze character frequencies
def analyze_char_frequency(samples):
    all_text = ''.join([''.join(s) for s in samples])
    e_count = all_text.lower().count('e')
    total_chars = len(all_text)
    return e_count / total_chars if total_chars > 0 else 0

base_e_freq = analyze_char_frequency(base_samples)
grpo_e_freq = analyze_char_frequency(grpo_samples)

print(f"\nCharacter 'e' frequency:")
print(f"  Base model:  {base_e_freq:.3f}")
print(f"  GRPO model:  {grpo_e_freq:.3f}")
print(f"  Change:      +{grpo_e_freq - base_e_freq:.3f}")

# Analyze length statistics
base_lengths = [len(s) for samples in base_samples for s in samples]
grpo_lengths = [len(s) for samples in grpo_samples for s in samples]

print(f"\nAverage completion length:")
print(f"  Base model:  {np.mean(base_lengths):.1f} chars")
print(f"  GRPO model:  {np.mean(grpo_lengths):.1f} chars")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)