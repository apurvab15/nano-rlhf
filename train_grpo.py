"""
GRPO (Group Relative Policy Optimization) training for RLVR.
"""

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm
import matplotlib.pyplot as plt
from verifier import verify_batch

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
# Model loading
init_from = 'resume'
out_dir = 'out-shakespeare-char'
grpo_out_dir = 'out-grpo'

# Training
num_epochs = 15
batch_size = 16  # prompts per batch
num_samples_per_prompt = 4  # K samples per prompt for GRPO
max_new_tokens = 50
learning_rate = 5e-6
weight_decay = 0.01
beta = 0.1  # KL penalty coefficient
grad_clip = 1.0

# Generation
temperature = 0.8
top_k = 200

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

os.makedirs(grpo_out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Load model and prompts
# -----------------------------------------------------------------------------
print("Loading model...")
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = checkpoint['model_args']

# Create model
from model import GPTConfig, GPT
config = GPTConfig(**gptconf)
model = GPT(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.train()
model.to(device)

# Save reference model for KL penalty
ref_model = GPT(config)
ref_model.load_state_dict(model.state_dict())
ref_model.eval()
ref_model.to(device)

# Load encoder/decoder
# Try multiple locations for meta.pkl (similar to sample.py)
load_meta = False
meta_path = None

# First try: out_dir/meta.pkl
meta_path = os.path.join(out_dir, 'meta.pkl')
if os.path.exists(meta_path):
    load_meta = True
else:
    # Second try: data directory from checkpoint config
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            load_meta = True
    else:
        # Third try: common data directories
        for dataset_name in ['shakespeare_char', 'shakespeare', 'taylor_swift_lyrics']:
            meta_path = os.path.join('data', dataset_name, 'meta.pkl')
            if os.path.exists(meta_path):
                load_meta = True
                break

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta.get('vocab_size', len(itos))
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f"Loaded tokenizer with vocab_size={vocab_size}")
else:
    # Fallback: Use GPT-2 tokenizer only if model vocab_size matches
    model_vocab_size = config.vocab_size
    if model_vocab_size == 50304 or model_vocab_size == 50257:
        print("No meta.pkl found, using GPT-2 tokenizer (model vocab_size matches GPT-2)")
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    else:
        raise FileNotFoundError(
            f"Could not find meta.pkl for tokenizer. Model vocab_size={model_vocab_size} "
            f"does not match GPT-2. Please ensure meta.pkl exists in:\n"
            f"  - {os.path.join(out_dir, 'meta.pkl')}\n"
            f"  - data/<dataset_name>/meta.pkl"
        )

# Load prompts
print("Loading prompts...")
with open('baseline_results.pkl', 'rb') as f:
    baseline = pickle.load(f)
prompts = baseline['prompts']
print(f"Loaded {len(prompts)} prompts")

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# -----------------------------------------------------------------------------
# GRPO functions
# -----------------------------------------------------------------------------

def generate_samples(model, prompt_ids, num_samples, max_tokens):
    """Generate multiple samples for a prompt and compute log probs."""
    samples = []
    log_probs_list = []
    prompt_len = prompt_ids.shape[1]
    block_size = model.config.block_size
    
    for _ in range(num_samples):
        # Generate sequence token by token while tracking log probs (WITH gradients)
        generated = prompt_ids.clone()  # Start with prompt
        gen_log_probs = []
        
        for _ in range(max_tokens):
            # Get context (crop if too long)
            idx_cond = generated if generated.size(1) <= block_size else generated[:, -block_size:]
            
            # Forward pass - KEEP GRADIENTS for log probs
            logits, _ = model(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature  # [batch, vocab_size]
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            log_probs_all = F.log_softmax(logits, dim=-1)
            
            # Sample next token (detach for sampling, but keep original for gradients)
            with torch.no_grad():
                next_token = torch.multinomial(probs.detach(), num_samples=1)
            
            # Record log probability of chosen action (KEEP GRADIENTS)
            # Use gather to get the log prob of the sampled token
            log_prob = torch.gather(log_probs_all, 1, next_token).squeeze(-1)  # [batch]
            gen_log_probs.append(log_prob)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        samples.append(generated)
        # Stack log probs: [batch, max_new_tokens]
        log_probs_list.append(torch.stack(gen_log_probs, dim=1))
    
    return samples, log_probs_list


def compute_grpo_loss(model, ref_model, samples, rewards, old_log_probs):
    """
    Compute GRPO loss.
    
    GRPO uses group-relative advantages:
    A_i = R_i - mean(R_group)
    
    Loss = -mean(A_i * log π_θ(y_i|x)) + β * KL(π_θ || π_ref)
    
    Note: old_log_probs contains log probs for generated tokens only (not prompt)
    """
    total_loss = 0.0
    num_samples = len(samples)
    block_size = model.config.block_size
    
    # Normalize rewards within group
    rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
    mean_reward = rewards_tensor.mean()
    advantages = rewards_tensor - mean_reward
    
    for i, (sample, advantage, old_lp) in enumerate(zip(samples, advantages, old_log_probs)):
        # Extract prompt length from sample
        # old_lp shape: [batch, max_new_tokens] - log probs for generated tokens
        prompt_len = sample.shape[1] - old_lp.shape[1]
        
        # Get the sequence for computing log probs (crop if too long)
        if sample.shape[1] > block_size:
            seq_for_compute = sample[:, -block_size:]
            # Adjust prompt_len for cropped sequence
            effective_prompt_len = max(0, prompt_len - (sample.shape[1] - block_size))
        else:
            seq_for_compute = sample
            effective_prompt_len = prompt_len
        
        # Forward pass with targets to get logits for all positions
        input_tokens = seq_for_compute[:, :-1]
        target_tokens = seq_for_compute[:, 1:]
        
        # Ensure we don't exceed block_size (shouldn't happen after cropping, but be safe)
        if input_tokens.shape[1] > block_size:
            input_tokens = input_tokens[:, -block_size:]
            target_tokens = target_tokens[:, -block_size:]
            effective_prompt_len = max(0, effective_prompt_len - (input_tokens.shape[1] - block_size))
        
        # Get current model log probs (with gradients) - use targets to get all logits
        logits, _ = model(input_tokens, targets=target_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, target_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Get only generated tokens (exclude prompt)
        if effective_prompt_len > 0:
            current_token_log_probs = token_log_probs[:, effective_prompt_len-1:]
        else:
            current_token_log_probs = token_log_probs
        
        # Policy loss: -advantage * sum of log probs (we want to maximize advantage-weighted log prob)
        policy_loss = -(advantage * current_token_log_probs.sum())
        
        # KL penalty with reference model (no gradients)
        with torch.no_grad():
            ref_logits, _ = ref_model(input_tokens, targets=target_tokens)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = torch.gather(ref_log_probs, 2, target_tokens.unsqueeze(-1)).squeeze(-1)
            
            if effective_prompt_len > 0:
                ref_token_log_probs_gen = ref_token_log_probs[:, effective_prompt_len-1:]
            else:
                ref_token_log_probs_gen = ref_token_log_probs
        
        # KL divergence: sum of (log p_current - log p_ref) for generated tokens
        kl_div = (current_token_log_probs - ref_token_log_probs_gen).sum()
        
        total_loss += policy_loss + beta * kl_div
    
    return total_loss / num_samples, mean_reward.item()


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
print("\nStarting GRPO training...")
print(f"Epochs: {num_epochs}")
print(f"Batch size: {batch_size}")
print(f"Samples per prompt: {num_samples_per_prompt}")
print(f"Learning rate: {learning_rate}")
print(f"Beta (KL): {beta}")
print("-" * 60)

reward_history = []
loss_history = []

for epoch in range(num_epochs):
    epoch_rewards = []
    epoch_losses = []
    
    # Shuffle prompts
    indices = torch.randperm(len(prompts))
    
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_indices = indices[batch_start:batch_end]
        
        batch_loss = 0.0
        batch_rewards = []
        
        for idx in batch_indices:
            prompt = prompts[idx]
            # Encode prompt
            start_ids = encode(prompt)
            
            # Validate token IDs are within vocab_size
            vocab_size = config.vocab_size
            max_id = max(start_ids) if start_ids else -1
            if max_id >= vocab_size:
                raise ValueError(
                    f"Token ID {max_id} is out of bounds! Model vocab_size={vocab_size}. "
                    f"This usually means the tokenizer doesn't match the model. "
                    f"Please check that you're using the correct meta.pkl file."
                )
            
            prompt_ids = torch.tensor(
                start_ids, 
                dtype=torch.long, 
                device=device
            )[None, ...]
            
            # Generate samples
            model.eval()
            samples, old_log_probs = generate_samples(
                model, 
                prompt_ids, 
                num_samples_per_prompt, 
                max_new_tokens
            )
            model.train()
            
            # Decode and compute rewards
            completions = [
                decode(s[0, prompt_ids.shape[1]:].tolist()) 
                for s in samples
            ]
            rewards = verify_batch(completions)
            batch_rewards.extend(rewards)
            
            # Compute loss
            loss, mean_reward = compute_grpo_loss(
                model, 
                ref_model, 
                samples, 
                rewards, 
                old_log_probs
            )
            batch_loss += loss
        
        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        epoch_losses.append(batch_loss.item() / len(batch_indices))
        epoch_rewards.extend(batch_rewards)
    
    # Log progress
    mean_epoch_reward = np.mean(epoch_rewards)
    mean_epoch_loss = np.mean(epoch_losses)
    reward_history.append(mean_epoch_reward)
    loss_history.append(mean_epoch_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Mean Reward: {mean_epoch_reward:.3f} | "
          f"Loss: {mean_epoch_loss:.3f}")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': gptconf,
            'epoch': epoch,
            'reward_history': reward_history,
        }
        torch.save(checkpoint, os.path.join(grpo_out_dir, f'ckpt_epoch_{epoch+1}.pt'))

# -----------------------------------------------------------------------------
# Save final model and results
# -----------------------------------------------------------------------------
print("\nSaving final model...")
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': gptconf,
    'epoch': num_epochs,
    'reward_history': reward_history,
    'loss_history': loss_history,
}
torch.save(checkpoint, os.path.join(grpo_out_dir, 'ckpt_final.pt'))

# Save training history
history = {
    'reward_history': reward_history,
    'loss_history': loss_history,
    'hyperparameters': {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'num_samples_per_prompt': num_samples_per_prompt,
        'learning_rate': learning_rate,
        'beta': beta,
        'max_new_tokens': max_new_tokens,
    }
}
with open(os.path.join(grpo_out_dir, 'training_history.pkl'), 'wb') as f:
    pickle.dump(history, f)

# Plot reward curve
plt.figure(figsize=(10, 6))
plt.plot(reward_history, linewidth=2)
plt.axhline(y=baseline['mean_reward'], color='r', linestyle='--', 
            label=f'Baseline ({baseline["mean_reward"]:.2f})')
plt.xlabel('Epoch')
plt.ylabel('Mean Verifier Reward')
plt.title('GRPO Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(grpo_out_dir, 'reward_curve.png'), dpi=300)
print(f"✓ Reward curve saved to '{grpo_out_dir}/reward_curve.png'")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Initial mean reward: {baseline['mean_reward']:.3f}")
print(f"Final mean reward: {reward_history[-1]:.3f}")
print(f"Improvement: {reward_history[-1] - baseline['mean_reward']:.3f}")