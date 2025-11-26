"""
Policy Gradient (REINFORCE) for NanoGPT
Trains the model to maximize lexical diversity rewards
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, GPTConfig
import pickle
import os
import numpy as np
from tqdm import tqdm

def load_base_model(checkpoint_path='out-shakespeare-char/ckpt.pt', device='cpu'):
    """Load pre-trained NanoGPT model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model, checkpoint

def load_reward_model(checkpoint_path='out-reward/reward_model.pt', device='cpu'):
    """Load trained reward model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Define RewardModel
    class RewardModel(nn.Module):
        def __init__(self, gpt_model):
            super().__init__()
            self.transformer = gpt_model.transformer
            self.config = gpt_model.config
            self.reward_head = nn.Linear(gpt_model.config.n_embd, 1)
        
        def forward(self, idx):
            device = idx.device
            b, t = idx.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            
            for block in self.transformer.h:
                x = block(x)
            
            x = self.transformer.ln_f(x)
            x = x.mean(dim=1)
            reward = self.reward_head(x).squeeze(-1)
            return reward
    
    gpt = GPT(config)
    model = RewardModel(gpt)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, stoi, itos

def generate_with_log_probs(model, prompt_ids, max_new_tokens, temperature=1.0, device='cpu'):
    """
    Generate text and track log probabilities for REINFORCE
    
    Returns:
        generated_ids: Full sequence including prompt
        log_probs: Log probability of each generated token
        entropy: Entropy of each distribution (for exploration bonus)
    """
    # Start with prompt
    generated = prompt_ids.clone()
    log_probs = []
    entropies = []
    
    for _ in range(max_new_tokens):
        # Get context (crop if too long)
        idx_cond = generated if generated.size(1) <= model.config.block_size else generated[:, -model.config.block_size:]
        
        # Forward pass - KEEP GRADIENTS for log probs
        logits, _ = model(idx_cond)
        
        # Get logits for last position
        logits = logits[:, -1, :] / temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs_all = F.log_softmax(logits, dim=-1)
        
        # Sample next token (detach probs for sampling, but keep original for gradients)
        with torch.no_grad():
            next_token = torch.multinomial(probs.detach(), num_samples=1)
        
        # Record log probability of chosen action (KEEP GRADIENTS)
        log_prob = log_probs_all[0, next_token.item()]
        log_probs.append(log_prob)
        
        # Record entropy
        entropy = -(probs * log_probs_all).sum()
        entropies.append(entropy)
        
        # Append to sequence (detach to save memory)
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated, torch.stack(log_probs), torch.stack(entropies)

def compute_reward(generated_ids, reward_model, reward_stoi, device='cpu'):
    """
    Compute reward for generated text using reward model
    """
    # Convert to text (assuming character-level)
    # Need to map from generation vocab to reward vocab
    # For simplicity, we'll pad/truncate to reward model's block size
    
    reward_block_size = reward_model.config.block_size
    
    # Truncate or pad
    if generated_ids.size(1) > reward_block_size:
        reward_input = generated_ids[:, :reward_block_size]
    else:
        # Pad with zeros
        padding = torch.zeros((1, reward_block_size - generated_ids.size(1)), dtype=torch.long, device=device)
        reward_input = torch.cat([generated_ids, padding], dim=1)
    
    # Get reward
    with torch.no_grad():
        reward = reward_model(reward_input)
    
    return reward.item()

def policy_gradient_update(model, optimizer, log_probs, reward, baseline, entropies, 
                          entropy_coef=0.01, device='cpu'):
    """
    Perform policy gradient update (REINFORCE)
    
    Loss = -log_prob * (reward - baseline) - entropy_coef * entropy
    
    The negative sign is because we do gradient descent, but want to maximize reward
    """
    model.train()
    
    # Advantage: reward - baseline (reduces variance)
    advantage = reward - baseline
    
    # Policy gradient loss
    # Negative because we want to maximize reward (but do gradient descent)
    policy_loss = -(log_probs * advantage).sum()
    
    # Entropy bonus (encourages exploration)
    entropy_loss = -entropies.sum() * entropy_coef
    
    # Total loss
    total_loss = policy_loss + entropy_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Clip gradients to prevent instability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return total_loss.item(), policy_loss.item(), entropy_loss.item()

def train_with_policy_gradient(
    base_model_path='out-shakespeare-char/ckpt.pt',
    reward_model_path='out-reward/reward_model.pt',
    out_dir='out-rl',
    num_iterations=1000,
    batch_size=4,  # Number of rollouts per update
    max_new_tokens=100,
    learning_rate=1e-5,  # Small LR for stability
    temperature=0.8,
    entropy_coef=0.01,  # Exploration bonus
    eval_interval=50,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Main training loop for policy gradient RL
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Training on {device}")
    print("=" * 80)
    
    # Load models
    print("Loading base model...")
    model, checkpoint = load_base_model(base_model_path, device)
    model.train()
    
    print("Loading reward model...")
    reward_model, reward_stoi, reward_itos = load_reward_model(reward_model_path, device)
    
    # Load metadata
    with open('data/shakespeare_char/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    # Optimizer - very small learning rate for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Prompts for generation
    prompts = ["ROMEO:", "JULIET:", "KING:", "The ", "First "]
    
    # Training loop
    baseline_reward = 0.0  # Running average of rewards
    best_reward = -float('inf')
    
    print("\nStarting RL training...")
    print("=" * 80)
    
    for iteration in range(num_iterations):
        iteration_rewards = []
        iteration_losses = []
        
        # Generate batch_size rollouts
        for _ in range(batch_size):
            # Random prompt
            prompt = np.random.choice(prompts)
            prompt_ids = torch.tensor([[meta['stoi'][c] for c in prompt]], 
                                     dtype=torch.long, device=device)
            
            # Set model to train mode for gradient tracking
            model.train()
            
            # Generate text with log probs (WITH GRADIENTS)
            generated_ids, log_probs, entropies = generate_with_log_probs(
                model, prompt_ids, max_new_tokens, temperature, device
            )
            
            # Compute reward (no gradients needed here)
            with torch.no_grad():
                reward = compute_reward(generated_ids, reward_model, reward_stoi, device)
            iteration_rewards.append(reward)
            
            # Policy gradient update
            loss, policy_loss, entropy_loss = policy_gradient_update(
                model, optimizer, log_probs, reward, baseline_reward, 
                entropies, entropy_coef, device
            )
            iteration_losses.append(loss)
        
        # Update baseline (moving average)
        avg_reward = np.mean(iteration_rewards)
        baseline_reward = 0.9 * baseline_reward + 0.1 * avg_reward
        
        # Logging
        if iteration % 10 == 0:
            print(f"Iter {iteration:4d} | Reward: {avg_reward:.4f} | "
                  f"Baseline: {baseline_reward:.4f} | Loss: {np.mean(iteration_losses):.4f}")
        
        # Evaluation
        if iteration % eval_interval == 0 and iteration > 0:
            print("\n" + "=" * 80)
            print(f"EVALUATION at iteration {iteration}")
            print("=" * 80)
            
            model.eval()
            eval_rewards = []
            
            # Generate samples
            for prompt in prompts:
                prompt_ids = torch.tensor([[meta['stoi'][c] for c in prompt]], 
                                         dtype=torch.long, device=device)
                
                with torch.no_grad():
                    generated_ids = model.generate(prompt_ids, max_new_tokens, 
                                                   temperature=0.8, top_k=200)
                
                reward = compute_reward(generated_ids, reward_model, reward_stoi, device)
                eval_rewards.append(reward)
                
                # Decode and print one example
                if prompt == prompts[0]:
                    text = ''.join([meta['itos'][i] for i in generated_ids[0].tolist()])
                    print(f"\nSample (reward={reward:.4f}):")
                    print(f"{text[:200]}...")
            
            avg_eval_reward = np.mean(eval_rewards)
            print(f"\nAverage evaluation reward: {avg_eval_reward:.4f}")
            print("=" * 80 + "\n")
            
            # Save best model
            if avg_eval_reward > best_reward:
                best_reward = avg_eval_reward
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iteration': iteration,
                    'best_reward': best_reward,
                    'config': model.config,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'rl_model_best.pt'))
                print(f"✓ Saved best model with reward {best_reward:.4f}")
            
            model.train()
    
    # Save final model
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': num_iterations,
        'best_reward': best_reward,
        'config': model.config,
    }
    torch.save(checkpoint, os.path.join(out_dir, 'rl_model_final.pt'))
    print(f"\n✓ Training complete! Final model saved.")
    print(f"Best reward achieved: {best_reward:.4f}")

if __name__ == '__main__':
    train_with_policy_gradient(
        num_iterations=500,  # Adjust based on your time
        batch_size=4,
        max_new_tokens=100,
        learning_rate=1e-5,
        eval_interval=50
    )