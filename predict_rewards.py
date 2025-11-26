import torch
import torch.nn as nn
from model import GPT, GPTConfig
import pickle
import os

def load_reward_model(checkpoint_path='out-reward/reward_model.pt', device='cpu'):
    """Load trained reward model"""
    # Fix for PyTorch 2.6+ weights_only security change
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Define RewardModel class (must match training definition)
    class RewardModel(nn.Module):
        def __init__(self, gpt_model):
            super().__init__()
            self.transformer = gpt_model.transformer
            self.config = gpt_model.config
            self.reward_head = nn.Linear(gpt_model.config.n_embd, 1)
        
        def forward(self, idx):
            device = idx.device
            b, t = idx.size()
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            
            # Forward through transformer
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            
            for block in self.transformer.h:
                x = block(x)
            
            x = self.transformer.ln_f(x)
            x = x.mean(dim=1)
            reward = self.reward_head(x).squeeze(-1)
            return reward
    
    # Rebuild model
    gpt = GPT(config)
    model = RewardModel(gpt)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, stoi, itos, config

def load_generation_model(checkpoint_path='out-shakespeare-char/ckpt.pt', device='cpu'):
    """Load trained NanoGPT model for text generation"""
    # Fix for PyTorch 2.6+ weights_only security change
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    # Remove unwanted prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, checkpoint

def generate_text(model, start_text, meta, max_new_tokens=200, temperature=0.8, top_k=200, device='cpu'):
    """Generate text from the NanoGPT model"""
    encode = lambda s: [meta['stoi'][c] for c in s]
    decode = lambda l: ''.join([meta['itos'][i] for i in l])
    
    # Encode starting text
    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    generated_text = decode(y[0].tolist())
    return generated_text

def predict_reward(text, model, stoi, block_size=256, device='cpu'):
    """Predict lexical diversity reward for given text"""
    # Encode text
    encoded = [stoi.get(c, 0) for c in text[:block_size]]
    
    # Pad if necessary
    if len(encoded) < block_size:
        encoded = encoded + [0] * (block_size - len(encoded))
    
    # Convert to tensor
    x = torch.tensor([encoded], dtype=torch.long).to(device)
    
    # Predict
    with torch.no_grad():
        reward = model(x)
    
    return reward.item()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load reward model
    print("Loading reward model...")
    reward_model, stoi, itos, reward_config = load_reward_model(device=device)
    
    # Load generation model
    print("Loading NanoGPT generation model...")
    gen_model, gen_checkpoint = load_generation_model(device=device)
    
    # Load metadata for encoding/decoding
    meta_path = 'data/shakespeare_char/meta.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Generate multiple samples with different prompts and temperatures
    prompts = [
        "ROMEO:",
        "JULIET:",
        "First Citizen:",
        "KING:",
        "The "
    ]
    
    temperatures = [0.5, 0.8, 1.0, 1.2]
    
    all_generations = []
    
    print("\nGenerating text samples...\n")
    print("=" * 80)
    
    for prompt in prompts:
        for temp in temperatures:
            generated = generate_text(
                gen_model, 
                prompt, 
                meta, 
                max_new_tokens=150, 
                temperature=temp,
                device=device
            )
            
            # Calculate reward
            reward = predict_reward(generated, reward_model, stoi, device=device)
            
            all_generations.append({
                'text': generated,
                'reward': reward,
                'prompt': prompt,
                'temperature': temp
            })
    
    # Sort by reward
    all_generations.sort(key=lambda x: x['reward'], reverse=True)
    
    # Report high-reward texts
    print("\n🏆 TOP 5 HIGH-REWARD TEXTS (High Lexical Diversity):")
    print("=" * 80)
    for i, gen in enumerate(all_generations[:5], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Reward Score: {gen['reward']:.4f}")
        print(f"Prompt: '{gen['prompt']}' | Temperature: {gen['temperature']}")
        print(f"Generated Text:\n{gen['text'][:300]}...")
        print("-" * 80)
    
    # Report low-reward texts
    print("\n\n📉 BOTTOM 5 LOW-REWARD TEXTS (Low Lexical Diversity):")
    print("=" * 80)
    for i, gen in enumerate(all_generations[-5:], 1):
        print(f"\n--- Sample {i} ---")
        print(f"Reward Score: {gen['reward']:.4f}")
        print(f"Prompt: '{gen['prompt']}' | Temperature: {gen['temperature']}")
        print(f"Generated Text:\n{gen['text'][:300]}...")
        print("-" * 80)
    
    # Save results
    print("\n\nSaving results to 'reward_analysis.txt'...")
    with open('reward_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("LEXICAL DIVERSITY REWARD ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("HIGH-REWARD SAMPLES:\n\n")
        for i, gen in enumerate(all_generations[:5], 1):
            f.write(f"Sample {i} - Reward: {gen['reward']:.4f}\n")
            f.write(f"Prompt: '{gen['prompt']}' | Temperature: {gen['temperature']}\n")
            f.write(f"{gen['text']}\n\n")
            f.write("-" * 80 + "\n\n")
        
        f.write("\n\nLOW-REWARD SAMPLES:\n\n")
        for i, gen in enumerate(all_generations[-5:], 1):
            f.write(f"Sample {i} - Reward: {gen['reward']:.4f}\n")
            f.write(f"Prompt: '{gen['prompt']}' | Temperature: {gen['temperature']}\n")
            f.write(f"{gen['text']}\n\n")
            f.write("-" * 80 + "\n\n")
    
    print("✅ Analysis complete! Results saved to 'reward_analysis.txt'")
    
    # Summary statistics
    print("\n\n📊 SUMMARY STATISTICS:")
    print("=" * 80)
    rewards = [g['reward'] for g in all_generations]
    print(f"Total samples generated: {len(all_generations)}")
    print(f"Reward range: {min(rewards):.4f} - {max(rewards):.4f}")
    print(f"Mean reward: {sum(rewards)/len(rewards):.4f}")
    print(f"Median reward: {sorted(rewards)[len(rewards)//2]:.4f}")

if __name__ == '__main__':
    main()