import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig
import numpy as np
from tqdm import tqdm

class RewardDataset(Dataset):
    """Dataset for reward model training"""
    def __init__(self, data_path, block_size=256):
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        self.block_size = block_size
        
        # Create vocabulary from all texts
        all_text = ''.join([s['text'] for s in self.samples])
        chars = sorted(list(set(all_text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        reward = sample['reward']
        
        # Encode text to indices
        encoded = [self.stoi.get(c, 0) for c in text[:self.block_size]]
        
        # Pad if necessary
        if len(encoded) < self.block_size:
            encoded = encoded + [0] * (self.block_size - len(encoded))
        
        x = torch.tensor(encoded, dtype=torch.long)
        y = torch.tensor(reward, dtype=torch.float32)
        
        return x, y

class RewardModel(nn.Module):
    """GPT backbone + regression head for reward prediction"""
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
        
        # Forward through transformer (copied from GPT.forward)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        # Pool across sequence (mean pooling)
        x = x.mean(dim=1)
        
        # Predict reward
        reward = self.reward_head(x).squeeze(-1)
        return reward

def train_reward_model(
    data_dir='data/reward',
    out_dir='out-reward',
    block_size=256,
    batch_size=64,
    learning_rate=3e-4,
    max_iters=5000,
    eval_interval=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = RewardDataset(os.path.join(data_dir, 'train.pkl'), block_size)
    val_dataset = RewardDataset(os.path.join(data_dir, 'val.pkl'), block_size)
    
    # Use same vocab as train
    val_dataset.stoi = train_dataset.stoi
    val_dataset.itos = train_dataset.itos
    val_dataset.vocab_size = train_dataset.vocab_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Vocab size: {train_dataset.vocab_size}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize GPT model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=train_dataset.vocab_size,
        n_layer=6,  # smaller model for reward
        n_head=6,
        n_embd=384
    )
    
    gpt = GPT(config)
    model = RewardModel(gpt).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    iter_num = 0
    best_val_loss = float('inf')
    
    print(f"Training on {device}")
    
    for epoch in range(max_iters // len(train_loader) + 1):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            pred_reward = model(x)
            loss = criterion(pred_reward, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter_num % eval_interval == 0:
                # Evaluate
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        val_pred = model(val_x)
                        val_loss = criterion(val_pred, val_y)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                print(f"iter {iter_num}: train loss {loss.item():.4f}, val loss {avg_val_loss:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    checkpoint = {
                        'model': model.state_dict(),
                        'config': config,
                        'stoi': train_dataset.stoi,
                        'itos': train_dataset.itos,
                        'val_loss': avg_val_loss
                    }
                    torch.save(checkpoint, os.path.join(out_dir, 'reward_model.pt'))
                    print(f"Saved checkpoint with val loss {avg_val_loss:.4f}")
                
                model.train()
            
            iter_num += 1
            if iter_num >= max_iters:
                break
        
        if iter_num >= max_iters:
            break
    
    print("Training complete!")
    return model

if __name__ == '__main__':
    model = train_reward_model()