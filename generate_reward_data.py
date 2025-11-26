import numpy as np
import pickle
import os

def calculate_lexical_diversity(text):
    """
    Calculate lexical diversity using Type-Token Ratio (TTR)
    TTR = (unique words) / (total words)
    
    You can also use other metrics like:
    - MTLD (Measure of Textual Lexical Diversity)
    - HD-D (Hypergeometric distribution D)
    - Yule's K
    """
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    # Type-Token Ratio
    ttr = unique_words / total_words
    
    # Normalize to 0-1 range (TTR is already 0-1, but you could scale differently)
    return 
    

def prepare_reward_dataset(input_file, output_dir='data/reward'):
    """
    Prepare dataset for reward model training.
    Format: each example is (text, reward_score)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read your text data
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into chunks (you can adjust chunk size)
    chunk_size = 100  # characters per sample
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Calculate rewards for each chunk
    samples = []
    for chunk in chunks:
        if len(chunk.strip()) > 10:  # skip very short chunks
            reward = calculate_lexical_diversity(chunk)
            samples.append({
                'text': chunk,
                'reward': reward
            })
    
    # Split into train/val
    split_idx = int(0.9 * len(samples))
    train_data = samples[:split_idx]
    val_data = samples[split_idx:]
    
    # Save as pickle files
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    
    print(f"Created {len(train_data)} training samples")
    print(f"Created {len(val_data)} validation samples")
    print(f"Reward range: {min(s['reward'] for s in samples):.3f} - {max(s['reward'] for s in samples):.3f}")

if __name__ == '__main__':
    # Using Shakespeare dataset
    input_file = 'data/shakespeare_char/input.txt'
    prepare_reward_dataset(input_file, output_dir='data/reward')