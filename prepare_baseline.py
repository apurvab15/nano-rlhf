"""
Prepare prompts and collect baseline samples from NanoGPT.
"""

import os
import pickle
import torch
import numpy as np
from contextlib import nullcontext
from verifier import verify_batch

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
init_from = 'resume'  # or 'gpt2', depending on your setup
out_dir = 'out-shakespeare-char'  # where your trained model is
start = "\n"  # or any starting text
num_prompts = 100  # number of prompts to generate
num_samples_per_prompt = 4  # samples per prompt for baseline
max_new_tokens = 50  # max tokens to generate
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = False  # set True if you have PyTorch 2.0+

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
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

model.eval()
model.to(device)
if compile_model:
    model = torch.compile(model)

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

# -----------------------------------------------------------------------------
# Generate prompts
# -----------------------------------------------------------------------------
print("Generating prompts...")
prompts = []

# Simple prompts - you can customize these
prompt_templates = [
    "Once upon a time",
    "The weather today",
    "In the future",
    "Scientists have discovered",
    "The best way to",
    "Everyone knows that",
    "I remember when",
    "According to experts",
    "It is often said",
    "One day",
]

# Extend prompts by repeating templates
for i in range(num_prompts):
    prompt = prompt_templates[i % len(prompt_templates)]
    prompts.append(prompt)

print(f"Created {len(prompts)} prompts")

# -----------------------------------------------------------------------------
# Collect baseline samples
# -----------------------------------------------------------------------------
print("\nCollecting baseline samples...")
all_samples = []
all_rewards = []

for i, prompt in enumerate(prompts):
    if i % 10 == 0:
        print(f"Processing prompt {i+1}/{num_prompts}...")
    
    prompt_samples = []
    prompt_rewards = []
    
    for _ in range(num_samples_per_prompt):
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
        
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        # Generate
        with torch.no_grad():
            with ctx:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        # Decode
        completion = decode(y[0].tolist())
        # Extract only the generated part (after prompt)
        generated = completion[len(prompt):]
        
        prompt_samples.append(generated)
    
    # Compute rewards
    prompt_rewards = verify_batch(prompt_samples)
    
    all_samples.append(prompt_samples)
    all_rewards.extend(prompt_rewards)

# -----------------------------------------------------------------------------
# Report statistics
# -----------------------------------------------------------------------------
mean_reward = np.mean(all_rewards)
std_reward = np.std(all_rewards)

print("\n" + "="*60)
print("BASELINE RESULTS")
print("="*60)
print(f"Number of prompts: {num_prompts}")
print(f"Samples per prompt: {num_samples_per_prompt}")
print(f"Total samples: {len(all_rewards)}")
print(f"\nMean verifier score: {mean_reward:.3f}")
print(f"Std verifier score: {std_reward:.3f}")
print(f"Min score: {min(all_rewards):.3f}")
print(f"Max score: {max(all_rewards):.3f}")

print("\n" + "="*60)
print("SAMPLE COMPLETIONS")
print("="*60)

# Show a few examples
for i in range(min(5, len(prompts))):
    print(f"\nPrompt: '{prompts[i]}'")
    for j, (sample, reward) in enumerate(zip(all_samples[i], verify_batch(all_samples[i]))):
        print(f"  Sample {j+1} (score={reward:.1f}): '{sample[:100]}'")

# Save results
results = {
    'prompts': prompts,
    'samples': all_samples,
    'rewards': all_rewards,
    'mean_reward': mean_reward,
    'std_reward': std_reward
}

with open('baseline_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n✓ Baseline results saved to 'baseline_results.pkl'")