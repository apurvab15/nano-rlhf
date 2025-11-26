import pickle, os

config = {
    'block_size': 128,
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 384,
    'dropout': 0.0,
}

out_dir = 'out-shakespeare-char'
with open(os.path.join(out_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

print("✅ config.pkl recreated!")
