# prepare_char_shared_vocab.py
import os, pickle, numpy as np

meta = pickle.load(open('data/shakespeare_char/meta.pkl', 'rb'))
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

data = open('data/taylor_swift_lyrics/taylor_swift_lyrics.txt', encoding='utf-8').read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
train_ids.tofile('data/taylor_swift_lyrics/train.bin')
val_ids.tofile('data/taylor_swift_lyrics/val.bin')
print("✅ Taylor data prepared using Shakespeare vocab")
