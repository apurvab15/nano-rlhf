# data/my_dataset/make_subsets.py
import numpy as np, os

base_dir = "data/taylor_swift_lyrics"

full_train = np.fromfile(os.path.join(base_dir, "train.bin"), dtype=np.uint16)
sizes = [10000, 20000, 50000,100000,200000,300000,len(full_train)]   # number of characters/tokens to keep
print(f"Full training size: {len(full_train):,} tokens")

for s in sizes:
    out_dir = f"{base_dir}_{s}"
    os.makedirs(out_dir, exist_ok=True)
    subset = full_train[:s]
    subset.tofile(os.path.join(out_dir, "train.bin"))
    # reuse the same validation set
    val = np.fromfile(os.path.join(base_dir, "val.bin"), dtype=np.uint16)
    val.tofile(os.path.join(out_dir, "val.bin"))
    print(f"Created subset: {out_dir}")
