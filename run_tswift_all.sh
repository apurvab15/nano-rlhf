#!/bin/bash
# Full experiment loop: Train → Sample → Evaluate (robust logging)

sizes=(10000 50000 100000 200000 446319)
train_ref="data/taylor_swift_lyrics/taylor_swift_lyrics.txt"

for size in "${sizes[@]}"; do
  echo "=============================================="
  echo ">>> Training on dataset size: ${size}"
  echo "=============================================="

  out_dir="out/taylor_swift_lyrics/taylor_swift_lyrics_${size}"
  mkdir -p "$out_dir"

  log_file="${out_dir}/run_${size}.log"

  {
    echo "=== Starting run for dataset size ${size} ==="
    echo "Timestamp: $(date)"
    echo "----------------------------------------------"

    # 1️ TRAIN
    echo ">>> Training..."
    python -u train.py config/train_shakespeare_char.py \
      --device=cuda \
      --out_dir="$out_dir" \
      --batch_size=8 \
      --block_size=128 \
      --dataset="taylor_swift_lyrics_${size}" \
      --max_iters=2000 \
      --lr_decay_iters=2000 \
      --log_interval=250

    # 2️ SAMPLE
    echo ">>> Generating samples..."
    python -u sample.py \
      --out_dir="$out_dir" \
      --device=cuda \
      > "${out_dir}/samples_${size}.txt"

    # 3️ EVALUATE
    echo ">>> Evaluating metrics..."
    python -u custom_data_eval.py \
      --train "$train_ref" \
      --gen "${out_dir}/samples_${size}.txt" \
      | tee "${out_dir}/metrics_${size}.txt"

    echo "Completed run for dataset size: ${size}"
    echo "Timestamp: $(date)"
    echo "=============================================="
  } 2>&1 | tee "$log_file"
done

echo "All experiments completed!"
