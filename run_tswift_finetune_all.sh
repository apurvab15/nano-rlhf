#!/bin/bash
# Fine-tune Shakespeare model on your dataset, generate samples, and evaluate

# === CONFIG ===
BASE_DIR="out-shakespeare-char"                 # pretrained Shakespeare model
DATASET="data/taylor_swift_lyrics"              # your dataset
OUT_DIR="out/taylor_finetune"                   # where fine-tuned model/logs go
MAX_ITERS=1000
LOG_INTERVAL=100
EVAL_SCRIPT="custom_data_eval.py"

# === 1️⃣ Prepare fine-tune folder ===
mkdir -p "$OUT_DIR"
cp $BASE_DIR/{ckpt.pt,config.pkl,meta.pkl} "$OUT_DIR"

echo "============================================="
echo "🚀 Starting fine-tuning..."
echo "============================================="

python -u train.py config/train_shakespeare_char.py \
  --device=cuda \
  --out_dir="$OUT_DIR" \
  --dataset="$DATASET" \
  --init_from=resume \
  --max_iters=$MAX_ITERS \
  --lr_decay_iters=$MAX_ITERS \
  --batch_size=8 \
  --block_size=128 \
  --log_interval=$LOG_INTERVAL

echo "✅ Fine-tuning complete."

# === 2️⃣ Generate text samples ===
echo "============================================="
echo "🪶 Generating samples..."
echo "============================================="

python -u sample.py \
  --out_dir="$OUT_DIR" \
  --device=cuda \
  > "$OUT_DIR/samples.txt"

echo "✅ Samples generated: $OUT_DIR/samples.txt"

# === 3️⃣ Evaluate samples ===
echo "============================================="
echo "📊 Evaluating samples..."
echo "============================================="

python -u "$EVAL_SCRIPT" \
  --train "$DATASET/taylor_swift_lyrics.txt" \
  --gen "$OUT_DIR/samples.txt" \
  | tee "$OUT_DIR/metrics.txt"

echo "✅ Evaluation complete. Results saved to $OUT_DIR/metrics.txt"
echo "============================================="
echo "🎯 Fine-tuning + evaluation finished successfully!"
