#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SYN="${SYN:-$ROOT/synthetic}"
python scripts/make_synthetic_features.py --out "$SYN"
python train_with_balanced_options.py \
  --h5_directory "$SYN/features" \
  --master_df_path "$SYN/master.csv" \
  --output_dir "$SYN/run" \
  --num_epochs 3 --batch_size 1 --val_split 0.2 --test_split 0.2 \
  --enable_balancing --use_layer_norm --latent_dim 256 \
  --operating_point_selection youden \
  --case_level --case_level_method max \
  --case_id_column specimen_id --part_column part_description \
  --device "${DEVICE:-cpu}" --random_state 42
python test_evaluation.py \
  --test_csv_path "$SYN/run/test_split_patient_level.csv" \
  --model_path "$SYN/run/models/best_model.pt" \
  --h5_directory "$SYN/features" \
  --output_dir "$SYN/eval" \
  --case_level --case_level_method max \
  --case_id_column specimen_id --part_column part_description \
  --device "${DEVICE:-cpu}" --batch_size 1
python compute_attention.py \
  --input_csv_path "$SYN/master.csv" \
  --h5_directory "$SYN/features" \
  --model_path "$SYN/run/models/best_model.pt" \
  --output_dir "$SYN/attention" \
  --device "${DEVICE:-cpu}"
echo "MIL smoke test complete under $SYN"
