#!/usr/bin/env bash
# Manuscript binary training recipe. Set H5_DIR, MASTER_CSV, OUTPUT_DIR, DEVICE.
set -euo pipefail
cd "$(dirname "$0")/.."
python train_with_balanced_options.py \
  --dropout 0.247 \
  --early_stopping_patience 5 \
  --num_epochs 30 \
  --use_layer_norm \
  --latent_dim 256 \
  --enable_balancing \
  --h5_directory "${H5_DIR:?set H5_DIR}" \
  --master_df_path "${MASTER_CSV:?set MASTER_CSV}" \
  --output_dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --learning_rate 0.0003 \
  --batch_size 1 \
  --val_split 0.1 \
  --save_training_logs_too \
  --operating_point_selection youden \
  --case_level --case_level_method max \
  --case_id_column "${CASE_ID_COLUMN:-sp_specimen_id}" \
  --part_column "${PART_COLUMN:-sl_part_description}" \
  --device "${DEVICE:-cuda:0}"
