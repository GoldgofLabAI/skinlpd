#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python compute_attention.py \
  --input_csv_path "${TEST_CSV:?set TEST_CSV}" \
  --h5_directory "${H5_DIR:?set H5_DIR}" \
  --model_path "${MODEL_PATH:?set MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --save_predictions_csv \
  --device "${DEVICE:-cuda:0}"
