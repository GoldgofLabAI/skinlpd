#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python test_evaluation.py \
  --test_csv_path "${TEST_CSV:?set TEST_CSV}" \
  --model_path "${MODEL_PATH:?set MODEL_PATH}" \
  --h5_directory "${H5_DIR:?set H5_DIR}" \
  --output_dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --perform_bootstrapping --save_predictions --create_detailed_plots \
  --fontsize 18 --case_level --case_level_method max --batch_size 1 \
  --case_id_column "${CASE_ID_COLUMN:-sp_specimen_id}" \
  --part_column "${PART_COLUMN:-sl_part_description}" \
  --device "${DEVICE:-cuda:0}"
