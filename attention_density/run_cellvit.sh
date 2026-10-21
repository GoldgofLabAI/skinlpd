#!/usr/bin/env bash
# CellViT++ nuclear detection (CellViT-Inference v1.0.0b, SAM-H).
# Requires the CellViT environment; this repository does not vendor the weights.
set -euo pipefail
FILELIST="${1:?usage: run_cellvit.sh FILELIST.csv OUTDIR}"
OUTDIR="${2:?usage: run_cellvit.sh FILELIST.csv OUTDIR}"
cellvit-inference \
  --model SAM \
  --geojson \
  --graph \
  --compression \
  --outdir "$OUTDIR" \
  process_dataset \
  --wsi_filelist "$FILELIST"
