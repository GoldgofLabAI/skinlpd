# High-attention nuclear density

Quantifies nuclei per mm² in tiles above the within-slide 80th-percentile MIL attention weight (Figure 4 E–G).

Ported from experimental scripts `inspect_segmentation.py`, `boxplot_caselevel.py`, `preprocess_csv.py`, and `cellvit_scripts.sh`.

## Pipeline

1. Detect nuclei on each WSI with CellViT++ (SAM-H). See `run_cellvit.sh`. Requires CellViT-Inference v1.0.0b; this repo does not vendor weights.
2. Build a filelist with `build_cellvit_filelist.py`.
3. Run `compute_attention_density.py` with tissue GeoJSON (HEST/Trident contours), attention H5 from `mil_training/compute_attention.py`, and CellViT `cell_detection.json(.snappy)`.
4. Plot case-level boxplots with `plot_density_boxplots.py`.

Per-tile tissue area:

```
A_i = patch_size^2 * coverage * (base_mpp / 1000)^2   [mm^2]
```

High-attention density:

```
density_high_attn = sum(C_i) / sum(A_i)   over tiles with attention > 80th percentile
```

See `methods.md` for the full write-up.

## Synthetic smoke test (no WSI, no CellViT weights)

```bash
python scripts/make_synthetic_density_inputs.py --out synthetic
python compute_attention_density.py \
  --process-batch --attention-threshold 80 \
  --geojson-dir synthetic/geojson \
  --attention-dir synthetic/attention \
  --nuclei-dir synthetic/nuclei \
  --output-analysis synthetic/results
```
