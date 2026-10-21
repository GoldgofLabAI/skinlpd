# skinlpd

Code that reproduces the algorithms in our study, *Large-scale deep learning–based discrimination of heterogeneous cutaneous T-cell lymphomas from diverse inflammatory skin diseases*:

1. **LLM report abstraction** (`llm_pipeline/`) — schema-guided extraction of diagnostic category, subtype, likelihood, and tissue site from pathology reports.
2. **Gated-attention MIL** (`mil_training/`) — UNI tile features, binary slide-level training, case–part evaluation, and attention maps.
3. **High-attention nuclear density** (`attention_density/`) — CellViT++ nuclei in tiles above the within-slide 80th-percentile attention weight.

Each package has its own README and a synthetic end-to-end demo. No institutional database, patient identifiers, or whole-slide images are required to run the demos.

See [docs/methods_to_code.md](docs/methods_to_code.md) for a mapping from the manuscript Methods to these scripts.

## Installation

```bash
# Python 3.12 recommended (3.11+ required for the pinned MIL deps)
python3.12 -m venv .venv   # or: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

GPU training needs a CUDA-enabled PyTorch build matching your driver; the pins in `mil_training/requirements.txt` install the default PyPI wheels.

## Quick start

```bash
# LLM pipeline (offline mock backend)
cd llm_pipeline
python scripts/make_sample_db.py --out sample.db
python scripts/run_pipeline.py --db sample.db --out runs/demo --backend mock

# MIL trainer (synthetic 1024-D bags)
cd ../mil_training
python scripts/make_synthetic_features.py --out synthetic
python train_with_balanced_options.py \
  --h5_directory synthetic/features \
  --master_df_path synthetic/master.csv \
  --output_dir synthetic/run \
  --num_epochs 3 --batch_size 1 --val_split 0.2 --test_split 0.2 \
  --enable_balancing --use_layer_norm --latent_dim 256 \
  --operating_point_selection youden --device cpu

# Attention density (synthetic GeoJSON + nuclei + attention H5)
cd ../attention_density
python scripts/make_synthetic_density_inputs.py --out synthetic
python compute_attention_density.py \
  --process-batch --attention-threshold 80 \
  --attention-dir synthetic/attention \
  --nuclei-dir synthetic/nuclei \
  --geojson-dir synthetic/geojson \
  --output-analysis synthetic/results
```

The Bedrock backend is optional. The study used Llama 3.1 405B Instruct (`us.meta.llama3-1-405b-instruct-v1:0`). That checkpoint is no longer offered on Amazon Bedrock in `us-east-2`; the demo uses `meta.llama3-3-70b-instruct-v1:0` when `--backend bedrock` is selected.
