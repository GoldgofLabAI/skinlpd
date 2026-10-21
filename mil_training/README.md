# Gated-attention MIL (paper subset)

Binary slide-level gated-attention MIL used for CTCL/CLPD vs BID. Hematology, survival, and regression trainers from later `mil-training` commits are not included.

## Synthetic demo

```bash
bash tests/test_end_to_end.sh
```

Or with a GPU: `DEVICE=cuda:1 bash tests/test_end_to_end.sh`

The trainer expects a patient identifier column named `MRN` (the study used medical record numbers). The synthetic CSV also has `patient_id` with the same values.

Manuscript hyperparameters are in `runners/01_train_binary.sh`–`04_compute_attention.sh`. Point `H5_DIR`, `MASTER_CSV` / `TEST_CSV`, `MODEL_PATH`, and `OUTPUT_DIR` at your data.

## WSI features

`preprocessing/extract_wsi_features.py` runs Trident tissue segmentation (`grandqc_artifact`) then UNI v1 features at 20×, 256 px, 0 overlap. Requires `HUGGINGFACE_HUB_TOKEN` and a Trident install.

## Attention overlays on WSIs

`postprocess_attention.py` fuses MIL tile attention with UNI token maps. It needs OpenSlide, Trident/UNI, and whole-slide images, so it is documented rather than covered by the synthetic smoke test.
