# Methods → code

| Methods step | Script |
|---|---|
| Keyword search of finalized reports; 20× H&E photo slides | `llm_pipeline/skinlpd_llm/sqlite_source.py` |
| Assemble dermatologist note + surgical report + linked molecular report + tissue sites | `llm_pipeline/skinlpd_llm/context_builder.py` |
| Llama 3.1 405B Instruct / Bedrock Converse, temperature 0.2, 2000 tokens, one generation | `llm_pipeline/skinlpd_llm/llm_backends.py`, `llm_processor.py`; prompt `prompt.txt`; schema `schema.py` |
| JSON object-span repair, partial Pydantic records, no retry | `llm_backends.LLMBedrock.generate_pydantic`; failed cases under `errored_outputs/` |
| Tissue-site substring match onto slides; keep definitive/likely/possible primary diagnoses | `slide_mapper.py`, `data_processor.py`, `pipeline.py` |
| HEST/Trident tissue detection; 20× 256 px tiles; frozen UNI 1024-D features | `mil_training/preprocessing/extract_wsi_features.py` |
| Gated-attention MIL, class-weighted CE, Youden threshold, case–part max aggregation | `mil_training/train_with_balanced_options.py`, `modules/gated_attn_pool.py`, `runners/01_train_binary.sh` |
| Held-out evaluation, bootstrap CIs | `mil_training/test_evaluation.py`, `runners/02_eval_internal.sh`, `03_eval_external.sh` |
| 20-trial random search | `mil_training/hyperparameter_search.py` |
| Tile attention maps | `mil_training/compute_attention.py` |
| UNI token-level attention overlays | `mil_training/postprocess_attention.py` (needs WSIs) |
| CellViT++ nuclei; 80th-percentile tiles; tissue area from GeoJSON × `base_mpp` | `attention_density/run_cellvit.sh`, `compute_attention_density.py` |
| Case-level density boxplots (Figure 4 E–G) | `attention_density/plot_density_boxplots.py` |

The study checkpoint `us.meta.llama3-1-405b-instruct-v1:0` is no longer offered on Bedrock in `us-east-2`. The demo Bedrock model is `meta.llama3-3-70b-instruct-v1:0`.
