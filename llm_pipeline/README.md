# LLM report-abstraction pipeline

Schema-guided extraction of diagnostic category, subtype, likelihood, and tissue site from pathology reports. Matches Supplementary Information Section S2.

Immunohistochemical, molecular, and clonality results may appear in the assembled report text. They are not structured output fields.

## Synthetic demo

```bash
python scripts/make_sample_db.py --out sample.db
python scripts/run_pipeline.py --db sample.db --out runs/demo --backend mock
python tests/test_end_to_end.py
```

Saved CSV outputs omit `report_text` and `clinical_note`. The assembled report is used only in memory during LLM calls.

SQLite tables:

- `reports(patient_id, specimen_id, accession_date, case_type, clinical_note, report_text)`
- `slides(specimen_id, slide_id, part_description, stain_name, magnification, wsi_path)`

## Bedrock

The study used Llama 3.1 405B Instruct (`us.meta.llama3-1-405b-instruct-v1:0`) via the Amazon Bedrock Converse API in `us-east-2`. That checkpoint is no longer offered there. To try a live call:

```bash
python scripts/run_pipeline.py --db sample.db --out runs/bedrock \
  --backend bedrock --model-id meta.llama3-3-70b-instruct-v1:0
```

Requires AWS credentials (default profile `mskEngineerUser`) and `boto3`.
