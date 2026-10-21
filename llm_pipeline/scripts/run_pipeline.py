#!/usr/bin/env python3
"""Run the LLM abstraction pipeline on a SQLite database."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skinlpd_llm.config import DEMO_BEDROCK_MODEL_ID, PAPER_MODEL_ID, PipelineConfig
from skinlpd_llm.pipeline import SkinPathologyPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--backend", choices=["mock", "bedrock"], default="mock")
    parser.add_argument(
        "--model-id",
        default=None,
        help=f"Bedrock model id (paper checkpoint {PAPER_MODEL_ID}; demo default {DEMO_BEDROCK_MODEL_ID})",
    )
    parser.add_argument("--prefix", default="run")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = PipelineConfig.create_default(
        sqlite_db_path=args.db,
        base_dir=args.out,
        backend=args.backend,
        model_id=args.model_id,
    )
    pipeline = SkinPathologyPipeline(config)
    final = pipeline.run_full_pipeline(output_prefix=args.prefix, case_limit=args.limit)
    print(f"Wrote {len(final)} labeled rows to {config.get_file_path('results', args.prefix + '_final_dataset.csv')}")


if __name__ == "__main__":
    main()
