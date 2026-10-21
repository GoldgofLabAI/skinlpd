"""End-to-end mock-backend test for the LLM pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_sample_db import write_db
from skinlpd_llm.config import PipelineConfig
from skinlpd_llm.pipeline import SkinPathologyPipeline


FORBIDDEN = {
    "immunostaining_requested",
    "immunostaining_outcome",
    "molecular_requested",
    "is_clonal",
    "clonality_comment",
}


def test_mock_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "sample.db"
        write_db(db_path)
        out = tmp_path / "run"
        config = PipelineConfig.create_default(str(db_path), str(out), backend="mock")
        pipeline = SkinPathologyPipeline(config)
        first = pipeline.run_full_pipeline(output_prefix="demo")
        assert not first.empty
        assert "report_text" not in first.columns
        assert "clinical_note" not in first.columns
        for col in FORBIDDEN:
            assert col not in first.columns
        assert set(first["overall_diagnosis"].unique()) <= {"positive", "negative"}
        assert first["specimen_id"].nunique() >= 5
        assert "SYN-DROP40-01" not in set(first["specimen_id"])
        assert "SYN-AK-01" not in set(first["specimen_id"])  # neither excluded

        errored = list((out / "errored_outputs").glob("SYN-MALFORMED-01.*"))
        assert errored, "malformed mock response should be captured"

        second = SkinPathologyPipeline(
            PipelineConfig.create_default(str(db_path), str(tmp_path / "run2"), backend="mock")
        ).run_full_pipeline(output_prefix="demo")
        assert list(first["specimen_id"]) == list(second["specimen_id"])
        assert list(first["subtype"]) == list(second["subtype"])


if __name__ == "__main__":
    test_mock_pipeline_end_to_end()
    print("ok")
