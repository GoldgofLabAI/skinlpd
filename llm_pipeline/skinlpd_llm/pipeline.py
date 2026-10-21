"""SQLite-only LLM abstraction pipeline."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .config import PipelineConfig
from .data_processor import DataProcessor
from .llm_processor import LLMProcessor
from .slide_mapper import SlideMapper
from .sqlite_source import SQLiteSource

logger = logging.getLogger(__name__)

REPORT_FIELD_COLUMNS = ("report_text", "clinical_note")


def _without_report_fields(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in REPORT_FIELD_COLUMNS if c in df.columns]
    return df.drop(columns=drop) if drop else df


class SkinPathologyPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.source = SQLiteSource(config.sqlite_db_path, config.case_discovery)
        self.data_processor = DataProcessor()
        self.slide_mapper = SlideMapper()
        self.reports = self.source.load_reports()
        self.llm_processor = LLMProcessor(
            llm_config=config.llm,
            output_dir=config.paths.data_dir,
            errored_outputs_dir=config.paths.errored_outputs_dir,
            processing_config=config.processing,
            reports=self.reports,
            slides=pd.DataFrame(),
            chart_window_days=config.case_discovery.chart_window_days,
        )

    def run_full_pipeline(
        self,
        output_prefix: str = "run",
        case_limit: Optional[int] = None,
        filter_unannotated: bool = True,
    ) -> pd.DataFrame:
        errors = self.config.validate()
        if errors:
            raise ValueError("; ".join(errors))

        slides_df = self.source.discover_slides(limit=case_limit)
        if slides_df.empty:
            raise RuntimeError("No slides discovered from the SQLite database")
        if self.config.save_intermediates:
            _without_report_fields(slides_df).to_csv(
                self.config.get_file_path("data", f"{output_prefix}_discovered_slides.csv"), index=False
            )

        self.llm_processor.slides = slides_df
        specimens = self.source.specimen_table(slides_df)
        llm_path = self.config.get_file_path("results", f"{output_prefix}_llm_processed.csv")
        processed = self.llm_processor.process_cases(specimens, save_to_file=llm_path)
        cleaned = self.data_processor.clean_llm_annotations(processed)
        success_df, error_df = self.data_processor.filter_error_cases(cleaned)
        if not error_df.empty:
            logger.info("%d specimens recorded as errors", len(error_df))
            error_df = _without_report_fields(error_df)
            error_df.to_csv(self.config.get_file_path("results", f"{output_prefix}_errors.csv"), index=False)
        if self.config.save_intermediates:
            _without_report_fields(cleaned).to_csv(self.config.get_file_path("results", f"{output_prefix}_cleaned.csv"), index=False)

        annotated = self.slide_mapper.merge_llm_annotations_with_slides(
            success_df,
            slides_df,
            filter_unannotated=filter_unannotated,
        )
        if self.config.save_intermediates:
            _without_report_fields(annotated).to_csv(
                self.config.get_file_path("results", f"{output_prefix}_annotated_slides.csv"),
                index=False,
            )

        final = self.data_processor.create_multilabel_dataset(
            annotated,
            self.config.processing.positive_subtypes,
            self.config.processing.negative_subtypes,
            self.config.processing.included_likelihoods,
        )
        # Keep only primary diagnoses for model development, matching Methods.
        if "is_primary" in final.columns:
            primary_mask = final["is_primary"].isin([True, "True", "true", 1, "1"])
            final = final[primary_mask].copy()
        final = _without_report_fields(final)
        final_path = self.config.get_file_path("results", f"{output_prefix}_final_dataset.csv")
        final.to_csv(final_path, index=False)
        stats = self.data_processor.get_dataset_statistics(final)
        logger.info("Final dataset: %s", stats)
        return final
