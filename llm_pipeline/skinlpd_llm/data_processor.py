"""Cleaning and filtering of LLM annotations."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    def convert_tissue_sites(self, subtypes_col: pd.Series) -> pd.Series:
        def process_entry(entry):
            if pd.isna(entry) or entry == "" or entry == "nan":
                return entry
            try:
                data = json.loads(entry)
                if not isinstance(data, list):
                    return entry
                for subtype_dict in data:
                    if "tissue_sites" in subtype_dict and isinstance(subtype_dict["tissue_sites"], list):
                        tissue_sites = subtype_dict["tissue_sites"]
                        if tissue_sites and isinstance(tissue_sites[0], dict):
                            subtype_dict["tissue_sites"] = [
                                ts["tissue_site"]
                                for ts in tissue_sites
                                if isinstance(ts, dict) and "tissue_site" in ts
                            ]
                        elif tissue_sites and not all(isinstance(ts, str) for ts in tissue_sites):
                            subtype_dict["tissue_sites"] = [str(ts) for ts in tissue_sites]
                return json.dumps(data, ensure_ascii=False)
            except Exception as exc:
                logger.warning("Failed to process tissue_sites entry: %s", exc)
                return entry

        return subtypes_col.apply(process_entry)

    def clean_llm_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        for col in ("negative_subtypes", "positive_subtypes"):
            if col in df_clean.columns:
                df_clean[col] = self.convert_tissue_sites(df_clean[col])
        return df_clean

    def filter_error_cases(self, df: pd.DataFrame, error_value: str = "ERROR") -> Tuple[pd.DataFrame, pd.DataFrame]:
        error_mask = df["overall_diagnosis"] == error_value
        return df[~error_mask].reset_index(drop=True), df[error_mask].reset_index(drop=True)

    def create_multilabel_dataset(
        self,
        df: pd.DataFrame,
        config_positive_subtypes: List[str],
        config_negative_subtypes: List[str],
        included_likelihoods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if included_likelihoods is None:
            included_likelihoods = ["likely", "definitive", "possible"]
        df_filtered = df[df["likelihood"].isin(included_likelihoods)].copy()
        df_positive = df_filtered[df_filtered["overall_diagnosis"] == "positive"]
        df_negative = df_filtered[df_filtered["overall_diagnosis"] == "negative"]
        df_positive_controlled = df_positive[df_positive["subtype"].isin(config_positive_subtypes)]
        df_negative_controlled = df_negative[df_negative["subtype"].isin(config_negative_subtypes)]
        return pd.concat([df_positive_controlled, df_negative_controlled], ignore_index=True)

    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
        }
        if "overall_diagnosis" in df.columns:
            stats["diagnosis_distribution"] = df["overall_diagnosis"].value_counts().to_dict()
        if "subtype" in df.columns:
            stats["subtype_distribution"] = df["subtype"].value_counts().to_dict()
        if "likelihood" in df.columns:
            stats["likelihood_distribution"] = df["likelihood"].value_counts().to_dict()
        if "specimen_id" in df.columns:
            stats["unique_specimens"] = df["specimen_id"].nunique()
        if "slide_id" in df.columns:
            stats["unique_slides"] = df["slide_id"].nunique()
        if "wsi_path" in df.columns:
            has_path = df["wsi_path"].notna().sum()
            stats["slides_with_filepath"] = int(has_path)
            stats["filepath_coverage"] = has_path / len(df) if len(df) else 0
        return stats
