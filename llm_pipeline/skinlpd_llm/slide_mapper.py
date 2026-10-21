"""Merge specimen-level LLM annotations onto slides via tissue-site matching."""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class SlideMapper:
    def matches_included_tissue_site(self, part_description: str, tissue_site: str) -> bool:
        if pd.isna(part_description) or pd.isna(tissue_site):
            return False
        return str(tissue_site).lower() in str(part_description).lower()

    def parse_subtypes_field(self, subtypes_str: Any) -> List[Dict]:
        if pd.isna(subtypes_str) or subtypes_str == "" or subtypes_str is None:
            return []
        if isinstance(subtypes_str, list):
            return subtypes_str
        if isinstance(subtypes_str, str):
            try:
                return json.loads(subtypes_str)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(subtypes_str)
                    return parsed if isinstance(parsed, list) else []
                except Exception:
                    logger.warning("Could not parse subtypes field")
                    return []
        return []

    def merge_llm_annotations_with_slides(
        self,
        case_table: pd.DataFrame,
        case_part_slide_mapping: pd.DataFrame,
        filter_unannotated: bool = False,
        drop_when_no_path: bool = False,
    ) -> pd.DataFrame:
        llm_columns = ["overall_diagnosis", "reason"]
        subtype_columns = ["subtype", "likelihood", "is_primary", "evidence_text"]
        case_dict = case_table.set_index("specimen_id").to_dict("index")
        annotated_rows = []

        for _, row in case_part_slide_mapping.iterrows():
            specimen_id = row["specimen_id"]
            part_description = row["part_description"]
            # Use plain dicts so pandas 3 does not coerce new columns (e.g. bool
            # is_primary) into a string-dtype Series inherited from the slide row.
            if specimen_id not in case_dict:
                new_row = row.to_dict()
                for col in llm_columns + subtype_columns:
                    new_row[col] = None
                annotated_rows.append(new_row)
                continue

            case_data = case_dict[specimen_id]
            base_row = row.to_dict()
            for col in llm_columns:
                base_row[col] = case_data.get(col)
            for col in subtype_columns:
                base_row[col] = None

            matching_subtypes = []
            for key in ("positive_subtypes", "negative_subtypes"):
                for subtype_dict in self.parse_subtypes_field(case_data.get(key)):
                    for tissue_site in subtype_dict.get("tissue_sites", []) or []:
                        if self.matches_included_tissue_site(part_description, tissue_site):
                            matching_subtypes.append(subtype_dict)
                            break

            if matching_subtypes:
                for subtype_dict in matching_subtypes:
                    annotated_row = dict(base_row)
                    annotated_row["subtype"] = subtype_dict.get("subtype")
                    annotated_row["likelihood"] = subtype_dict.get("likelihood")
                    is_primary = subtype_dict.get("is_primary")
                    annotated_row["is_primary"] = (
                        None if is_primary is None else bool(is_primary)
                    )
                    annotated_row["evidence_text"] = subtype_dict.get("evidence_text")
                    annotated_rows.append(annotated_row)
            else:
                annotated_rows.append(base_row)

        annotated = pd.DataFrame(annotated_rows)
        if not annotated.empty and "is_primary" in annotated.columns:
            annotated["is_primary"] = annotated["is_primary"].astype("boolean")
        if filter_unannotated and not annotated.empty:
            annotated = annotated[annotated["subtype"].notna()].copy()
        if drop_when_no_path and "wsi_path" in annotated.columns and not annotated.empty:
            annotated = annotated[annotated["wsi_path"].notna() & (annotated["wsi_path"] != "")]
        return annotated
