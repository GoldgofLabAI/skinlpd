"""Assemble the per-case text context from SQLite reports.

Reproduces the format used at study time:

    === DATE: YYYY-MM-DD ===
    --- DERMATOLOGIST NOTE ---
    ...
    --- SURGICAL PATHOLOGY ---
    ...
    --- MOLECULAR PATHOLOGY ---
    ...
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_date(value) -> Optional[datetime]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text[:10], fmt)
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _format_block(title: str, body: Optional[str]) -> str:
    if body is None or (isinstance(body, float) and pd.isna(body)):
        return ""
    text = str(body).strip()
    if not text:
        return ""
    return f"--- {title} ---\n{text}\n"


def build_case_context(
    reports: pd.DataFrame,
    specimen_id: str,
    window_days: int = 30,
) -> str:
    """Build the assembled report context for one surgical specimen."""
    index_rows = reports[
        (reports["specimen_id"] == specimen_id) & (reports["case_type"] == "surgical")
    ]
    if index_rows.empty:
        logger.warning("No surgical report for specimen %s", specimen_id)
        return ""

    index = index_rows.iloc[0]
    patient_id = index["patient_id"]
    index_date = _parse_date(index.get("accession_date"))
    clinical_note = index.get("clinical_note")

    patient_reports = reports[reports["patient_id"] == patient_id].copy()
    if index_date is not None:
        lo = index_date - timedelta(days=window_days)
        hi = index_date + timedelta(days=window_days)

        def in_window(row) -> bool:
            if row["specimen_id"] == specimen_id:
                return True
            d = _parse_date(row.get("accession_date"))
            if d is None:
                return False
            return lo <= d <= hi

        patient_reports = patient_reports[patient_reports.apply(in_window, axis=1)]

    patient_reports = patient_reports.copy()
    patient_reports["_date"] = patient_reports["accession_date"].map(_parse_date)
    patient_reports = patient_reports.sort_values(["_date", "case_type"], na_position="last")

    sections = []
    date_label = index_date.strftime("%Y-%m-%d") if index_date else "unknown"
    sections.append(f"=== DATE: {date_label} ===\n")
    derm = _format_block("DERMATOLOGIST NOTE", clinical_note)
    if derm:
        sections.append(derm)

    for _, row in patient_reports.iterrows():
        case_type = str(row.get("case_type") or "").lower()
        if case_type == "surgical":
            title = "SURGICAL PATHOLOGY"
            if row["specimen_id"] != specimen_id:
                title = f"SURGICAL PATHOLOGY ({row['specimen_id']})"
        elif case_type == "molecular":
            title = "MOLECULAR PATHOLOGY"
        else:
            title = case_type.upper() or "REPORT"
        block = _format_block(title, row.get("report_text"))
        if block:
            sections.append(block)

    return "\n".join(sections).strip() + "\n"


def unique_tissue_sites(slides_df: pd.DataFrame, specimen_id: str) -> str:
    subset = slides_df[slides_df["specimen_id"] == specimen_id]
    sites = []
    for value in subset["part_description"].tolist():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in sites:
            sites.append(text)
    return "\n".join(sites)
