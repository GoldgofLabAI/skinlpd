"""SQLite source for report and slide discovery.

Tables use reviewer-facing names rather than the original CoPath dump columns:

    reports(patient_id, specimen_id, accession_date, case_type, clinical_note, report_text)
    slides(specimen_id, slide_id, part_description, stain_name, magnification, wsi_path)
"""

from __future__ import annotations

import logging
import sqlite3
from typing import List, Optional

import pandas as pd

from .config import CaseDiscoveryConfig

logger = logging.getLogger(__name__)


class SQLiteSource:
    def __init__(self, db_path: str, config: Optional[CaseDiscoveryConfig] = None):
        self.db_path = db_path
        self.config = config or CaseDiscoveryConfig()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _term_clause(self, column: str) -> str:
        terms = self.config.combined_include_terms
        if not terms:
            return "1=1"
        parts = [f"lower({column}) LIKE '%{term.lower().replace(chr(39), chr(39)*2)}%'" for term in terms]
        return "(" + " OR ".join(parts) + ")"

    def _stain_clause(self) -> str:
        stains = self.config.stain_criteria
        if not stains:
            return "1=1"
        parts = [f"lower(stain_name) LIKE '%{s.lower()}%'" for s in stains]
        return "(" + " OR ".join(parts) + ")"

    def discover_slides(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Return slide rows that pass the manuscript discovery filters."""
        cfg = self.config
        where = [
            "lower(part_description) LIKE ?",
            self._stain_clause(),
            "slide_id IS NOT NULL",
            "slide_id != ''",
        ]
        params: List = [f"%{cfg.slide_description_contains.lower()}%"]
        if cfg.require_20x:
            where.append("lower(magnification) = '20x'")
        if cfg.start_date:
            where.append("accession_date >= ?")
            params.append(cfg.start_date)
        if cfg.end_date:
            where.append("accession_date <= ?")
            params.append(cfg.end_date)

        sql = f"""
            SELECT
                s.specimen_id,
                s.slide_id,
                s.part_description,
                s.stain_name,
                s.magnification,
                s.wsi_path,
                r.patient_id,
                r.accession_date,
                r.case_type,
                r.clinical_note,
                r.report_text
            FROM slides s
            JOIN reports r
              ON r.specimen_id = s.specimen_id
             AND r.case_type = 'surgical'
            WHERE {' AND '.join(where)}
              AND {self._term_clause('r.report_text')}
            ORDER BY r.accession_date DESC, s.specimen_id, s.slide_id
        """
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)

        if cfg.exclude_if_has_40x and not df.empty:
            specimen_ids = df["specimen_id"].unique().tolist()
            placeholders = ",".join("?" * len(specimen_ids))
            with self._connect() as conn:
                has_40x = pd.read_sql_query(
                    f"""
                    SELECT DISTINCT specimen_id
                    FROM slides
                    WHERE specimen_id IN ({placeholders})
                      AND lower(magnification) = '40x'
                    """,
                    conn,
                    params=specimen_ids,
                )
            drop_ids = set(has_40x["specimen_id"].tolist())
            if drop_ids:
                logger.info("Dropping %d specimens that also have a 40x slide", len(drop_ids))
                df = df[~df["specimen_id"].isin(drop_ids)].copy()

        case_limit = limit if limit is not None else cfg.case_limit
        if case_limit is not None:
            keep = df["specimen_id"].drop_duplicates().head(case_limit)
            df = df[df["specimen_id"].isin(keep)].copy()

        logger.info("Discovered %d slides from %d specimens", len(df), df["specimen_id"].nunique() if not df.empty else 0)
        return df.reset_index(drop=True)

    def load_reports(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM reports", conn)

    def specimen_table(self, slides_df: pd.DataFrame) -> pd.DataFrame:
        """One row per surgical specimen for LLM processing."""
        cols = [
            "patient_id",
            "specimen_id",
            "accession_date",
            "clinical_note",
            "report_text",
        ]
        available = [c for c in cols if c in slides_df.columns]
        return slides_df[available].drop_duplicates(subset=["specimen_id"]).reset_index(drop=True)
