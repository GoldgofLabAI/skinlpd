"""Multithreaded LLM processing of pathology cases."""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .config import LLMConfig
from .context_builder import build_case_context, unique_tissue_sites
from .llm_backends import LLMBackend, create_backend
from .schema import DiagnosisResponse

logger = logging.getLogger(__name__)
thread_local_data = threading.local()


class ThreadSafeProgress:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.lock = threading.Lock()

    def update(self, success: bool = True):
        with self.lock:
            self.completed += 1
            if not success:
                self.failed += 1
            logger.info(
                "Progress: %s/%s completed (%s failed)",
                self.completed,
                self.total,
                self.failed,
            )

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "success_rate": (self.completed - self.failed) / max(self.completed, 1),
            }


class LLMProcessor:
    def __init__(
        self,
        llm_config: LLMConfig,
        output_dir: str,
        errored_outputs_dir: str,
        processing_config=None,
        reports: Optional[pd.DataFrame] = None,
        slides: Optional[pd.DataFrame] = None,
        backend: Optional[LLMBackend] = None,
        chart_window_days: int = 30,
    ):
        self.llm_config = llm_config
        self.processing_config = processing_config
        self.output_dir = Path(output_dir)
        self.errored_outputs_dir = Path(errored_outputs_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.errored_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.reports = reports if reports is not None else pd.DataFrame()
        self.slides = slides if slides is not None else pd.DataFrame()
        self.chart_window_days = chart_window_days
        self._shared_backend = backend
        self.base_prompt = self._load_prompt()
        self.llm_columns = [
            "combined_context",
            "tissue_sites",
            "overall_diagnosis",
            "reason",
            "positive_subtypes",
            "negative_subtypes",
            "primary_positive_subtype",
            "primary_negative_subtype",
        ]

    def _load_prompt(self) -> str:
        prompt_file = self.llm_config.prompt_file
        if prompt_file and Path(prompt_file).exists():
            return Path(prompt_file).read_text(encoding="utf-8")
        default = Path(__file__).parent / "prompt.txt"
        return default.read_text(encoding="utf-8")

    def _get_thread_llm(self) -> LLMBackend:
        if self._shared_backend is not None and self.llm_config.backend == "mock":
            return self._shared_backend
        if not hasattr(thread_local_data, "llm"):
            thread_local_data.llm = self._shared_backend or create_backend(self.llm_config)
        return thread_local_data.llm

    def _create_text_data_prompt(self, text_data: str, tissue_sites: str) -> str:
        return self.base_prompt + "\n\nPatient report: \n" + text_data + "\n\nTissue Sites: \n" + tissue_sites

    def _construct_prompt(self, specimen_id: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            context = build_case_context(self.reports, specimen_id, window_days=self.chart_window_days)
            if not context:
                return None, None
            tissue_sites = unique_tissue_sites(self.slides, specimen_id)
            return context, self._create_text_data_prompt(context, tissue_sites)
        except Exception as exc:
            logger.error("Error constructing prompt for %s: %s", specimen_id, exc)
            logger.error(traceback.format_exc())
            return None, None

    def _save_errored_llm_output(self, specimen_id: str, llm_response: Any, error_msg: str):
        try:
            response_content = str(llm_response)
            parsed = None
            try:
                parsed = json.loads(response_content) if response_content else None
            except (json.JSONDecodeError, TypeError):
                parsed = None
            payload = {
                "specimen_id": specimen_id,
                "error_message": error_msg,
                "llm_response": parsed if parsed is not None else response_content,
                "response_type": str(type(llm_response)),
                "timestamp": datetime.now().isoformat(),
            }
            suffix = "json" if isinstance(parsed, dict) else "txt"
            path = self.errored_outputs_dir / f"{specimen_id}.{suffix}"
            if suffix == "json":
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            else:
                path.write_text(
                    f"Specimen ID: {specimen_id}\nError Message: {error_msg}\n"
                    f"Response Type: {type(llm_response)}\nTimestamp: {payload['timestamp']}\n"
                    + "-" * 50
                    + "\n"
                    + response_content,
                    encoding="utf-8",
                )
            logger.info("Saved errored LLM output: %s", path)
        except Exception as save_error:
            logger.error("Failed to save errored output for %s: %s", specimen_id, save_error)

    def _process_response_to_dataframe_row(self, response: DiagnosisResponse) -> Dict[str, Any]:
        return {
            "overall_diagnosis": response.overall_diagnosis,
            "reason": response.reason,
            "positive_subtypes": json.dumps([s.model_dump() for s in response.positive_subtypes]),
            "negative_subtypes": json.dumps([s.model_dump() for s in response.negative_subtypes]),
            "primary_positive_subtype": response.primary_positive_subtype,
            "primary_negative_subtype": response.primary_negative_subtype,
        }

    def _process_response_dict_to_dataframe_row(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "overall_diagnosis": response_dict.get("overall_diagnosis", ""),
            "reason": response_dict.get("reason", ""),
            "positive_subtypes": json.dumps(response_dict.get("positive_subtypes") or []),
            "negative_subtypes": json.dumps(response_dict.get("negative_subtypes") or []),
            "primary_positive_subtype": response_dict.get("primary_positive_subtype") or "",
            "primary_negative_subtype": response_dict.get("primary_negative_subtype") or "",
        }

    def _process_single_specimen(self, args: Tuple[str, str]) -> Tuple[str, Dict[str, Any], bool]:
        specimen_id, _specimen_id_column = args
        try:
            logger.info("Processing specimen %s", specimen_id)
            formatted_reports, prompt = self._construct_prompt(specimen_id)
            if prompt is None:
                return specimen_id, {"error": "Could not construct prompt"}, False

            llm = self._get_thread_llm()
            start_time = time.time()
            extra_instructions = specimen_id if self.llm_config.backend == "mock" else None
            response = llm.generate_pydantic(
                prompt,
                DiagnosisResponse,
                instructions=extra_instructions,
                allow_partial=self.llm_config.allow_partial,
                fallback=self.llm_config.fallback,
            )
            elapsed = time.time() - start_time

            if response is None:
                error_msg = "LLM returned None response"
                self._save_errored_llm_output(specimen_id, response, error_msg)
                return specimen_id, {"error": error_msg}, False

            if isinstance(response, dict):
                try:
                    response = DiagnosisResponse.model_validate(response)
                except Exception as exc:
                    error_msg = f"Failed to convert dict response: {exc}"
                    self._save_errored_llm_output(specimen_id, response, error_msg)
                    return specimen_id, {"error": error_msg}, False
            elif isinstance(response, str):
                error_msg = "Failed to parse string response as DiagnosisResponse"
                logger.error("%s for %s", error_msg, specimen_id)
                self._save_errored_llm_output(specimen_id, response, error_msg)
                return specimen_id, {"error": error_msg}, False

            logger.info("LLM response for %s in %.2fs", specimen_id, elapsed)
            result_data = {
                "combined_context": formatted_reports,
                "tissue_sites": prompt.split("Tissue Sites:")[-1].strip() if "Tissue Sites:" in prompt else "",
            }
            try:
                result_data.update(self._process_response_to_dataframe_row(response))
            except Exception as exc:
                try:
                    dumped = response.model_dump() if hasattr(response, "model_dump") else dict(response)
                    result_data.update(self._process_response_dict_to_dataframe_row(dumped))
                except Exception:
                    error_msg = f"Response processing failed: {exc}"
                    self._save_errored_llm_output(specimen_id, response, error_msg)
                    return specimen_id, {"error": error_msg}, False
            return specimen_id, result_data, True
        except Exception as exc:
            logger.error("Unexpected error processing %s: %s", specimen_id, exc)
            logger.error(traceback.format_exc())
            return specimen_id, {"error": f"Unexpected error: {exc}"}, False

    def _save_checkpoint(self, df: pd.DataFrame, checkpoint_num: int, lock: threading.Lock):
        with lock:
            checkpoint_file = self.output_dir / f"checkpoint_{checkpoint_num}.csv"
            df.to_csv(checkpoint_file, index=False)
            logger.info("Saved checkpoint to %s", checkpoint_file)

    def process_cases(
        self,
        df: pd.DataFrame,
        specimen_id_column: str = "specimen_id",
        save_to_file: Optional[str] = None,
    ) -> pd.DataFrame:
        logger.info("Starting LLM processing of %s cases", len(df))
        if specimen_id_column not in df.columns:
            raise ValueError(f"Specimen ID column '{specimen_id_column}' not found")
        specimen_ids = df[specimen_id_column].tolist()
        for col in self.llm_columns:
            if col not in df.columns:
                df[col] = None
        if "report_text" in df.columns:
            df = df.drop(columns=["report_text"])
        if "clinical_note" in df.columns:
            df = df.drop(columns=["clinical_note"])

        progress = ThreadSafeProgress(len(specimen_ids))
        checkpoint_lock = threading.Lock()
        start_time = time.time()

        for batch_start in range(0, len(specimen_ids), self.llm_config.batch_size):
            batch_end = min(batch_start + self.llm_config.batch_size, len(specimen_ids))
            batch_ids = specimen_ids[batch_start:batch_end]
            batch_args = [(sid, specimen_id_column) for sid in batch_ids]
            with ThreadPoolExecutor(max_workers=self.llm_config.max_workers) as executor:
                future_to_specimen = {
                    executor.submit(self._process_single_specimen, args): args[0] for args in batch_args
                }
                for future in as_completed(
                    future_to_specimen,
                    timeout=self.llm_config.timeout_seconds * max(len(batch_args), 1),
                ):
                    specimen_id = future_to_specimen[future]
                    try:
                        result_id, result_data, success = future.result()
                        row_idx = df[df[specimen_id_column] == result_id].index[0]
                        if success:
                            for col, value in result_data.items():
                                df.loc[row_idx, col] = value
                        else:
                            for col in self.llm_columns:
                                df.loc[row_idx, col] = "ERROR"
                        progress.update(success)
                        if (
                            self.processing_config
                            and progress.completed % self.processing_config.checkpoint_interval == 0
                        ):
                            self._save_checkpoint(df, progress.completed, checkpoint_lock)
                    except Exception as exc:
                        logger.error("Future error for %s: %s", specimen_id, exc)
                        progress.update(False)

        stats = progress.get_stats()
        logger.info("Processing completed in %.2fs (success rate %.2f)", time.time() - start_time, stats["success_rate"])
        if save_to_file:
            df.to_csv(save_to_file, index=False)
            logger.info("Saved results to %s", save_to_file)
        return df
