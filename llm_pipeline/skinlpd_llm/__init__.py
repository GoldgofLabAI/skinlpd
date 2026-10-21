"""Schema-guided LLM abstraction of dermatopathology reports."""

from .config import PipelineConfig, get_default_prompt_path
from .pipeline import SkinPathologyPipeline
from .schema import DiagnosisResponse

__all__ = [
    "DiagnosisResponse",
    "PipelineConfig",
    "SkinPathologyPipeline",
    "get_default_prompt_path",
]
