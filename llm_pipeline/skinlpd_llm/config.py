"""Configuration for the SQLite-only LLM abstraction pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


PAPER_MODEL_ID = "us.meta.llama3-1-405b-instruct-v1:0"
DEMO_BEDROCK_MODEL_ID = "meta.llama3-3-70b-instruct-v1:0"


def get_default_prompt_path() -> str:
    return str(Path(__file__).parent / "prompt.txt")


@dataclass
class LLMConfig:
    model_id: str = PAPER_MODEL_ID
    max_tokens: int = 2000
    temperature: float = 0.2
    max_workers: int = 4
    batch_size: int = 20
    timeout_seconds: int = 300
    prompt_file: Optional[str] = None
    allow_partial: bool = True
    fallback: bool = True
    aws_profile: str = "mskEngineerUser"
    region_name: str = "us-east-2"
    backend: str = "mock"


@dataclass
class ProcessingConfig:
    checkpoint_interval: int = 50
    included_likelihoods: List[str] = field(
        default_factory=lambda: ["likely", "definitive", "possible"]
    )
    positive_subtypes: List[str] = field(
        default_factory=lambda: [
            "mf_non_folliculotropic",
            "mf_folliculotropic",
            "cytotoxic_ctcl_not_cd30",
            "cd30_tcell_lpd",
            "sezary_syndrome",
            "extracutaneous_or_pctcl_nos",
            "other_ctcl_tcell",
        ]
    )
    negative_subtypes: List[str] = field(
        default_factory=lambda: [
            "spongiotic_dermatosis",
            "psoriasiform_dermatosis",
            "lichenoid_dermatosis",
            "interface_dermatosis",
            "panniculitis",
            "perivascular_dermatitis",
            "interstitial_dermatitis",
            "pityriasiform_dermatosis",
            "depigmentation",
            "folliculitis",
            "blistering_dermatosis",
            "mucinoses_fibrosing_dermopathy",
            "neutrophilic_dermatosis",
            "infection",
            "vasculitis",
            "thrombotic_vasculopathy",
            "mast_cell",
            "other_inflammatory",
        ]
    )


@dataclass
class PathConfig:
    base_dir: str
    data_dir: str
    results_dir: str
    logs_dir: str
    checkpoints_dir: str
    errored_outputs_dir: str

    def __post_init__(self):
        for dir_path in [
            self.data_dir,
            self.results_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.errored_outputs_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_base_dir(cls, base_dir: str) -> "PathConfig":
        base_path = Path(base_dir)
        return cls(
            base_dir=str(base_path),
            data_dir=str(base_path / "data"),
            results_dir=str(base_path / "results"),
            logs_dir=str(base_path / "logs"),
            checkpoints_dir=str(base_path / "checkpoints"),
            errored_outputs_dir=str(base_path / "errored_outputs"),
        )


@dataclass
class CaseDiscoveryConfig:
    combined_include_terms: List[str] = field(
        default_factory=lambda: [
            "sézary syndrome",
            "folliculotropic",
            "pagetoid reticulosis",
            "woringer-kolopp",
            "granulomatous slack skin",
            "large cell transformation",
            "transformed mycosis fungoides",
            "lymphomatoid papulosis",
            "cd30",
            "anaplastic large cell lymphoma",
            "panniculitis-like",
            "gamma-delta",
            "epidermotropic",
            "psoriasiform",
            "lichenoid dermatitis",
            "interface dermatitis",
            "discoid lupus",
            "cutaneous lupus",
            "lupus panniculitis",
            "lupus profundus",
            "drug-induced",
            "drug eruption",
            "dress syndrome",
            "graft-versus-host",
            "graft vs host",
            "insect bite",
            "bite reaction",
            "pityriasis lichenoides",
            "pseudolymphoma",
            "reactive lymphoid",
            "bacterial folliculitis",
            "impetigo",
            "bacterial infection",
            "parapsoriasis",
            "mycosis fungoides",
            "sezary syndrome",
            "cutaneous t-cell lymphoma",
            "lymphoproliferative",
            "eczema",
            "psoriasis",
            "spongiotic dermatitis",
            "lupus",
            "drug-eruption",
            "gvhd",
            "arthropod bite",
            "lichen planus",
            "tinea",
            "dermatophyt",
        ]
    )
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    case_limit: Optional[int] = None
    slide_description_contains: str = "photo"
    stain_criteria: List[str] = field(default_factory=lambda: ["h&e", "recut"])
    require_20x: bool = True
    exclude_if_has_40x: bool = True
    chart_window_days: int = 30


@dataclass
class PipelineConfig:
    sqlite_db_path: str
    llm: LLMConfig
    processing: ProcessingConfig
    paths: PathConfig
    case_discovery: CaseDiscoveryConfig
    skip_existing: bool = True
    save_intermediates: bool = True

    @classmethod
    def create_default(
        cls,
        sqlite_db_path: str,
        base_dir: str,
        backend: str = "mock",
        model_id: Optional[str] = None,
    ) -> "PipelineConfig":
        llm = LLMConfig(backend=backend)
        if model_id:
            llm.model_id = model_id
        elif backend == "bedrock":
            llm.model_id = DEMO_BEDROCK_MODEL_ID
        llm.prompt_file = get_default_prompt_path()
        return cls(
            sqlite_db_path=sqlite_db_path,
            llm=llm,
            processing=ProcessingConfig(),
            paths=PathConfig.from_base_dir(base_dir),
            case_discovery=CaseDiscoveryConfig(),
        )

    def get_file_path(self, category: str, filename: str) -> str:
        category_dir = getattr(self.paths, f"{category}_dir")
        return str(Path(category_dir) / filename)

    def validate(self) -> List[str]:
        errors = []
        if not Path(self.sqlite_db_path).exists():
            errors.append(f"SQLite database not found: {self.sqlite_db_path}")
        if self.llm.prompt_file and not Path(self.llm.prompt_file).exists():
            errors.append(f"Prompt file not found: {self.llm.prompt_file}")
        if self.llm.backend not in {"mock", "bedrock"}:
            errors.append(f"Unknown LLM backend: {self.llm.backend}")
        return errors
