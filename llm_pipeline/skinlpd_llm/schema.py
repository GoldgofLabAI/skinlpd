"""S2-aligned diagnostic schema for LLM report abstraction.

Fields follow Supplementary Information Section S2: overall category, subtype
mentions with likelihood and tissue sites, and a short reason. Immunohistochemical,
molecular, and clonality findings may appear in the report text supplied to the
model; they are not structured output fields.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class OverallDiagnosisEnum(str, Enum):
    POSITIVE = "positive"  # CTCL / cutaneous T-cell lymphoproliferative disorder
    NEGATIVE = "negative"  # inflammatory / benign mimics
    NEITHER = "neither"


class LikelihoodEnum(str, Enum):
    DEFINITIVE = "definitive"
    LIKELY = "likely"
    POSSIBLE = "possible"
    RULED_OUT = "ruled_out"
    UNSPECIFIED = "unspecified"


class PositiveSubtypeEnum(str, Enum):
    MF_NON_FOLLICULOTROPIC = "mf_non_folliculotropic"
    MF_FOLLICULOTROPIC = "mf_folliculotropic"
    CYTOTOXIC_CTCL_NOT_CD30 = "cytotoxic_ctcl_not_cd30"
    CD30_TCELL_LPD = "cd30_tcell_lpd"
    SEZARY_SYNDROME = "sezary_syndrome"
    EXTRACUTANEOUS_OR_PCTCL_NOS = "extracutaneous_or_pctcl_nos"
    OTHER_CTCL_TCELL = "other_ctcl_tcell"


class NegativeSubtypeEnum(str, Enum):
    SPONGIOTIC_DERMATOSIS = "spongiotic_dermatosis"
    PSORIASIFORM_DERMATOSIS = "psoriasiform_dermatosis"
    LICHENOID_DERMATOSIS = "lichenoid_dermatosis"
    INTERFACE_DERMATOSIS = "interface_dermatosis"
    PANNICULITIS = "panniculitis"
    PERIVASCULAR_DERMATITIS = "perivascular_dermatitis"
    INTERSTITIAL_DERMATITIS = "interstitial_dermatitis"
    PITYRIASIFORM_DERMATOSIS = "pityriasiform_dermatosis"
    DEPIGMENTATION = "depigmentation"
    FOLLICULITIS = "folliculitis"
    BLISTERING_DERMATOSIS = "blistering_dermatosis"
    MUCINOSES_FIBROSING_DERMOPATHY = "mucinoses_fibrosing_dermopathy"
    NEUTROPHILIC_DERMATOSIS = "neutrophilic_dermatosis"
    INFECTION = "infection"
    VASCULITIS = "vasculitis"
    THROMBOTIC_VASCULOPATHY = "thrombotic_vasculopathy"
    MAST_CELL = "mast_cell"
    OTHER_INFLAMMATORY = "other_inflammatory"


class PositiveSubtypeMention(BaseModel):
    subtype: PositiveSubtypeEnum = Field(..., description="CTCL/CLPD subtype code")
    likelihood: LikelihoodEnum = Field(default=LikelihoodEnum.UNSPECIFIED)
    is_primary: bool = Field(default=False, description="True if this is the primary/favored CTCL subtype")
    evidence_text: Optional[str] = Field(
        default=None,
        description="Shortest quote from the report supporting this subtype (optional)",
    )
    tissue_sites: List[str] = Field(
        description="Tissue site names for this subtype, copied from the Tissue Sites section"
    )
    other_label_detail: Optional[str] = Field(
        default=None,
        description="If subtype is OTHER_CTCL_TCELL, a brief descriptor",
    )


class NegativeSubtypeMention(BaseModel):
    subtype: NegativeSubtypeEnum = Field(..., description="Inflammatory mimic subtype code")
    likelihood: LikelihoodEnum = Field(default=LikelihoodEnum.UNSPECIFIED)
    is_primary: bool = Field(default=False, description="True if this is the primary/favored inflammatory subtype")
    evidence_text: Optional[str] = Field(
        default=None,
        description="Shortest quote from the report supporting this subtype (optional)",
    )
    tissue_sites: List[str] = Field(
        description="Tissue site names for this subtype, copied from the Tissue Sites section"
    )
    other_label_detail: Optional[str] = Field(
        default=None,
        description="If subtype is OTHER_INFLAMMATORY, a brief descriptor",
    )


class DiagnosisResponse(BaseModel):
    overall_diagnosis: OverallDiagnosisEnum = Field(
        description="'positive' (CTCL/CLPD), 'negative' (inflammatory), or 'neither'"
    )
    reason: str = Field(description="Brief explanation for the classification decision")
    positive_subtypes: List[PositiveSubtypeMention] = Field(
        default_factory=list,
        description="All CTCL/CLPD subclassifications mentioned (if any)",
    )
    negative_subtypes: List[NegativeSubtypeMention] = Field(
        default_factory=list,
        description="All negative subclassifications mentioned (if any)",
    )
    primary_positive_subtype: Optional[PositiveSubtypeEnum] = Field(
        default=None,
        description="Primary CTCL/CLPD subtype if overall_diagnosis == positive",
    )
    primary_negative_subtype: Optional[NegativeSubtypeEnum] = Field(
        default=None,
        description="Primary inflammatory/mimic subtype if overall_diagnosis == negative",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_consistency(cls, values):
        overall = values.get("overall_diagnosis")
        pos = values.get("positive_subtypes", []) or []
        neg = values.get("negative_subtypes", []) or []
        primary_pos = values.get("primary_positive_subtype")
        primary_neg = values.get("primary_negative_subtype")

        def _subtype_of(mention):
            if isinstance(mention, dict):
                return mention.get("subtype")
            return getattr(mention, "subtype", None)

        def _is_primary(mention):
            if isinstance(mention, dict):
                return bool(mention.get("is_primary"))
            return bool(getattr(mention, "is_primary", False))

        if overall == OverallDiagnosisEnum.POSITIVE or overall == "positive":
            if not pos:
                raise ValueError("overall_diagnosis is 'positive' but no positive_subtypes were provided")
            if primary_pos is None:
                primaries = [_subtype_of(m) for m in pos if _is_primary(m)]
                values["primary_positive_subtype"] = primaries[0] if primaries else _subtype_of(pos[0])
        elif overall == OverallDiagnosisEnum.NEGATIVE or overall == "negative":
            if not neg:
                raise ValueError("overall_diagnosis is 'negative' but no negative_subtypes were provided")
            if primary_neg is None:
                primaries = [_subtype_of(m) for m in neg if _is_primary(m)]
                values["primary_negative_subtype"] = primaries[0] if primaries else _subtype_of(neg[0])
        elif overall == OverallDiagnosisEnum.NEITHER or overall == "neither":
            values["primary_positive_subtype"] = None
            values["primary_negative_subtype"] = None
        return values
