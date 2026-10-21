"""LLM backends: optional Amazon Bedrock Converse, plus a deterministic mock."""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from .schema import (
    DiagnosisResponse,
    LikelihoodEnum,
    NegativeSubtypeEnum,
    NegativeSubtypeMention,
    OverallDiagnosisEnum,
    PositiveSubtypeEnum,
    PositiveSubtypeMention,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MALFORMED_SPECIMEN_IDS = {"SYN-MALFORMED-01"}


class LLMBackend:
    def generate_pydantic(
        self,
        query: Union[str, List[Dict[str, str]]],
        response_model: Type[T],
        instructions: Optional[str] = None,
        allow_partial: bool = False,
        fallback: bool = False,
    ) -> Union[T, Dict, str]:
        raise NotImplementedError


class MockBackend(LLMBackend):
    """Deterministic, credential-free backend for the synthetic demo.

    Returns schema-valid DiagnosisResponse objects for ordinary specimen IDs.
    Specimen IDs in MALFORMED_SPECIMEN_IDS yield unparseable text so the
    repair/exclusion path is exercised.
    """

    def __init__(self, **_kwargs):
        pass

    def generate_pydantic(
        self,
        query: Union[str, List[Dict[str, str]]],
        response_model: Type[T],
        instructions: Optional[str] = None,
        allow_partial: bool = False,
        fallback: bool = False,
    ) -> Union[T, Dict, str]:
        specimen_id = instructions or ""
        if specimen_id in MALFORMED_SPECIMEN_IDS:
            return "THIS IS NOT JSON {{{ broken"
        text = query if isinstance(query, str) else json.dumps(query)
        if "Patient report:" in text:
            text = text.split("Patient report:", 1)[1]
        response = _mock_response(text)
        if response_model is DiagnosisResponse:
            return response
        return response_model.model_validate(response.model_dump())


def _first_tissue_site(text: str) -> str:
    if "Tissue Sites:" in text:
        tail = text.split("Tissue Sites:")[-1].strip()
        for line in tail.splitlines():
            line = line.strip()
            if line:
                return line
    return "Skin, left arm, punch biopsy, photo"


def _mock_response(text: str) -> DiagnosisResponse:
    lower = text.lower()
    site = _first_tissue_site(text)

    if "actinic keratosis" in lower:
        return DiagnosisResponse(
            overall_diagnosis=OverallDiagnosisEnum.NEITHER,
            reason="Report describes a process outside CTCL/CLPD and BID.",
            positive_subtypes=[],
            negative_subtypes=[],
        )

    if any(k in lower for k in ("mycosis fungoides", "folliculotropic", "lymphomatoid papulosis", "sezary", "ctcl")):
        subtype = PositiveSubtypeEnum.MF_NON_FOLLICULOTROPIC
        if "folliculotropic" in lower:
            subtype = PositiveSubtypeEnum.MF_FOLLICULOTROPIC
        elif "lymphomatoid papulosis" in lower or "cd30" in lower:
            subtype = PositiveSubtypeEnum.CD30_TCELL_LPD
        likelihood = LikelihoodEnum.LIKELY
        if "diagnosis:" in lower or "consistent with mycosis" in lower:
            likelihood = LikelihoodEnum.DEFINITIVE
        if "cannot exclude" in lower or "suggestive of" in lower:
            likelihood = LikelihoodEnum.POSSIBLE
        return DiagnosisResponse(
            overall_diagnosis=OverallDiagnosisEnum.POSITIVE,
            reason="Favored interpretation is a CTCL/CLPD entity.",
            positive_subtypes=[
                PositiveSubtypeMention(
                    subtype=subtype,
                    likelihood=likelihood,
                    is_primary=True,
                    evidence_text="synthetic",
                    tissue_sites=[site],
                )
            ],
            negative_subtypes=[],
            primary_positive_subtype=subtype,
        )

    subtype = NegativeSubtypeEnum.SPONGIOTIC_DERMATOSIS
    if "lichen" in lower:
        subtype = NegativeSubtypeEnum.LICHENOID_DERMATOSIS
    elif "psoria" in lower:
        subtype = NegativeSubtypeEnum.PSORIASIFORM_DERMATOSIS
    elif "lupus" in lower or "interface" in lower:
        subtype = NegativeSubtypeEnum.INTERFACE_DERMATOSIS
    likelihood = LikelihoodEnum.LIKELY
    if "diagnosis:" in lower:
        likelihood = LikelihoodEnum.DEFINITIVE
    if "cannot exclude" in lower or "suggestive of" in lower:
        likelihood = LikelihoodEnum.POSSIBLE
    return DiagnosisResponse(
        overall_diagnosis=OverallDiagnosisEnum.NEGATIVE,
        reason="Favored interpretation is an inflammatory mimic.",
        positive_subtypes=[],
        negative_subtypes=[
            NegativeSubtypeMention(
                subtype=subtype,
                likelihood=likelihood,
                is_primary=True,
                evidence_text="synthetic",
                tissue_sites=[site],
            )
        ],
        primary_negative_subtype=subtype,
    )


class LLMBedrock(LLMBackend):
    """Amazon Bedrock Converse client with Pydantic JSON repair, as used in the study."""

    def __init__(
        self,
        model_id: str = "meta.llama3-3-70b-instruct-v1:0",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        aws_profile: str = "mskEngineerUser",
        region_name: str = "us-east-2",
        system_prompt: Optional[str] = None,
    ):
        import boto3
        from botocore.exceptions import ClientError  # noqa: F401

        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        try:
            self.session = boto3.Session(profile_name=aws_profile, region_name=region_name)
            self.client = self.session.client(
                service_name="bedrock-runtime",
                region_name=region_name,
            )
        except Exception:
            self.session = boto3.Session(region_name=region_name)
            self.client = self.session.client(
                service_name="bedrock-runtime",
                region_name=region_name,
            )

    def generate_pydantic(
        self,
        query: Union[str, List[Dict[str, str]]],
        response_model: Type[T],
        instructions: Optional[str] = None,
        allow_partial: bool = False,
        fallback: bool = False,
    ) -> Union[T, Dict, str]:
        pydantic_json_schema = response_model.model_json_schema()
        schema_prompt = (
            "Please respond with a valid JSON object that exactly matches this schema:\n\n"
            f"{json.dumps(pydantic_json_schema, indent=2)}\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Use the EXACT field names from the schema\n"
            "- Do not rename or modify any field names\n"
            "- Follow the exact structure specified\n"
            "- Your response must be ONLY valid JSON, no additional text\n"
        )

        system_content_parts = []
        if self.system_prompt:
            system_content_parts.append(self.system_prompt)
        if isinstance(query, list):
            for msg in query:
                if msg.get("role") == "system":
                    system_content_parts.append(msg["content"])
        system_content_parts.append(schema_prompt)
        if instructions:
            system_content_parts.append(f"Additional instructions: {instructions}")

        if isinstance(query, str):
            messages = [{"role": "user", "content": [{"text": query}]}]
        else:
            messages = []
            for msg in query:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})

        system_message = [{"text": "\n\n".join(system_content_parts)}] if system_content_parts else None
        raw_response = self._call_bedrock(messages, system_message, allow_partial, fallback)
        return self._coerce_pydantic(raw_response, response_model, allow_partial, fallback)

    def _call_bedrock(
        self,
        messages: List[Dict],
        system_message: Optional[List[Dict]],
        allow_partial: bool,
        fallback: bool,
    ) -> Union[str, Dict]:
        from botocore.exceptions import ClientError

        inference_config = {"temperature": self.temperature}
        if self.max_tokens:
            inference_config["maxTokens"] = self.max_tokens
        api_params: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if system_message:
            api_params["system"] = system_message
        try:
            response = self.client.converse(**api_params)
            content = response["output"]["message"]["content"][0]["text"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                stripped = content.strip()
                start_idx = stripped.find("{")
                end_idx = stripped.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    json_str = stripped[start_idx : end_idx + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        if allow_partial or fallback:
                            try:
                                obj = ast.literal_eval(json_str)
                                if isinstance(obj, dict):
                                    return obj
                            except Exception:
                                pass
                if fallback or allow_partial:
                    return content
                raise
        except ClientError as exc:
            raise RuntimeError(f"Error calling AWS Bedrock API: {exc}") from exc

    @staticmethod
    def _coerce_pydantic(
        raw_response: Union[str, Dict],
        response_model: Type[T],
        allow_partial: bool,
        fallback: bool,
    ) -> Union[T, Dict, str]:
        if isinstance(raw_response, dict):
            try:
                return response_model.model_validate(raw_response)
            except Exception:
                if allow_partial:
                    model_fields = getattr(response_model, "model_fields", {})
                    partial = {k: v for k, v in raw_response.items() if k in model_fields}
                    return response_model.model_construct(**partial)
                if fallback:
                    return raw_response
                raise
        try:
            parsed = json.loads(raw_response)
            return LLMBedrock._coerce_pydantic(parsed, response_model, allow_partial, fallback)
        except json.JSONDecodeError:
            if allow_partial or fallback:
                try:
                    obj = ast.literal_eval(raw_response)
                    if isinstance(obj, dict):
                        return LLMBedrock._coerce_pydantic(obj, response_model, allow_partial, fallback)
                except Exception:
                    pass
            if fallback:
                return raw_response
            raise ValueError(f"Could not parse response as JSON: {raw_response}")


def create_backend(llm_config) -> LLMBackend:
    backend = (llm_config.backend or "mock").lower()
    if backend == "mock":
        return MockBackend()
    if backend == "bedrock":
        return LLMBedrock(
            model_id=llm_config.model_id,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            aws_profile=llm_config.aws_profile,
            region_name=llm_config.region_name,
        )
    raise ValueError(f"Unknown LLM backend: {backend}")
