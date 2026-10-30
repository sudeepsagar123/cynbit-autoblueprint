"""Genuine Gemini API Integration with Structured Outputs & Strict Pydantic Validation.

Supports:
1. Official `google-genai` SDK (`from google import genai`)
2. Official `google-generativeai` SDK (`import google.generativeai as genai`)
3. Direct Gemini v1beta REST API (`https://generativelanguage.googleapis.com/v1beta/models/...:generateContent`)
   via standard library `urllib.request`, enabling live Gemini structured output generation even in minimal
   Python environments when `GEMINI_API_KEY` is set.
4. Strict Pydantic validation (`validate_pydantic_model`) on every LLM response.
5. Transparent fallback to clearly labeled `OFFLINE_DEMO_MODE` when `GEMINI_API_KEY` is not set or when
   network/API failures occur.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError

T = TypeVar("T", bound=BaseModel)


# --- Structured Output Pydantic Schemas for the 3 Agents ---

class ResearcherLLMOutput(BaseModel):
    classified_domain: str = Field(..., min_length=3)
    business_problem_summary: str = Field(..., min_length=10)
    extracted_business_objectives: List[str] = Field(..., min_length=1)
    functional_requirements: List[str] = Field(..., min_length=1)
    non_functional_requirements: List[str] = Field(..., min_length=1)
    technical_constraints: List[str] = Field(..., min_length=1)
    missing_information: List[str] = Field(..., min_length=1)


class FunctionalReqItem(BaseModel):
    req_id: str
    requirement: str
    architectural_mapping: str


class TechStackItem(BaseModel):
    layer: str
    technology: str
    rationale: str


class DatabaseTableItem(BaseModel):
    table: str
    engine: str
    purpose: str
    key_columns: str


class ApiEndpointItem(BaseModel):
    method: str
    endpoint: str
    description: str


class ImplementationPhaseItem(BaseModel):
    phase: str
    duration_assumption: str
    deliverables: str


class RiskMitigationItem(BaseModel):
    risk: str
    severity: str
    mitigation: str


class ArchitectLLMOutput(BaseModel):
    project_overview: str = Field(..., min_length=15)
    customer_business_problem: str = Field(..., min_length=10)
    project_objectives: List[str] = Field(..., min_length=1)
    extracted_functional_requirements: List[FunctionalReqItem] = Field(..., min_length=1)
    non_functional_requirements: List[str] = Field(..., min_length=1)
    proposed_solution: str = Field(..., min_length=15)
    architecture_summary: str = Field(..., min_length=15)
    mermaid_diagram: str = Field(..., min_length=15)
    key_architectural_decisions: List[str] = Field(..., min_length=1)
    technology_stack: List[TechStackItem] = Field(..., min_length=1)
    database_design: List[DatabaseTableItem] = Field(..., min_length=1)
    api_and_integration_requirements: List[ApiEndpointItem] = Field(..., min_length=1)
    implementation_phases: List[ImplementationPhaseItem] = Field(..., min_length=1)
    estimated_duration_range: str = Field(..., min_length=5)
    assumed_team_composition: str = Field(..., min_length=5)
    effort_drivers: str = Field(..., min_length=5)
    infrastructure_requirements: List[str] = Field(..., min_length=1)
    security_considerations: List[str] = Field(..., min_length=1)
    technical_risks_and_mitigation: List[RiskMitigationItem] = Field(..., min_length=1)
    key_assumptions: List[str] = Field(..., min_length=1)
    missing_information_to_clarify: List[str] = Field(..., min_length=1)
    recommended_next_steps: List[str] = Field(..., min_length=1)


class CriticLLMOutput(BaseModel):
    overall_quality_score: int = Field(..., ge=0, le=100)
    requirement_coverage_pct: float = Field(..., ge=0.0, le=100.0)
    technical_consistency_score: int = Field(..., ge=0, le=100)
    feasibility_assessment: str = Field(..., min_length=10)
    identified_risks: List[str]
    missing_information: List[str]
    recommended_corrections: List[str]
    approval_recommendation: str = Field(..., pattern=r"^(READY_FOR_HUMAN_REVIEW|NEEDS_REVISION)$")


def validate_pydantic_json(schema_cls: Type[T], raw_payload: str | Dict[str, Any]) -> T:
    """Validates a JSON string or dict against a Pydantic BaseModel (supports Pydantic v1 and v2)."""
    if isinstance(raw_payload, str):
        cleaned = raw_payload.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        data = json.loads(cleaned)
    else:
        data = raw_payload

    if hasattr(schema_cls, "model_validate"):
        return schema_cls.model_validate(data)
    return schema_cls.parse_obj(data)


def model_to_dict(instance: BaseModel) -> Dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump()
    return instance.dict()


class GeminiStructuredClient:
    """Client for Gemini structured JSON generation with Pydantic validation and offline fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_sec: float = 20.0,
        transport_override: Optional[Callable[[str, str, Type[BaseModel]], str]] = None,
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")
        ).strip()
        if self.api_key == "your_gemini_api_key_here":
            self.api_key = ""
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.timeout_sec = timeout_sec
        self.transport_override = transport_override

    def is_live_configured(self) -> bool:
        return bool(self.api_key) or (self.transport_override is not None)

    def get_execution_mode_label(self) -> str:
        return "LIVE_GEMINI_API" if self.is_live_configured() else "OFFLINE_DEMO_MODE"

    def _call_gemini_api(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: Type[T],
    ) -> str:
        """Invokes Gemini using transport_override (for tests), google-genai SDK, or REST API."""
        if self.transport_override is not None:
            return self.transport_override(system_prompt, user_prompt, schema_cls)

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured in environment.")

        # 1. Try official google-genai SDK if available
        try:
            from google import genai  # type: ignore
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "system_instruction": system_prompt,
                    "response_mime_type": "application/json",
                    "response_schema": schema_cls,
                    "temperature": 0.1,
                },
            )
            if response and getattr(response, "text", None):
                return response.text
        except ImportError:
            pass

        # 2. Try google-generativeai SDK if available
        try:
            import google.generativeai as legacy_genai  # type: ignore
            legacy_genai.configure(api_key=self.api_key)
            model = legacy_genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                user_prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            if response and getattr(response, "text", None):
                return response.text
        except ImportError:
            pass

        # 3. Direct Gemini v1beta REST API call via urllib.request
        schema_json = (
            schema_cls.model_json_schema()
            if hasattr(schema_cls, "model_json_schema")
            else schema_cls.schema()
        )
        schema_prompt = (
            f"{user_prompt}\n\nRespond strictly with valid JSON matching this schema:\n"
            f"{json.dumps(schema_json, indent=2)}"
        )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": schema_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {body}")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or "text" not in parts[0]:
            raise RuntimeError(f"Gemini API candidate missing text part: {candidates[0]}")
        return parts[0]["text"]

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_cls: Type[T],
    ) -> Tuple[Optional[T], str, Optional[str]]:
        """Generates and validates a structured Pydantic output from Gemini.

        Returns:
            (validated_model_or_None, execution_mode_str, error_message_or_None)
        """
        if not self.is_live_configured():
            return None, "OFFLINE_DEMO_MODE", "GEMINI_API_KEY not configured; using deterministic offline engine."

        try:
            raw_text = self._call_gemini_api(system_prompt, user_prompt, schema_cls)
            validated = validate_pydantic_json(schema_cls, raw_text)
            return validated, "LIVE_GEMINI_API", None
        except (ValidationError, json.JSONDecodeError) as val_err:
            return None, "OFFLINE_FALLBACK_AFTER_VALIDATION_ERROR", f"Structured output validation failed: {val_err}"
        except Exception as api_err:
            return None, "OFFLINE_FALLBACK_AFTER_API_ERROR", f"Gemini API call failed: {api_err}"
