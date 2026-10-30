"""Explicit Pydantic & TypedDict State Schemas for Cynbit AutoBlueprint Multi-Agent Workflow.

Provides `WorkflowStateTypedDict` for native `langgraph.graph.StateGraph(WorkflowStateTypedDict)`
state reducer compatibility alongside Pydantic validation models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class WorkflowStateTypedDict(TypedDict, total=False):
    inquiry_id: str
    inquiry: Dict[str, Any]
    execution_mode: str
    orchestration_backend: str
    validation_passed: bool
    validation_errors: List[str]
    security_flags: List[str]
    research_findings: Optional[Dict[str, Any]]
    rag_context: str
    blueprint_sections: Optional[Dict[str, Any]]
    blueprint_markdown: str
    mermaid_diagram: str
    critic_review: Optional[Dict[str, Any]]
    critic_history: List[Dict[str, Any]]
    revision_count: int
    max_retries: int
    approval_status: str
    reviewer_name: Optional[str]
    reviewer_notes: Optional[str]
    approved_at: Optional[str]
    is_synthetic_demo: bool
    agent_execution_logs: List[Dict[str, Any]]
    processing_time_ms: float


class ProjectInquiryInput(BaseModel):
    inquiry_id: Optional[str] = None
    customer_name: str = Field(..., min_length=2)
    company_name: str = Field(..., min_length=2)
    email: Optional[str] = ""
    industry: str = Field(..., min_length=2)
    project_title: str = Field(..., min_length=3)
    project_description: str = Field(..., min_length=20)
    business_objectives: Optional[str] = ""
    budget_range: Optional[str] = "Not specified"
    expected_timeline: Optional[str] = "Not specified"
    additional_requirements: Optional[str] = ""
    simulate_initial_defect: bool = False


class ResearchFindingsSchema(BaseModel):
    classified_domain: str
    business_problem_summary: str
    extracted_business_objectives: List[str]
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    technical_constraints: List[str]
    missing_information: List[str]
    retrieved_case_studies: List[Dict[str, Any]]
    reusable_cynbit_components: List[Dict[str, Any]]
    rag_status_message: str
    has_verified_references: bool
    execution_mode: str = "OFFLINE_DEMO_MODE"


class CriticReviewSchema(BaseModel):
    overall_quality_score: int
    requirement_coverage_pct: float
    technical_consistency_score: int
    feasibility_assessment: str
    identified_risks: List[str]
    missing_information: List[str]
    recommended_corrections: List[str]
    approval_recommendation: str
    iteration_number: int
    execution_mode: str = "OFFLINE_DEMO_MODE"


class AutoBlueprintState(BaseModel):
    inquiry_id: str
    inquiry: Dict[str, Any]
    execution_mode: str = "OFFLINE_DEMO_MODE"
    orchestration_backend: str = "FallbackStateGraph"
    validation_passed: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    security_flags: List[str] = Field(default_factory=list)
    research_findings: Optional[Dict[str, Any]] = None
    rag_context: str = ""
    blueprint_sections: Optional[Dict[str, Any]] = None
    blueprint_markdown: str = ""
    mermaid_diagram: str = ""
    critic_review: Optional[Dict[str, Any]] = None
    critic_history: List[Dict[str, Any]] = Field(default_factory=list)
    revision_count: int = 0
    max_retries: int = 2
    approval_status: str = "PENDING_APPROVAL"
    reviewer_name: Optional[str] = None
    reviewer_notes: Optional[str] = None
    approved_at: Optional[str] = None
    is_synthetic_demo: bool = False
    agent_execution_logs: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = 0.0
