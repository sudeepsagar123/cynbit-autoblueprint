"""Automated Tests for Genuine Gemini API Integration, Mocks, Error Handling,
Structured Pydantic Validation, Semantic Embedding Fallback, and LangGraph Orchestration.
"""

import json
import os
import tempfile
import unittest
from typing import Type
from pydantic import BaseModel

from app import seed_initial_records_if_empty
from database.db import AutoBlueprintDatabase
from orchestration.graph import FallbackStateGraph
from orchestration.workflow import CynbitAutoBlueprintWorkflow
from rag.retrieval import CynbitVectorRAGIndex
from services.llm_client import (
    ArchitectLLMOutput,
    CriticLLMOutput,
    GeminiStructuredClient,
    ResearcherLLMOutput,
)


def mock_gemini_transport_success(system_prompt: str, user_prompt: str, schema_cls: Type[BaseModel]) -> str:
    """Simulated Gemini API JSON response matching each agent's Pydantic schema."""
    if schema_cls is ResearcherLLMOutput:
        return json.dumps({
            "classified_domain": "Healthcare & Clinical AI Systems (Gemini Live)",
            "business_problem_summary": "MedCare Hospital requires an AI symptom triage intake portal with clinician HITL review.",
            "extracted_business_objectives": ["Reduce outpatient triage wait times by 45%"],
            "functional_requirements": [
                "Capture patient symptom intake via web/mobile form",
                "Route urgent cases to physician Human-in-the-Loop queue",
            ],
            "non_functional_requirements": [
                "99.9% uptime SLA",
                "Zero-Trust RBAC and HIPAA-aligned encryption",
            ],
            "technical_constraints": ["Must integrate with existing PostgreSQL EHR"],
            "missing_information": ["HL7/FHIR endpoint credentials to be confirmed during Discovery"],
        })
    if schema_cls is ArchitectLLMOutput:
        return json.dumps({
            "project_overview": "Gemini-synthesized 18-section Technical Blueprint for MedCare AI Triage.",
            "customer_business_problem": "Outpatient clinics face manual triage delays.",
            "project_objectives": ["Reduce outpatient triage wait times by 45%"],
            "extracted_functional_requirements": [
                {
                    "req_id": "FR-01",
                    "requirement": "Capture patient symptom intake via web/mobile form",
                    "architectural_mapping": "Next.js Patient Intake Module + FastAPI Triage Service",
                },
                {
                    "req_id": "FR-02",
                    "requirement": "Route urgent cases to physician Human-in-the-Loop queue",
                    "architectural_mapping": "LangGraph HITL Escalation Queue + PostgreSQL Audit Table",
                },
            ],
            "non_functional_requirements": ["99.9% uptime SLA", "Zero-Trust RBAC and HIPAA-aligned encryption"],
            "proposed_solution": "Cloud-native Next.js + FastAPI + LangGraph triage platform with physician HITL approval.",
            "architecture_summary": "4-tier decoupled architecture on AWS with PostgreSQL and Redis.",
            "mermaid_diagram": "graph TD\n  Patient --> NextJS\n  NextJS --> FastAPI\n  FastAPI --> HITL",
            "key_architectural_decisions": ["Use asynchronous Redis task queues for LLM triage calls"],
            "technology_stack": [
                {"layer": "Frontend", "technology": "Next.js & TypeScript", "rationale": "Type-safe responsive clinician UI"},
                {"layer": "Backend", "technology": "Python FastAPI", "rationale": "Async Pydantic-validated AI serving"},
            ],
            "database_design": [
                {
                    "table": "triage_intakes",
                    "engine": "PostgreSQL",
                    "purpose": "Stores patient intake records and HITL status",
                    "key_columns": "id, patient_ref, urgency_tier, hitl_approved_by",
                }
            ],
            "api_and_integration_requirements": [
                {"method": "POST", "endpoint": "/api/v1/triage", "description": "Submit symptom intake"}
            ],
            "implementation_phases": [
                {"phase": "Phase 01: Discovery", "duration_assumption": "Weeks 1-2 (Preliminary Assumption)", "deliverables": "EHR schema audit"}
            ],
            "estimated_duration_range": "10-12 Weeks (Preliminary Engineering Assumption)",
            "assumed_team_composition": "1 Architect, 2 Engineers, 1 QA (Preliminary Assumption)",
            "effort_drivers": "2 core functional modules and EHR integration",
            "infrastructure_requirements": ["AWS ECS Fargate + Multi-AZ RDS PostgreSQL"],
            "security_considerations": [
                "Zero-Trust RBAC with OAuth2/OIDC",
                "TLS 1.3 and AES-256 encryption",
                "Mandatory physician Human-in-the-Loop (HITL) sign-off",
            ],
            "technical_risks_and_mitigation": [
                {"risk": "EHR schema drift", "severity": "Medium", "mitigation": "Pydantic adapter validation"}
            ],
            "key_assumptions": ["Sandbox EHR access provided in Week 1"],
            "missing_information_to_clarify": ["HL7/FHIR endpoint credentials"],
            "recommended_next_steps": ["Human architect review in Approval Center"],
        })
    if schema_cls is CriticLLMOutput:
        return json.dumps({
            "overall_quality_score": 97,
            "requirement_coverage_pct": 100.0,
            "technical_consistency_score": 96,
            "feasibility_assessment": "Verified feasible and grounded in Cynbit stack.",
            "identified_risks": ["Confirm FHIR rate limits during Phase 01"],
            "missing_information": ["HL7/FHIR endpoint credentials"],
            "recommended_corrections": [],
            "approval_recommendation": "READY_FOR_HUMAN_REVIEW",
        })
    raise ValueError(f"Unexpected schema class: {schema_cls}")


class TestLLMIntegrationAndUpgrades(unittest.TestCase):
    def test_01_live_gemini_structured_output_end_to_end_with_mock(self) -> None:
        mock_client = GeminiStructuredClient(
            api_key="test-mock-gemini-key",
            model_name="gemini-2.5-flash",
            transport_override=mock_gemini_transport_success,
        )
        self.assertTrue(mock_client.is_live_configured())
        self.assertEqual(mock_client.get_execution_mode_label(), "LIVE_GEMINI_API")

        workflow = CynbitAutoBlueprintWorkflow(max_retries=2, llm_client=mock_client)
        inquiry = {
            "customer_name": "Dr. Kavita Rao",
            "company_name": "MedCare Hospital",
            "industry": "Healthcare",
            "project_title": "AI Symptom Triage Portal",
            "project_description": "We need an AI symptom triage intake portal with physician HITL review.",
            "business_objectives": "Reduce wait times by 45%.",
        }
        state = workflow.execute_inquiry(inquiry)
        self.assertEqual(state["execution_mode"], "LIVE_GEMINI_API")
        self.assertEqual(state["research_findings"]["execution_mode"], "LIVE_GEMINI_API")
        self.assertEqual(state["critic_review"]["execution_mode"], "LIVE_GEMINI_API")
        self.assertEqual(state["critic_review"]["overall_quality_score"], 97)
        self.assertEqual(state["approval_status"], "PENDING_APPROVAL")

    def test_02_gemini_api_failure_falls_back_to_offline_demo_mode(self) -> None:
        def failing_transport(sys_p: str, usr_p: str, schema_cls: Type[BaseModel]) -> str:
            raise RuntimeError("Simulated HTTP 503 Service Unavailable from Gemini endpoint")

        failing_client = GeminiStructuredClient(
            api_key="test-failing-key",
            transport_override=failing_transport,
        )
        workflow = CynbitAutoBlueprintWorkflow(max_retries=2, llm_client=failing_client)
        inquiry = {
            "customer_name": "Arjun Mehta",
            "company_name": "LogiSync ERP",
            "industry": "Supply Chain",
            "project_title": "Legacy ERP Cloud Migration",
            "project_description": "Migrate our monolithic warehouse ERP to distributed microservices with PostgreSQL.",
        }
        state = workflow.execute_inquiry(inquiry)
        self.assertTrue(state["validation_passed"])
        self.assertEqual(state["research_findings"]["execution_mode"], "OFFLINE_FALLBACK_AFTER_API_ERROR")
        self.assertEqual(len(state["blueprint_sections"]), 18)
        self.assertEqual(state["approval_status"], "PENDING_APPROVAL")

    def test_03_malformed_json_and_pydantic_validation_error_fallback(self) -> None:
        def invalid_json_transport(sys_p: str, usr_p: str, schema_cls: Type[BaseModel]) -> str:
            return '{"classified_domain": "Only one field, missing all required lists"}'

        bad_schema_client = GeminiStructuredClient(
            api_key="test-bad-schema-key",
            transport_override=invalid_json_transport,
        )
        out, mode, err = bad_schema_client.generate_structured("sys", "user", ResearcherLLMOutput)
        self.assertIsNone(out)
        self.assertEqual(mode, "OFFLINE_FALLBACK_AFTER_VALIDATION_ERROR")
        self.assertIn("validation failed", (err or "").lower())

    def test_04_semantic_embedding_router_mock_and_fallback(self) -> None:
        # 1. Test live semantic embedding callback
        def fake_gemini_embed(texts):
            # Deterministic 8-dim vector based on text keywords
            vecs = []
            for t in texts:
                tl = t.lower()
                v = [
                    1.0 if ("triage" in tl or "medical" in tl or "wizhealth" in tl) else 0.1,
                    1.0 if ("ticket" in tl or "evento" in tl or "50k" in tl) else 0.1,
                    1.0 if ("property" in tl or "tenantease" in tl or "rent" in tl) else 0.1,
                    0.2, 0.3, 0.1, 0.4, 0.2,
                ]
                vecs.append(v)
            return vecs

        rag_live = CynbitVectorRAGIndex(remote_embed_fn=fake_gemini_embed)
        self.assertEqual(rag_live.embedder.active_backend, "GEMINI_TEXT_EMBEDDING_004")
        res = rag_live.retrieve("medical triage system for clinic patients", top_k=2, source_types=["case_study"])
        self.assertTrue(res["has_verified_matches"])
        self.assertEqual(res["embedding_backend"], "GEMINI_TEXT_EMBEDDING_004")
        self.assertEqual(res["references"][0]["doc_id"], "CS-WIZHEALTH")

        # 2. Test embedding API error graceful fallback to offline engine
        def failing_embed(texts):
            raise ConnectionError("Embedding endpoint unreachable")

        rag_fallback = CynbitVectorRAGIndex(remote_embed_fn=failing_embed)
        self.assertEqual(rag_fallback.embedder.active_backend, "OFFLINE_HYBRID_TFIDF_SVD")
        res_fb = rag_fallback.retrieve("Scaling real-time ticketing for 50k users", top_k=2, source_types=["case_study"])
        self.assertTrue(res_fb["has_verified_matches"])
        self.assertEqual(res_fb["references"][0]["doc_id"], "CS-EVENTO")

    def test_05_langgraph_state_graph_wiring_contract(self) -> None:
        recorded_nodes = []

        class SpyLangGraph(FallbackStateGraph):
            def add_node(self, name, action):
                recorded_nodes.append(name)
                super().add_node(name, action)

        wf = CynbitAutoBlueprintWorkflow(graph_cls_override=SpyLangGraph)
        self.assertEqual(wf.orchestration_backend, "langgraph.StateGraph")
        self.assertEqual(
            recorded_nodes,
            ["input_validation", "researcher_agent", "architect_agent", "critic_agent", "human_approval_gate"],
        )

    def test_06_seeded_records_use_only_synthetic_demo_reviewers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = AutoBlueprintDatabase(os.path.join(tmp, "seed_test.db"))
            wf = CynbitAutoBlueprintWorkflow()
            seed_initial_records_if_empty(db, wf)
            records = db.list_blueprints()
            self.assertEqual(len(records), 4)
            real_leadership_names = ["mohak saini", "ashutosh upadhyay", "dhawan nama"]
            for r in records:
                rev = (r.get("reviewer_name") or "").lower()
                for leader in real_leadership_names:
                    self.assertNotIn(leader, rev)
                if rev:
                    self.assertIn("synthetic demo reviewer", rev)


if __name__ == "__main__":
    unittest.main()
