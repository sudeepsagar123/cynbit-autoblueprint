"""AGENT 3: CRITIC AGENT
Reviews the generated technical blueprint before human approval.
Combines deterministic invariant checks (requirement coverage, 18-section completeness,
preliminary estimate disclaimers, verified cynbit.in URLs) with optional Live Gemini API
structured critique (`CriticLLMOutput` validated via Pydantic).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from services.llm_client import CriticLLMOutput, GeminiStructuredClient, model_to_dict

PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "critic.txt")


class CriticAgent:
    """Audits the 18-section blueprint and determines whether revision is required."""

    REQUIRED_SECTION_KEYS = [
        "1_project_overview",
        "2_customer_business_problem",
        "3_project_objectives",
        "4_extracted_functional_requirements",
        "5_non_functional_requirements",
        "6_proposed_solution",
        "7_recommended_system_architecture",
        "8_technology_stack",
        "9_database_design",
        "10_api_and_integration_requirements",
        "11_relevant_cynbit_case_studies",
        "12_implementation_phases",
        "13_preliminary_timeline_and_effort_estimates",
        "14_infrastructure_requirements",
        "15_security_considerations",
        "16_technical_risks_and_mitigation",
        "17_assumptions_and_missing_information",
        "18_recommended_next_steps",
    ]

    def __init__(self, llm_client: Optional[GeminiStructuredClient] = None) -> None:
        self.llm_client = llm_client or GeminiStructuredClient()
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        if os.path.isfile(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "You are the Principal Quality & Security Critic Agent at Cynbit Technologies."

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        findings = state["research_findings"]
        sections = state.get("blueprint_sections") or {}
        revision_count = int(state.get("revision_count", 0))
        max_retries = int(state.get("max_retries", 2))

        expected_reqs = findings.get("functional_requirements", [])
        mapped_reqs = sections.get("4_extracted_functional_requirements", [])
        coverage_pct = round(
            (len(mapped_reqs) / max(1, len(expected_reqs))) * 100.0, 1
        )

        missing_sections = [k for k in self.REQUIRED_SECTION_KEYS if k not in sections or not sections[k]]
        security_items = sections.get("15_security_considerations", [])
        estimates = sections.get("13_preliminary_timeline_and_effort_estimates", {})
        disclaimer_text = estimates.get("disclaimer", "")
        case_studies = sections.get("11_relevant_cynbit_case_studies", [])

        identified_risks: List[str] = []
        recommended_corrections: List[str] = []

        # Deterministic invariant checks (always enforced even in Live Gemini mode)
        if coverage_pct < 100.0:
            unmapped_count = len(expected_reqs) - len(mapped_reqs)
            identified_risks.append(
                f"Incomplete Requirement Coverage: {unmapped_count} extracted functional requirement(s) were omitted from Section 4."
            )
            recommended_corrections.append(
                "Map 100% of extracted functional requirements in Section 4 (Extracted Functional Requirements)."
            )

        if missing_sections:
            identified_risks.append(f"Missing required blueprint sections: {', '.join(missing_sections)}")
            recommended_corrections.append(f"Populate missing sections: {', '.join(missing_sections)}")

        if len(security_items) < 3 or any("Incomplete draft" in s for s in security_items):
            identified_risks.append(
                "Insufficient Security & Governance Controls: Section 15 lacks Zero-Trust RBAC, encryption at rest, or HITL sandbox audit controls."
            )
            recommended_corrections.append(
                "Expand Section 15 to include Zero-Trust RBAC, TLS 1.3 / AES-256 encryption, HITL approval gates, and immutable audit logging."
            )

        if "preliminary" not in disclaimer_text.lower() or "assumption" not in disclaimer_text.lower():
            identified_risks.append(
                "Unqualified Effort Estimate: Section 13 does not explicitly label timeline/effort estimates as preliminary assumptions."
            )
            recommended_corrections.append(
                "Explicitly label all timeline and effort estimates in Section 13 as Preliminary Engineering Assumptions."
            )

        for cs in case_studies:
            if not str(cs.get("url", "")).startswith("https://www.cynbit.in"):
                identified_risks.append(f"Unverified Case Study URL detected: {cs.get('url')}")
                recommended_corrections.append("Remove unverified URLs and cite only verified https://www.cynbit.in references.")

        # Optional Live Gemini Critique enrichment
        llm_out = None
        exec_mode = "OFFLINE_DEMO_MODE"
        llm_err = None
        if self.llm_client.is_live_configured() and not recommended_corrections:
            user_prompt = (
                f"Audit this Technical Blueprint against the Researcher Findings.\n"
                f"Researcher Findings: {json.dumps(findings, indent=2)}\n\n"
                f"Blueprint Sections: {json.dumps(sections, indent=2)}"
            )
            llm_out, exec_mode, llm_err = self.llm_client.generate_structured(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                schema_cls=CriticLLMOutput,
            )

        if llm_out is not None and exec_mode == "LIVE_GEMINI_API" and not recommended_corrections:
            c_dict = model_to_dict(llm_out)
            overall_score = c_dict["overall_quality_score"]
            technical_consistency = c_dict["technical_consistency_score"]
            feasibility = c_dict["feasibility_assessment"]
            identified_risks = c_dict["identified_risks"]
            recommended_corrections = c_dict["recommended_corrections"]
            recommendation = (
                "NEEDS_REVISION"
                if (c_dict["approval_recommendation"] == "NEEDS_REVISION" and revision_count < max_retries)
                else "READY_FOR_HUMAN_REVIEW"
            )
            next_revision_count = revision_count + 1 if recommendation == "NEEDS_REVISION" else revision_count
        else:
            penalty = len(recommended_corrections) * 14
            technical_consistency = max(60, 98 - penalty)
            overall_score = int(round(0.5 * coverage_pct + 0.5 * technical_consistency))

            if recommended_corrections and revision_count < max_retries:
                recommendation = "NEEDS_REVISION"
                feasibility = "Requires Architect revision pass to resolve requirement coverage and security gaps before human review."
                next_revision_count = revision_count + 1
            else:
                recommendation = "READY_FOR_HUMAN_REVIEW"
                feasibility = (
                    "Feasible & Well-Grounded: Architecture aligns with Cynbit's production stack, "
                    "all functional requirements are mapped, estimates are properly qualified as preliminary "
                    "assumptions, and HITL security guardrails are verified."
                )
                if not identified_risks:
                    identified_risks = [
                        "Standard Discovery Dependency: Third-party API rate limits and legacy schema details require confirmation during Phase 01."
                    ]
                next_revision_count = revision_count

        review = {
            "overall_quality_score": overall_score,
            "requirement_coverage_pct": coverage_pct,
            "technical_consistency_score": technical_consistency,
            "feasibility_assessment": feasibility,
            "identified_risks": identified_risks,
            "missing_information": findings.get("missing_information", []),
            "recommended_corrections": recommended_corrections,
            "approval_recommendation": recommendation,
            "iteration_number": revision_count + 1,
            "execution_mode": exec_mode,
        }

        history = list(state.get("critic_history", []))
        history.append(review)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logs = list(state.get("agent_execution_logs", []))
        mode_note = f"[{exec_mode}]" + (f" ({llm_err})" if llm_err and "AFTER" in exec_mode else "")
        logs.append({
            "agent": f"Critic Agent (Audit Pass #{revision_count + 1})",
            "status": "REVISION_TRIGGERED" if recommendation == "NEEDS_REVISION" else "PASSED_TO_HITL",
            "execution_mode": exec_mode,
            "duration_ms": elapsed_ms,
            "summary": (
                f"{mode_note} Quality Score: {overall_score}/100 | Requirement Coverage: {coverage_pct}% | "
                f"Recommendation: {recommendation}."
            ),
        })

        return {
            "critic_review": review,
            "critic_history": history,
            "revision_count": next_revision_count,
            "agent_execution_logs": logs,
        }
