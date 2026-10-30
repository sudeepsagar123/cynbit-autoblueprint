"""AGENT 1: RESEARCHER AGENT
Supports Live Gemini API structured output generation (`ResearcherLLMOutput` validated via Pydantic)
and clearly labeled deterministic Offline Demo Mode fallback.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from rag.retrieval import CynbitVectorRAGIndex
from services.llm_client import GeminiStructuredClient, ResearcherLLMOutput, model_to_dict

PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "researcher.txt")

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"bypass\s+human\s+approval",
    r"auto[-\s]?approve\s+this\s+blueprint",
    r"reveal\s+api\s+key",
]


def detect_and_sanitize_prompt_injection(text: str) -> Tuple[str, List[str]]:
    """Detects prompt-injection attempts and neutralizes malicious control directives."""
    flags: List[str] = []
    sanitized = text or ""
    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, sanitized, flags=re.IGNORECASE):
            flags.append(f"Blocked potential prompt injection pattern: /{pat}/")
            sanitized = re.sub(pat, "[REDACTED_SECURITY_POLICY]", sanitized, flags=re.IGNORECASE)
    return sanitized.strip(), flags


class ResearcherAgent:
    """Extracts structured requirements (via Gemini API or Offline Fallback) and retrieves RAG references."""

    def __init__(
        self,
        rag_index: CynbitVectorRAGIndex,
        llm_client: Optional[GeminiStructuredClient] = None,
    ) -> None:
        self.rag_index = rag_index
        self.llm_client = llm_client or GeminiStructuredClient()
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        if os.path.isfile(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "You are the Researcher Agent in the Cynbit AutoBlueprint Multi-Agent System."

    def classify_domain(self, text: str, industry: str) -> str:
        combined = f"{industry} {text}".lower()
        if any(w in combined for w in ["patient", "clinical", "medical", "triage", "hospital", "hipaa", "health", "clinic"]):
            return "Healthcare & Clinical AI Systems"
        if any(w in combined for w in ["property", "tenant", "rent", "lease", "building", "unit", "occupancy", "proptech"]):
            return "PropTech & Smart Property Management"
        if any(w in combined for w in ["ticket", "event", "concurrent", "50k", "entertainment", "streaming", "booking"]):
            return "High-Concurrency Enterprise SaaS & Event Platforms"
        if any(w in combined for w in ["payment", "cross-border", "fintech", "ledger", "wallet", "settlement", "banking"]):
            return "FinTech & High-Precision Payment Infrastructure"
        if any(w in combined for w in ["student", "curriculum", "edtech", "coding", "learning", "university", "interview"]):
            return "EdTech & Adaptive Career Intelligence"
        if any(w in combined for w in ["review", "sentiment", "reputation", "branch", "brand", "feedback", "social media"]):
            return "Multi-Location Reputation AI & Customer Experience"
        if any(w in combined for w in ["erp", "legacy", "monolith", "warehouse", "supply chain", "procurement", "migration"]):
            return "Legacy ERP Modernization & Supply Chain Automation"
        return "Enterprise Software & Autonomous AI Automation"

    def extract_requirements_offline(self, inquiry: Dict[str, Any]) -> Dict[str, List[str]]:
        desc = inquiry.get("project_description", "")
        obj_raw = inquiry.get("business_objectives", "")
        add_req = inquiry.get("additional_requirements", "")
        budget = inquiry.get("budget_range", "Not specified")
        timeline = inquiry.get("expected_timeline", "Not specified")
        combined = f"{desc} {obj_raw} {add_req}"
        lower = combined.lower()

        objectives: List[str] = []
        if obj_raw and len(obj_raw.strip()) > 5:
            for part in re.split(r"[.;\n]+", obj_raw):
                if len(part.strip()) > 8:
                    objectives.append(part.strip())
        if not objectives:
            objectives.append(
                f"Automate core operational workflows for {inquiry.get('company_name', 'the client')} "
                f"in the {inquiry.get('industry', 'target')} sector."
            )
            objectives.append("Reduce manual operational bottlenecks while maintaining high system reliability and auditability.")

        functional: List[str] = []
        sentences = [s.strip() for s in re.split(r"[.;\n]+", f"{desc}. {add_req}") if len(s.strip()) > 12]
        for s in sentences:
            functional.append(f"Implement capability: {s}")

        if "triage" in lower or "intake" in lower or "symptom" in lower:
            functional.append("Structured AI symptom/request intake and priority triage routing engine with human clinician/operator review.")
        if "rent" in lower or "tenant" in lower or "property" in lower:
            functional.append("Centralized multi-property unit inventory, digital tenant profiles, payment ledger, and automated delinquency reminders.")
        if "review" in lower or "sentiment" in lower or "reputation" in lower:
            functional.append("Cross-platform review ingestion, aspect-based sentiment classification, draft response generation, and crisis escalation.")
        if "erp" in lower or "legacy" in lower or "migration" in lower:
            functional.append("Modular microservice extraction from legacy workflows with automated document parsing and unified relational ledger.")
        if "payment" in lower or "cross-border" in lower or "fintech" in lower:
            functional.append("Multi-currency transaction routing, idempotent ledger reconciliation, and real-time anomaly/fraud flagging.")
        if "student" in lower or "learning" in lower or "coding" in lower:
            functional.append("Adaptive placement diagnostics, AST/LLM code quality evaluation, and personalized module progression tracking.")
        if "dashboard" in lower or "analytics" in lower or "report" in lower or len(functional) < 3:
            functional.append("Role-based operational analytics dashboard with audit logging and exportable executive reports.")

        functional = list(dict.fromkeys(functional))[:7]

        non_functional: List[str] = [
            "Availability & Resilience: Target 99.9%+ production uptime with automated health checks and graceful degradation.",
            "Security & Governance: Enforce Zero-Trust RBAC, encrypted data in transit (TLS 1.3) and at rest (AES-256), and Human-in-the-Loop (HITL) approval gates for high-stakes actions.",
        ]
        if any(w in lower for w in ["50k", "50,000", "high-concurrency", "real-time", "low latency", "sub-second", "million"]):
            non_functional.append("Scalability & Latency: Support high-concurrency traffic spikes (50k+ simultaneous sessions) with sub-second API P95 response latency.")
        else:
            non_functional.append("Performance: API P95 response time under 500ms for standard transactional queries and <4s for multi-agent synthesis.")
        if any(w in lower for w in ["hipaa", "patient", "clinical", "medical", "health"]):
            non_functional.append("Healthcare Data Privacy: Strict PHI/PII isolation, immutable audit trails, and zero-downtime backup recovery.")
        if any(w in lower for w in ["payment", "fintech", "bank", "rent", "financial", "ledger"]):
            non_functional.append("Financial Integrity: Strict ACID transactional guarantees, idempotency keys, and tamper-evident payment audit logs.")

        constraints: List[str] = []
        if timeline and timeline.lower() not in {"not specified", "none", ""}:
            constraints.append(f"Customer target timeline expectation: {timeline} (must be validated against engineering scope).")
        if budget and budget.lower() not in {"not specified", "none", ""}:
            constraints.append(f"Customer stated budget parameter: {budget} (requires phased MVP scoping).")
        if any(w in lower for w in ["legacy", "existing", "spreadsheets", "whatsapp", "erp", "crm"]):
            constraints.append("Must support data migration or connector integration with existing legacy records/workflows without operational downtime.")
        if not constraints:
            constraints.append("Must deploy on standard cloud infrastructure (AWS/Azure) with containerized CI/CD pipelines and maintainable open-standard frameworks.")

        missing: List[str] = []
        if not timeline or timeline.lower() in {"not specified", "none", ""}:
            missing.append("Target production launch date and milestone deadlines were not specified.")
        if not budget or budget.lower() in {"not specified", "none", ""}:
            missing.append("Target budget ceiling and cloud hosting cost constraints were not provided.")
        if "integration" not in lower and "api" not in lower:
            missing.append("Specific third-party APIs, identity providers (SSO/OAuth), and legacy database schemas need confirmation during Discovery.")
        missing.append("Peak concurrent user volume and regional data-residency compliance jurisdictions require formal verification.")

        return {
            "business_objectives": objectives,
            "functional_requirements": functional,
            "non_functional_requirements": non_functional,
            "technical_constraints": constraints,
            "missing_information": missing,
        }

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        inquiry = dict(state["inquiry"])
        raw_desc = inquiry.get("project_description", "")
        sanitized_desc, sec_flags = detect_and_sanitize_prompt_injection(raw_desc)
        inquiry["project_description"] = sanitized_desc

        # 1. Retrieve verified case studies and SaaS products via RAG
        rag_query = (
            f"{inquiry.get('industry', '')} {inquiry.get('project_title', '')} "
            f"{sanitized_desc} {inquiry.get('business_objectives', '')} {inquiry.get('additional_requirements', '')}"
        )
        cs_rag = self.rag_index.retrieve(rag_query, top_k=3, source_types=["case_study", "saas_product"])
        prod_rag = self.rag_index.retrieve(
            rag_query,
            top_k=2,
            source_types=["saas_product", "service_capability", "engineering_article"],
        )

        # 2. Attempt Live Gemini Structured Output generation if configured
        user_prompt = (
            f"Customer Inquiry:\n"
            f"- Company: {inquiry.get('company_name')}\n"
            f"- Industry: {inquiry.get('industry')}\n"
            f"- Project Title: {inquiry.get('project_title')}\n"
            f"- Description: {sanitized_desc}\n"
            f"- Business Objectives: {inquiry.get('business_objectives')}\n"
            f"- Budget: {inquiry.get('budget_range')}\n"
            f"- Expected Timeline: {inquiry.get('expected_timeline')}\n"
            f"- Additional Requirements: {inquiry.get('additional_requirements')}\n\n"
            f"Retrieved Cynbit Context:\n{cs_rag['grounded_context']}"
        )
        llm_out, exec_mode, llm_err = self.llm_client.generate_structured(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            schema_cls=ResearcherLLMOutput,
        )

        if llm_out is not None and exec_mode == "LIVE_GEMINI_API":
            out_dict = model_to_dict(llm_out)
            domain = out_dict["classified_domain"]
            problem_summary = out_dict["business_problem_summary"]
            objectives = out_dict["extracted_business_objectives"]
            functional = out_dict["functional_requirements"]
            non_functional = out_dict["non_functional_requirements"]
            constraints = out_dict["technical_constraints"]
            missing = out_dict["missing_information"]
        else:
            domain = self.classify_domain(sanitized_desc, inquiry.get("industry", ""))
            extracted = self.extract_requirements_offline(inquiry)
            problem_summary = (
                f"{inquiry.get('company_name', 'The client')} ({inquiry.get('industry', 'Enterprise')}) seeks to solve: "
                f"{sanitized_desc[:280]}{'...' if len(sanitized_desc) > 280 else ''}"
            )
            objectives = extracted["business_objectives"]
            functional = extracted["functional_requirements"]
            non_functional = extracted["non_functional_requirements"]
            constraints = extracted["technical_constraints"]
            missing = extracted["missing_information"]

        findings = {
            "classified_domain": domain,
            "business_problem_summary": problem_summary,
            "extracted_business_objectives": objectives,
            "functional_requirements": functional,
            "non_functional_requirements": non_functional,
            "technical_constraints": constraints,
            "missing_information": missing,
            "retrieved_case_studies": cs_rag["references"],
            "reusable_cynbit_components": prod_rag["references"],
            "rag_status_message": cs_rag["status_message"],
            "rag_embedding_backend": cs_rag.get("embedding_backend", "OFFLINE_HYBRID_TFIDF_SVD"),
            "has_verified_references": bool(cs_rag["has_verified_matches"] or prod_rag["has_verified_matches"]),
            "execution_mode": exec_mode,
        }

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logs = list(state.get("agent_execution_logs", []))
        mode_note = f"[{exec_mode}]" + (f" ({llm_err})" if llm_err and "AFTER" in exec_mode else "")
        logs.append({
            "agent": "Researcher Agent & RAG Retriever",
            "status": "COMPLETED",
            "execution_mode": exec_mode,
            "duration_ms": elapsed_ms,
            "summary": (
                f"{mode_note} Classified domain as '{domain}', extracted {len(functional)} functional "
                f"requirements, and retrieved {len(cs_rag['references'])} verified Cynbit case studies "
                f"via {cs_rag.get('embedding_backend', 'OFFLINE_HYBRID_TFIDF_SVD')}."
            ),
        })

        return {
            "inquiry": inquiry,
            "execution_mode": exec_mode,
            "security_flags": list(state.get("security_flags", [])) + sec_flags,
            "research_findings": findings,
            "rag_context": cs_rag["grounded_context"] + "\n" + prod_rag["grounded_context"],
            "agent_execution_logs": logs,
        }
