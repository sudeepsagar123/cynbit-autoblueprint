"""AGENT 2: ARCHITECT AGENT
Transforms Researcher findings and RAG evidence into a complete 18-section Technical Blueprint.
Supports Live Gemini API structured output generation (`ArchitectLLMOutput` validated via Pydantic)
and clearly labeled deterministic Offline Demo Mode fallback.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from services.llm_client import ArchitectLLMOutput, GeminiStructuredClient, model_to_dict

PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "architect.txt")


class ArchitectAgent:
    """Synthesizes a domain-specific 18-section Technical Blueprint (via Gemini API or Offline Fallback)."""

    def __init__(self, llm_client: Optional[GeminiStructuredClient] = None) -> None:
        self.llm_client = llm_client or GeminiStructuredClient()
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        if os.path.isfile(PROMPT_FILE):
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "You are the Lead Technical Architect Agent at Cynbit Technologies."

    def _select_stack(self, domain: str, revision_notes: List[str]) -> List[Dict[str, str]]:
        """Selects technologies from Cynbit's verified stack on https://www.cynbit.in/services."""
        stack = [
            {
                "layer": "Frontend & Client Portal",
                "technology": "TypeScript, Next.js (React) & Tailwind CSS",
                "rationale": (
                    "Listed in Cynbit's Services Hub (https://www.cynbit.in/services); provides SSR/ISR "
                    "performance, strict type safety, and responsive role-based operational dashboards."
                ),
            },
            {
                "layer": "Core Backend & API Gateway",
                "technology": "Python (FastAPI) & NestJS (TypeScript)",
                "rationale": (
                    "FastAPI provides async, Pydantic-validated AI/ML model serving and agent orchestration, "
                    "while NestJS handles modular enterprise RBAC and webhook integrations."
                ),
            },
        ]

        if "High-Concurrency" in domain or "FinTech" in domain:
            stack.append({
                "layer": "High-Throughput Event / Payment Core",
                "technology": "Go & Rust Microservices with Redis Streams",
                "rationale": (
                    "Part of Cynbit's published language stack on https://www.cynbit.in/services for "
                    "low-latency, high-concurrency event processing without garbage-collection pauses."
                ),
            })

        stack.extend([
            {
                "layer": "AI / Multi-Agent & RAG Layer",
                "technology": "LangGraph / LangChain, Gemini / OpenAI API & Pinecone / Hybrid Vector Index",
                "rationale": (
                    "Enables stateful multi-agent orchestration with bounded retries, semantic retrieval over "
                    "domain documents, and Human-in-the-Loop (HITL) sandbox governance."
                ),
            },
            {
                "layer": "Primary Transactional & Caching Data Stores",
                "technology": "PostgreSQL (Prisma ORM) & Redis",
                "rationale": (
                    "PostgreSQL ensures ACID compliance, relational integrity, and row-level multi-tenant security; "
                    "Redis provides sub-millisecond caching, rate limiting, and distributed task queues."
                ),
            },
            {
                "layer": "Cloud Infrastructure & DevSecOps",
                "technology": "AWS / Azure Zero-Trust, Docker, Kubernetes & Terraform CI/CD",
                "rationale": (
                    "Delivers reproducible Infrastructure-as-Code, automated container scaling, VPC isolation, "
                    "and zero-downtime blue/green deployments (99.9% uptime target)."
                ),
            },
        ])

        if revision_notes:
            stack.append({
                "layer": "Critic-Mandated Hardening & Observability",
                "technology": "OpenTelemetry, Tamper-Evident Audit Log & Dead-Letter Queue (DLQ)",
                "rationale": (
                    "Added during Architect revision pass to satisfy Critic Agent audit requirements for "
                    "end-to-end traceability, failure recovery, and compliance verification."
                ),
            })

        return stack

    def _build_mermaid_diagram(self, company: str) -> str:
        return f"""graph TD
    Client["{company} Users & Operators"] -->|HTTPS / TLS 1.3| CDN["Cloudflare / AWS WAF & API Gateway"]
    CDN --> UI["Next.js + TypeScript Web Portal"]
    UI --> API["FastAPI / NestJS Application Core"]
    API --> Auth["Zero-Trust RBAC & OAuth2/OIDC"]
    API --> Orch["LangGraph Multi-Agent Orchestrator"]
    Orch --> ResAgent["Researcher & Extraction Node"]
    Orch --> RAG["Vector RAG Index (Pinecone / Semantic Embeddings)"]
    Orch --> ArchNode["Domain Logic & Scoring Engine"]
    Orch --> CriticNode["Automated Policy & Guardrail Verifier"]
    CriticNode -->|High-Stakes Action| HITL["Human-in-the-Loop (HITL) Approval Queue"]
    HITL -->|Approved| DB[("PostgreSQL Primary + Read Replica")]
    API --> Cache[("Redis Cache & Task Queue")]
    API --> Audit[("Immutable Compliance Audit Log")]"""

    def _build_case_study_items(self, findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        case_studies = findings.get("retrieved_case_studies", [])
        if case_studies:
            return [
                {
                    "doc_id": cs["doc_id"],
                    "title": cs["title"],
                    "client": cs["client_or_entity"],
                    "similarity": cs["similarity_score"],
                    "verified_metric": cs["verified_metric"],
                    "url": cs["url"],
                    "relevance_note": (
                        f"Verified public reference on {cs['url']} ({cs['category']}: {cs['title']})."
                    ),
                }
                for cs in case_studies
            ]
        return [{
            "doc_id": "NONE",
            "title": "No Verified Case Study Above Similarity Threshold",
            "client": "N/A",
            "similarity": 0.0,
            "verified_metric": "N/A",
            "url": "https://www.cynbit.in/case-studies",
            "relevance_note": (
                "No verified reference was found in the public Cynbit knowledge base matching this exact query. "
                "Architecture below is a first-principles engineering proposal."
            ),
        }]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        inquiry = state["inquiry"]
        findings = state["research_findings"]
        revision_count = int(state.get("revision_count", 0))
        prev_critic = state.get("critic_review")

        simulate_defect = bool(inquiry.get("simulate_initial_defect", False)) and (revision_count == 0)
        corrections_applied: List[str] = []
        if prev_critic and prev_critic.get("recommended_corrections"):
            corrections_applied = list(prev_critic["recommended_corrections"])

        cs_section_items = self._build_case_study_items(findings)

        # Attempt Live Gemini Structured Output generation when configured and not simulating a deterministic defect
        llm_out = None
        exec_mode = "OFFLINE_DEMO_MODE"
        llm_err = None
        if not simulate_defect and self.llm_client.is_live_configured():
            user_prompt = (
                f"Generate a complete 18-section Technical Blueprint for:\n"
                f"Inquiry: {json.dumps(inquiry, indent=2)}\n\n"
                f"Researcher Findings: {json.dumps(findings, indent=2)}\n\n"
                f"Critic Revision Corrections to Apply: {json.dumps(corrections_applied)}"
            )
            llm_out, exec_mode, llm_err = self.llm_client.generate_structured(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                schema_cls=ArchitectLLMOutput,
            )

        if llm_out is not None and exec_mode == "LIVE_GEMINI_API":
            d = model_to_dict(llm_out)
            disclaimer_text = (
                "PRELIMINARY ASSUMPTION NOTICE: All timeline and effort figures below are non-binding preliminary "
                "engineering assumptions generated for discovery scoping. Final delivery commitments, pricing, "
                "and resource allocation require formal Senior Architect scoping with the customer."
            )
            est_range = d["estimated_duration_range"]
            if "preliminary" not in est_range.lower():
                est_range = f"{est_range} (Preliminary Engineering Assumption)"
            sections: Dict[str, Any] = {
                "1_project_overview": d["project_overview"],
                "2_customer_business_problem": d["customer_business_problem"],
                "3_project_objectives": d["project_objectives"],
                "4_extracted_functional_requirements": d["extracted_functional_requirements"],
                "5_non_functional_requirements": d["non_functional_requirements"],
                "6_proposed_solution": d["proposed_solution"],
                "7_recommended_system_architecture": {
                    "summary": d["architecture_summary"],
                    "mermaid_diagram": d["mermaid_diagram"],
                    "key_architectural_decisions": d["key_architectural_decisions"],
                },
                "8_technology_stack": d["technology_stack"],
                "9_database_design": d["database_design"],
                "10_api_and_integration_requirements": d["api_and_integration_requirements"],
                "11_relevant_cynbit_case_studies": cs_section_items,
                "12_implementation_phases": d["implementation_phases"],
                "13_preliminary_timeline_and_effort_estimates": {
                    "disclaimer": disclaimer_text,
                    "estimated_duration_range": est_range,
                    "assumed_team_composition": d["assumed_team_composition"],
                    "effort_drivers": d["effort_drivers"],
                },
                "14_infrastructure_requirements": d["infrastructure_requirements"],
                "15_security_considerations": d["security_considerations"],
                "16_technical_risks_and_mitigation": d["technical_risks_and_mitigation"],
                "17_assumptions_and_missing_information": {
                    "key_assumptions": d["key_assumptions"],
                    "missing_information_to_clarify": d["missing_information_to_clarify"],
                },
                "18_recommended_next_steps": d["recommended_next_steps"],
            }
            mermaid_code = d["mermaid_diagram"]
        else:
            domain = findings["classified_domain"]
            company = inquiry.get("company_name", "Client Organization")
            title = inquiry.get("project_title", "Enterprise AI & Software Platform")

            func_reqs = list(findings["functional_requirements"])
            addressed_func_reqs = func_reqs[:-1] if (simulate_defect and len(func_reqs) > 1) else func_reqs
            non_func_reqs = list(findings["non_functional_requirements"])
            tech_stack = self._select_stack(domain, corrections_applied)
            mermaid_code = self._build_mermaid_diagram(company)

            base_weeks = max(8, min(20, 6 + len(func_reqs) * 2))
            est_range = f"{base_weeks}–{base_weeks + 4} Weeks (Preliminary Engineering Assumption)"
            reusable = findings.get("reusable_cynbit_components", [])

            security_items = (
                ["Basic HTTPS transport encryption (Incomplete draft — missing RBAC & HITL audit controls)."]
                if simulate_defect
                else [
                    "Zero-Trust Role-Based Access Control (RBAC) with OAuth2/OIDC and short-lived JWT session tokens.",
                    "Encryption in transit (TLS 1.3) and at rest (AES-256) across PostgreSQL volumes and S3 object buckets.",
                    "Human-in-the-Loop (HITL) Sandbox Gate: Autonomous AI agents draft recommendations and stage actions, while high-stakes mutations require explicit human operator sign-off.",
                    "Input Sanitization & Prompt-Injection Firewall: Strict Pydantic schema validation, token ceiling enforcement, and system-prompt isolation.",
                    "Immutable Audit Logging: Every agent invocation, retrieval source, and human approval action is logged with actor ID and UTC timestamp.",
                ]
            )

            sections = {
                "1_project_overview": (
                    f"This Technical Blueprint defines the proposed architecture, technology stack, data model, and phased "
                    f"implementation plan for '{title}' requested by {company} ({inquiry.get('customer_name', 'Client')}). "
                    f"Classified under '{domain}', the solution aligns with Cynbit Technologies' 6-stage architectural "
                    f"lifecycle and Human-in-the-Loop (HITL) engineering governance."
                ),
                "2_customer_business_problem": findings["business_problem_summary"],
                "3_project_objectives": findings["extracted_business_objectives"],
                "4_extracted_functional_requirements": [
                    {
                        "req_id": f"FR-{i:02d}",
                        "requirement": req,
                        "architectural_mapping": f"Mapped to Core Application Service & Workflow Module #{i}",
                    }
                    for i, req in enumerate(addressed_func_reqs, start=1)
                ],
                "5_non_functional_requirements": non_func_reqs,
                "6_proposed_solution": (
                    f"We propose a modular, cloud-native platform for {company} combining a responsive Next.js operator "
                    f"portal, a FastAPI/NestJS service layer, a grounded RAG + domain intelligence engine, and a "
                    f"PostgreSQL + Redis data backbone. Where applicable, reusable architectural patterns from "
                    f"{', '.join(c['client_or_entity'] for c in reusable) if reusable else 'Cynbit Core Services'} "
                    f"accelerate foundational delivery while preserving strict tenant isolation."
                ),
                "7_recommended_system_architecture": {
                    "summary": (
                        "Decoupled 4-tier architecture: (1) Edge WAF & Next.js Presentation Tier, (2) FastAPI/NestJS "
                        "Transactional & API Tier, (3) LangGraph Agentic & Hybrid Vector RAG Tier with HITL Approval "
                        "Sandbox, and (4) PostgreSQL + Redis + Immutable Audit Storage Tier."
                    ),
                    "mermaid_diagram": mermaid_code,
                    "key_architectural_decisions": [
                        "Event-driven decoupling via Redis Streams prevents slow AI/LLM inference calls from blocking synchronous user HTTP requests.",
                        "Strict separation between retrieved RAG evidence and generated agent proposals prevents ungrounded hallucinations.",
                        "Sandboxed HITL state machine ensures no automated agent can execute destructive or compliance-sensitive operations without human sign-off.",
                    ] + ([f"Revision Applied (Pass #{revision_count}): {c}" for c in corrections_applied] if corrections_applied else []),
                },
                "8_technology_stack": tech_stack,
                "9_database_design": [
                    {
                        "table": "organizations & users",
                        "engine": "PostgreSQL",
                        "purpose": "Multi-tenant organization hierarchy, RBAC roles (Admin, Manager, Operator), and SSO metadata.",
                        "key_columns": "id (UUID PK), org_id (UUID FK), email, role, mfa_enabled, created_at",
                    },
                    {
                        "table": "domain_records & workflows",
                        "engine": "PostgreSQL",
                        "purpose": f"Primary transactional entities for {title} with state-machine status tracking and JSONB metadata.",
                        "key_columns": "id (UUID PK), org_id (UUID FK), status, payload_jsonb, version, updated_at",
                    },
                    {
                        "table": "ai_agent_runs & hitl_approvals",
                        "engine": "PostgreSQL",
                        "purpose": "Stores agent execution traces, retrieved RAG citation IDs, Critic scores, and human reviewer sign-offs.",
                        "key_columns": "run_id (UUID PK), entity_id (FK), agent_stage, critic_score, hitl_status, reviewer_id, reviewed_at",
                    },
                    {
                        "table": "vector_knowledge_chunks",
                        "engine": "Pinecone / pgvector",
                        "purpose": "Stores chunked domain documents, embeddings, and verified source URLs for semantic retrieval.",
                        "key_columns": "chunk_id (PK), doc_id, embedding_vector, source_url, verified_flag",
                    },
                ],
                "10_api_and_integration_requirements": [
                    {"method": "POST", "endpoint": "/api/v1/workflows/ingest", "description": "Validates and ingests incoming domain records/requests with idempotency key."},
                    {"method": "POST", "endpoint": "/api/v1/agents/execute", "description": "Triggers stateful Researcher -> Architect/Domain -> Critic pipeline asynchronously."},
                    {"method": "GET", "endpoint": "/api/v1/workflows/{id}/trace", "description": "Returns step-by-step agent trace, RAG citations, and Critic audit findings."},
                    {"method": "POST", "endpoint": "/api/v1/hitl/{id}/decision", "description": "Records authenticated human reviewer Approval, Rejection, or Revision Request."},
                    {"method": "GET", "endpoint": "/api/v1/analytics/summary", "description": "Provides real-time operational KPIs, SLA latency metrics, and audit exports."},
                ],
                "11_relevant_cynbit_case_studies": cs_section_items,
                "12_implementation_phases": [
                    {
                        "phase": "Phase 01: Discovery & Audit",
                        "duration_assumption": "Weeks 1–2 (Preliminary Assumption)",
                        "deliverables": "Stakeholder requirements workshops, legacy data/API audit, and HITL governance boundary sign-off.",
                    },
                    {
                        "phase": "Phase 02: Strategy & Blueprinting",
                        "duration_assumption": "Weeks 2–3 (Preliminary Assumption)",
                        "deliverables": "Finalized OpenAPI specs, PostgreSQL ER diagrams, cloud security topology, and UI wireframes.",
                    },
                    {
                        "phase": "Phase 03: High-Fidelity Design",
                        "duration_assumption": "Weeks 3–4 (Preliminary Assumption)",
                        "deliverables": "Interactive Next.js design system, operator workflows, and HITL review console prototypes.",
                    },
                    {
                        "phase": "Phase 04: Technical Implementation",
                        "duration_assumption": f"Weeks 5–{base_weeks - 2} (Preliminary Assumption)",
                        "deliverables": "Core FastAPI/NestJS services, PostgreSQL migrations, RAG/AI pipeline integration, and frontend build.",
                    },
                    {
                        "phase": "Phase 05: Quality & Integrity Check",
                        "duration_assumption": f"Week {base_weeks - 1} (Preliminary Assumption)",
                        "deliverables": "Automated Pytest suite, load/stress testing (99.9% uptime target), security red-teaming, and UAT.",
                    },
                    {
                        "phase": "Phase 06: Deployment & Evolution",
                        "duration_assumption": f"Week {base_weeks} (Preliminary Assumption)",
                        "deliverables": "Zero-downtime production launch on AWS/Azure, operator training, and telemetry SLA monitoring.",
                    },
                ],
                "13_preliminary_timeline_and_effort_estimates": {
                    "disclaimer": (
                        "PRELIMINARY ASSUMPTION NOTICE: All timeline and effort figures below are non-binding preliminary "
                        "engineering assumptions generated for discovery scoping. Final delivery commitments, pricing, "
                        "and resource allocation require formal Senior Architect scoping with the customer."
                    ),
                    "estimated_duration_range": est_range,
                    "assumed_team_composition": "1 Senior Technical Architect (Part-Time), 2 Full-Stack/Backend Engineers, 1 AI/ML Engineer, 1 QA/DevSecOps Engineer (Preliminary Assumption).",
                    "effort_drivers": f"{len(func_reqs)} functional capability modules, {len(non_func_reqs)} non-functional SLA targets, and external integration validation.",
                },
                "14_infrastructure_requirements": [
                    "Compute: Containerized Docker services orchestrated via managed Kubernetes (EKS/AKS) or AWS ECS Fargate with auto-scaling.",
                    "Database & Cache: Multi-AZ managed PostgreSQL 16 with automated point-in-time recovery (PITR) and managed Redis 7 cluster.",
                    "CI/CD & Observability: GitHub Actions + Terraform Infrastructure-as-Code with Prometheus/Grafana and structured JSON logging.",
                ],
                "15_security_considerations": security_items,
                "16_technical_risks_and_mitigation": [
                    {
                        "risk": "Third-Party API Schema Drift or Rate Limiting",
                        "severity": "Medium",
                        "mitigation": "Implement circuit breakers, exponential backoff, Redis request queuing, and strict Pydantic schema validation.",
                    },
                    {
                        "risk": "AI / LLM Hallucination or Ungrounded Recommendations",
                        "severity": "High",
                        "mitigation": "Enforce strict RAG citation thresholds, deterministic Critic Agent verification, and mandatory Human-in-the-Loop sign-off.",
                    },
                    {
                        "risk": "Legacy Data Quality & Migration Inconsistencies",
                        "severity": "Medium",
                        "mitigation": "Execute staged dry-run ETL migrations in a sandbox environment with automated reconciliation checksums before cutover.",
                    },
                ],
                "17_assumptions_and_missing_information": {
                    "key_assumptions": [
                        "Customer will provide timely access to domain subject-matter experts (SMEs) and sandbox API credentials during Phase 01.",
                        "All production deployments will utilize standard AWS or Azure cloud regions compliant with customer data residency rules.",
                    ],
                    "missing_information_to_clarify": findings["missing_information"],
                },
                "18_recommended_next_steps": [
                    "Step 1: Internal Cynbit Senior Architect reviews and approves this AutoBlueprint in the Approval Center.",
                    "Step 2: Conduct 45-minute technical discovery call with the customer to resolve the Missing Information items in Section 17.",
                    "Step 3: Finalize Phase 01 Discovery & Audit statement of work and provision isolated sandbox environment.",
                ],
            }

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logs = list(state.get("agent_execution_logs", []))
        mode_note = f"[{exec_mode}]" + (f" ({llm_err})" if llm_err and "AFTER" in exec_mode else "")
        logs.append({
            "agent": f"Architect Agent (Pass #{revision_count + 1})",
            "status": "COMPLETED",
            "execution_mode": exec_mode,
            "duration_ms": elapsed_ms,
            "summary": (
                f"{mode_note} Generated 18-section technical blueprint for '{inquiry.get('project_title')}' "
                f"with {len(sections.get('8_technology_stack', []))} stack layers and {len(cs_section_items)} verified reference(s)."
                + (f" Addressed {len(corrections_applied)} Critic correction(s)." if corrections_applied else "")
            ),
        })

        return {
            "blueprint_sections": sections,
            "mermaid_diagram": mermaid_code,
            "agent_execution_logs": logs,
        }
