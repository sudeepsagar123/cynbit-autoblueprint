"""Renders the 18-section structured Technical Blueprint into clean, professional Markdown."""

from __future__ import annotations

from typing import Any, Dict


def format_blueprint_markdown(state: Dict[str, Any]) -> str:
    """Formats all 18 required sections of the generated blueprint into structured Markdown."""
    inq = state.get("inquiry") or {}
    sec = state.get("blueprint_sections") or {}
    critic = state.get("critic_review") or {}
    status = state.get("approval_status", "PENDING_APPROVAL")

    lines = [
        f"# CYNBIT AUTOBLUEPRINT: {inq.get('project_title', 'Technical Architecture Blueprint')}",
        f"**Client Organization:** {inq.get('company_name', 'N/A')} | **Contact:** {inq.get('customer_name', 'N/A')}  ",
        f"**Industry:** {inq.get('industry', 'N/A')} | **Governance Status:** `{status}`  ",
        f"**Critic Quality Score:** `{critic.get('overall_quality_score', 'N/A')}/100` | "
        f"**Requirement Coverage:** `{critic.get('requirement_coverage_pct', 'N/A')}%`",
        "",
        "## 1. Project Overview",
        str(sec.get("1_project_overview", "")),
        "",
        "## 2. Customer Business Problem",
        str(sec.get("2_customer_business_problem", "")),
        "",
        "## 3. Project Objectives",
    ]
    for obj in sec.get("3_project_objectives", []):
        lines.append(f"- {obj}")

    lines.extend(["", "## 4. Extracted Functional Requirements"])
    for fr in sec.get("4_extracted_functional_requirements", []):
        lines.append(f"- **{fr.get('req_id')}**: {fr.get('requirement')} *(Mapping: {fr.get('architectural_mapping')})*")

    lines.extend(["", "## 5. Non-Functional Requirements"])
    for nfr in sec.get("5_non_functional_requirements", []):
        lines.append(f"- {nfr}")

    lines.extend([
        "",
        "## 6. Proposed Solution",
        str(sec.get("6_proposed_solution", "")),
        "",
        "## 7. Recommended System Architecture",
        str(sec.get("7_recommended_system_architecture", {}).get("summary", "")),
        "",
        "### Architecture Diagram (Mermaid)",
        "```mermaid",
        str(sec.get("7_recommended_system_architecture", {}).get("mermaid_diagram", "")),
        "```",
        "",
        "### Key Architectural Decisions",
    ])
    for dec in sec.get("7_recommended_system_architecture", {}).get("key_architectural_decisions", []):
        lines.append(f"- {dec}")

    lines.extend(["", "## 8. Technology Stack"])
    for item in sec.get("8_technology_stack", []):
        lines.append(f"- **{item.get('layer')} — {item.get('technology')}**: {item.get('rationale')}")

    lines.extend(["", "## 9. Database Design"])
    for tbl in sec.get("9_database_design", []):
        lines.append(
            f"- **`{tbl.get('table')}` ({tbl.get('engine')})**: {tbl.get('purpose')} "
            f"| *Columns:* `{tbl.get('key_columns')}`"
        )

    lines.extend(["", "## 10. API and Integration Requirements"])
    for api in sec.get("10_api_and_integration_requirements", []):
        lines.append(f"- **`{api.get('method')} {api.get('endpoint')}`**: {api.get('description')}")

    lines.extend(["", "## 11. Relevant Cynbit Case Studies (Verified RAG Evidence)"])
    for cs in sec.get("11_relevant_cynbit_case_studies", []):
        lines.append(
            f"- **[{cs.get('doc_id')}] {cs.get('title')}** (Client: {cs.get('client')} | "
            f"Similarity: `{cs.get('similarity')}` | Verified Metric: *{cs.get('verified_metric')}* | "
            f"Source: {cs.get('url')}) — {cs.get('relevance_note')}"
        )

    lines.extend(["", "## 12. Implementation Phases (Cynbit 6-Stage Lifecycle)"])
    for ph in sec.get("12_implementation_phases", []):
        lines.append(f"- **{ph.get('phase')}** (*{ph.get('duration_assumption')}*): {ph.get('deliverables')}")

    est = sec.get("13_preliminary_timeline_and_effort_estimates", {})
    lines.extend([
        "",
        "## 13. Preliminary Timeline and Effort Estimates",
        f"> **Disclaimer:** {est.get('disclaimer', '')}",
        f"- **Estimated Duration Range:** {est.get('estimated_duration_range', '')}",
        f"- **Assumed Team Composition:** {est.get('assumed_team_composition', '')}",
        f"- **Primary Effort Drivers:** {est.get('effort_drivers', '')}",
        "",
        "## 14. Infrastructure Requirements",
    ])
    for inf in sec.get("14_infrastructure_requirements", []):
        lines.append(f"- {inf}")

    lines.extend(["", "## 15. Security Considerations"])
    for s_item in sec.get("15_security_considerations", []):
        lines.append(f"- {s_item}")

    lines.extend(["", "## 16. Technical Risks and Mitigation"])
    for r in sec.get("16_technical_risks_and_mitigation", []):
        lines.append(f"- **{r.get('risk')} (Severity: {r.get('severity')})**: {r.get('mitigation')}")

    assump = sec.get("17_assumptions_and_missing_information", {})
    lines.extend(["", "## 17. Assumptions and Missing Information", "### Key Engineering Assumptions"])
    for a in assump.get("key_assumptions", []):
        lines.append(f"- {a}")
    lines.append("### Missing Information to Clarify in Discovery Call")
    for m in assump.get("missing_information_to_clarify", []):
        lines.append(f"- {m}")

    lines.extend(["", "## 18. Recommended Next Steps"])
    for ns in sec.get("18_recommended_next_steps", []):
        lines.append(f"- {ns}")

    return "\n".join(lines)
