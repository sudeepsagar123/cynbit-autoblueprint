#!/usr/bin/env python3
"""Cynbit AutoBlueprint — Multi-Agent AI Discovery & Technical Blueprint Generation System.

Provides a 6-Page Streamlit SaaS Application:
  - Page 1: Overview Dashboard (Real SQLite metrics, execution mode status, processing times)
  - Page 2: New Project Inquiry (Form + synthetic sample inquiries loader)
  - Page 3: Multi-Agent Execution (Live trace of Researcher, RAG, Architect, Critic, and HITL Gate)
  - Page 4: Blueprint Viewer (18-section structured viewer, Mermaid diagram, and ReportLab PDF export)
  - Page 5: Approval Center (HITL review interface: Approve, Reject, Edit, or Request Revisions)
  - Page 6: Blueprint History (SQLite search, status filter, and audit log viewer)

Works seamlessly in both:
  1. LIVE_GEMINI_API mode (when GEMINI_API_KEY is configured in environment / .env)
  2. OFFLINE_DEMO_MODE (clearly labeled deterministic fallback when GEMINI_API_KEY is not set)

NOTE ON SEEDED RECORDS:
Initial seeded records in `data/autoblueprint.db` use clearly labeled synthetic demo reviewers
(`Synthetic Demo Reviewer A` / `Synthetic Demo Reviewer B`) and never claim real approvals by
Cynbit leadership.
"""

from __future__ import annotations

import argparse
import json
import os

# Load Streamlit Community Cloud secrets into environment variables.
try:
    import streamlit as st
    for key in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_EMBEDDING_MODEL",
    ):
        value = st.secrets.get(key)
        if value and not os.environ.get(key):
            os.environ[key] = str(value)
except Exception:
    pass

from typing import Any, Dict, List

from agents.approval import HumanApprovalManager
from database.db import AutoBlueprintDatabase
from orchestration.workflow import CynbitAutoBlueprintWorkflow
from services.pdf_export import export_blueprint_to_pdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

DEFAULT_DB_PATH = os.environ.get("AUTOBLUEPRINT_DB_PATH", os.path.join(BASE_DIR, "data", "autoblueprint.db"))
SAMPLE_INQUIRIES_PATH = os.path.join(BASE_DIR, "data", "sample_inquiries.json")


def load_sample_inquiries() -> List[Dict[str, Any]]:
    with open(SAMPLE_INQUIRIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_initial_records_if_empty(db: AutoBlueprintDatabase, workflow: CynbitAutoBlueprintWorkflow) -> None:
    """Seeds synthetic demonstration inquiries through the Multi-Agent Workflow if the database is empty.

    Uses clearly labeled synthetic demo reviewer names (never real Cynbit personnel).
    """
    existing = db.list_blueprints()
    if existing:
        return
    samples = load_sample_inquiries()
    for idx, sample in enumerate(samples):
        inq_copy = dict(sample)
        if idx == 2:
            inq_copy["simulate_initial_defect"] = True
        state = workflow.execute_inquiry(inq_copy, is_synthetic_demo=True)
        if idx == 0:
            state = HumanApprovalManager.apply_human_decision(
                state,
                decision="APPROVED",
                reviewer_name="Synthetic Demo Reviewer A (Demo Principal Architect)",
                reviewer_notes="[SYNTHETIC DEMO APPROVAL] Verified WizHealth RAG alignment and clinical HITL security guardrails.",
            )
        elif idx == 1:
            state = HumanApprovalManager.apply_human_decision(
                state,
                decision="APPROVED",
                reviewer_name="Synthetic Demo Reviewer B (Demo Solutions Architect)",
                reviewer_notes="[SYNTHETIC DEMO APPROVAL] Verified TenantEase reusable module fit and PostgreSQL multi-property schema.",
            )
        elif idx == 3:
            state = HumanApprovalManager.apply_human_decision(
                state,
                decision="REVISION_REQUESTED",
                reviewer_name="Synthetic Demo Reviewer A (Demo Principal Architect)",
                reviewer_notes="[SYNTHETIC DEMO REVIEW] Please confirm Instagram Graph API rate-limit quotas with client during Phase 01.",
            )
        db.save_blueprint(state)


def run_streamlit_ui() -> None:
    import streamlit as st

    # Streamlit Community Cloud exposes secrets through st.secrets, whereas local
    # development commonly uses environment variables or a .env file. Bridge both
    # configuration sources before constructing the workflow or database.
    try:
        if st.secrets.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"])
        if st.secrets.get("GEMINI_MODEL"):
            os.environ["GEMINI_MODEL"] = str(st.secrets["GEMINI_MODEL"])
        if st.secrets.get("GEMINI_EMBEDDING_MODEL"):
            os.environ["GEMINI_EMBEDDING_MODEL"] = str(st.secrets["GEMINI_EMBEDDING_MODEL"])
        if st.secrets.get("AUTOBLUEPRINT_DB_PATH"):
            os.environ["AUTOBLUEPRINT_DB_PATH"] = str(st.secrets["AUTOBLUEPRINT_DB_PATH"])
    except Exception:
        # No Streamlit secrets file is normal for local/offline demo mode.
        pass

    db_path = os.environ.get("AUTOBLUEPRINT_DB_PATH", DEFAULT_DB_PATH)

    st.set_page_config(
        page_title="Cynbit AutoBlueprint | Multi-Agent Discovery & Blueprint SaaS",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Polished SaaS visual system: dark ink, cool surfaces, violet/teal accents.
    st.markdown("""
    
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ab-ink: #172039;
    --ab-muted: #64748b;
    --ab-line: #e7ebf3;
    --ab-bg: #f5f7fb;
    --ab-card: #ffffff;
    --ab-violet: #7567f8;
    --ab-teal: #159b88;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--ab-bg);
    color: var(--ab-ink);
}

[data-testid="stHeader"] {
    background: rgba(245,247,251,.94);
}

[data-testid="stMain"] .block-container {
    max-width: 1480px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

/* Typography */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--ab-ink) !important;
    letter-spacing: -0.035em;
}

[data-testid="stMain"] h1 {
    font-size: 2.15rem !important;
    font-weight: 700 !important;
}

[data-testid="stMain"] h2 {
    font-size: 1.55rem !important;
    font-weight: 650 !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {
    color: #34415b !important;
}

[data-testid="stCaptionContainer"] p {
    color: var(--ab-muted) !important;
    font-size: .88rem;
}

/* Premium dark sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12192d 0%, #1b2540 100%);
    border-right: 1px solid rgba(255,255,255,.07);
}

[data-testid="stSidebar"] * {
    color: #e8edfb !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 10px;
    padding: .55rem .7rem;
    transition: background .18s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.09);
}

[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 13px;
}

/* Brand */
.ab-brand {
    padding: 17px 15px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 15px;
    background: linear-gradient(
        135deg,
        rgba(117,103,248,.25),
        rgba(21,155,136,.13)
    );
}

.ab-brand-title {
    font: 700 1.08rem 'Space Grotesk', sans-serif;
    color: #ffffff !important;
    letter-spacing: -.03em;
}

.ab-brand-sub {
    color: #b9c4df !important;
    font-size: .75rem;
    margin-top: 5px;
}

.ab-eyebrow {
    color: var(--ab-violet) !important;
    text-transform: uppercase;
    font-size: .72rem;
    letter-spacing: .14em;
    font-weight: 700;
    margin-bottom: 7px;
}

/* Dashboard hero */
.ab-hero {
    padding: 30px 32px;
    margin: 5px 0 24px;
    border-radius: 19px;
    color: #ffffff !important;
    background: linear-gradient(
        115deg, #20294a 0%, #5146a8 58%, #237f83 100%
    );
    box-shadow: 0 12px 30px rgba(31,42,75,.12);
}

.ab-hero h1,
.ab-hero h2,
.ab-hero h3,
.ab-hero p,
.ab-hero span,
.ab-hero div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.ab-hero h2 {
    font: 700 1.8rem 'Space Grotesk', sans-serif !important;
    margin: 0;
    line-height: 1.3;
}

.ab-hero p {
    color: #e7eaff !important;
    -webkit-text-fill-color: #e7eaff !important;
    margin-top: 9px;
    font-size: .95rem;
}

.ab-pill {
    display: inline-block;
    padding: 5px 11px;
    border-radius: 99px;
    background: rgba(255,255,255,.17);
    color: #ffffff !important;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .07em;
    margin-bottom: 12px;
}

/* Metric cards */
[data-testid="stMain"] [data-testid="stMetric"] {
    background: var(--ab-card);
    border: 1px solid var(--ab-line);
    border-radius: 15px;
    padding: 17px 18px;
    box-shadow: 0 4px 15px rgba(31,42,75,.035);
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: var(--ab-ink) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 650 !important;
}

/* Forms and input fields */
[data-testid="stMain"] input,
[data-testid="stMain"] textarea,
[data-testid="stMain"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #172039 !important;
    -webkit-text-fill-color: #172039 !important;
    border: 1px solid #dce2ed !important;
    border-radius: 10px !important;
}

[data-testid="stMain"] input:focus,
[data-testid="stMain"] textarea:focus {
    border-color: var(--ab-violet) !important;
    box-shadow: 0 0 0 2px rgba(117,103,248,.12) !important;
}

[data-testid="stMain"] input::placeholder,
[data-testid="stMain"] textarea::placeholder {
    color: #8994a8 !important;
    -webkit-text-fill-color: #8994a8 !important;
}

[data-testid="stMain"] [data-baseweb="select"] span {
    color: #172039 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #ffffff !important;
}

[data-baseweb="menu"] [role="option"] {
    color: #172039 !important;
}

/* Buttons */
[data-testid="stMain"] .stButton > button,
[data-testid="stMain"] .stDownloadButton > button,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button {
    min-height: 2.7rem;
    border-radius: 10px;
    font-weight: 650;
    border: 1px solid #dedcfb;
    transition: all .18s ease;
}

[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button[kind="primary"] {
    color: #ffffff !important;
    background: linear-gradient(110deg,#7165f5,#8b5cf6) !important;
    border: 0 !important;
    box-shadow: 0 5px 14px rgba(117,103,248,.20);
}

[data-testid="stMain"] .stButton > button:hover,
[data-testid="stMain"] .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 7px 18px rgba(31,42,75,.10);
}

/* Containers, forms, expanders */
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMain"] [data-testid="stExpander"],
[data-testid="stMain"] [data-testid="stForm"] {
    border-radius: 14px;
}

[data-testid="stMain"] [data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid var(--ab-line);
}

/* Tables */
[data-testid="stMain"] [data-testid="stDataFrame"] {
    background: #ffffff !important;
    border: 1px solid var(--ab-line);
    border-radius: 13px;
    overflow: hidden;
}

/* Alerts */
[data-testid="stMain"] [data-testid="stAlert"] {
    border-radius: 12px;
}

/* Checkboxes */
[data-testid="stMain"] [data-testid="stCheckbox"] label,
[data-testid="stMain"] [data-testid="stCheckbox"] p {
    color: #34415b !important;
}

/* JSON and code panels */
[data-testid="stMain"] [data-testid="stJson"] {
    background: #ffffff;
    border: 1px solid var(--ab-line);
    border-radius: 12px;
    padding: 12px;
}

/* Clean footer */
footer {
    visibility: hidden;
}

/* Responsive */
@media (max-width: 768px) {
    [data-testid="stMain"] .block-container {
        padding: 1rem 1rem 2rem;
    }

    .ab-hero {
        padding: 22px 20px;
    }

    .ab-hero h2 {
        font-size: 1.4rem !important;
    }
}

/* Sidebar navigation toggle */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    visibility: visible !important;
    opacity: 1 !important;
}


/* Dashboard telemetry readability */
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: #34415b !important;
    -webkit-text-fill-color: #34415b !important;
    background: transparent !important;
    font-weight: 500 !important;
}

[data-testid="stMain"] [data-testid="stCaptionContainer"] code {
    color: #26324b !important;
    background: #e9edf5 !important;
    -webkit-text-fill-color: #26324b !important;
    border: 1px solid #dce2ed;
    border-radius: 5px;
    padding: 2px 5px;
}

</style>

    """, unsafe_allow_html=True)

    if "workflow" not in st.session_state:
        st.session_state.workflow = CynbitAutoBlueprintWorkflow(max_retries=int(os.environ.get("MAX_CRITIC_RETRIES", "2")))
    if "db" not in st.session_state:
        st.session_state.db = AutoBlueprintDatabase(db_path)
        seed_initial_records_if_empty(st.session_state.db, st.session_state.workflow)

    db: AutoBlueprintDatabase = st.session_state.db
    workflow: CynbitAutoBlueprintWorkflow = st.session_state.workflow
    mode_label = workflow.llm_client.get_execution_mode_label()

    st.sidebar.markdown(
        '<div class="ab-brand"><div class="ab-brand-title">✦ CYNBIT <span style="color:#70e0d2">AUTOBLUEPRINT</span></div>'
        '<div class="ab-brand-sub">AI-powered project discovery workspace</div></div>',
        unsafe_allow_html=True,
    )
    if mode_label == "LIVE_GEMINI_API":
        st.sidebar.success(f"Mode: {mode_label} ({workflow.llm_client.model_name})")
    else:
        st.sidebar.warning(f"Mode: {mode_label} (Set GEMINI_API_KEY for Live LLM)")
    st.sidebar.caption(f"Orchestrator: `{workflow.orchestration_backend}` | RAG: `{workflow.rag_index.embedder.active_backend}`")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Page 1: Overview Dashboard",
            "Page 2: New Project Inquiry",
            "Page 3: Multi-Agent Execution",
            "Page 4: Blueprint Viewer",
            "Page 5: Approval Center",
            "Page 6: Blueprint History",
        ],
    )

    metrics = db.get_dashboard_metrics()
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Blueprints", metrics["blueprints_generated"])
    st.sidebar.metric("Pending HITL Approval", metrics["pending_approvals"])
    st.sidebar.metric("Approved Blueprints", metrics["approved_blueprints"])

    if page == "Page 1: Overview Dashboard":
        st.markdown('<div class="ab-hero"><span class="ab-pill">WORKSPACE OVERVIEW</span><h2>Turn project inquiries into clear technical direction.</h2><p>Your multi-agent discovery pipeline, blueprint activity and review status — all in one place.</p></div>', unsafe_allow_html=True)
        st.caption(
            f"Real-time telemetry from SQLite (`data/autoblueprint.db`) | "
            f"Execution Mode: `{mode_label}` | Orchestrator: `{workflow.orchestration_backend}`"
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Inquiries Processed", metrics["total_inquiries_processed"])
        c2.metric("Blueprints Generated", metrics["blueprints_generated"])
        c3.metric("Pending Approvals", metrics["pending_approvals"])
        c4.metric("Approved Blueprints", metrics["approved_blueprints"])
        c5.metric("Rejected / Revisions", metrics["rejected_blueprints"] + metrics["revision_requested_blueprints"])
        c6.metric("Avg Processing Time", f"{metrics['average_processing_time_ms']:.1f} ms")

        st.subheader("Stored Multi-Agent Blueprint Runs")
        records = db.list_blueprints()
        table_rows = [
            {
                "Inquiry ID": r["inquiry_id"],
                "Company": r["inquiry"].get("company_name"),
                "Project Title": r["inquiry"].get("project_title"),
                "Domain": (r.get("research_findings") or {}).get("classified_domain"),
                "Mode": r.get("execution_mode", "OFFLINE_DEMO_MODE"),
                "Critic Score": f"{(r.get('critic_review') or {}).get('overall_quality_score', 0)}/100",
                "Revisions": r.get("revision_count", 0),
                "Status": r.get("approval_status"),
                "Reviewer": r.get("reviewer_name") or "Awaiting HITL Review",
            }
            for r in records
        ]
        st.dataframe(table_rows, use_container_width=True)

        # Project management and safe deletion
        st.markdown("---")
        st.subheader("Manage Projects")
        st.caption(
            "Select one or more project records to remove accidental "
            "submissions or duplicate test inquiries."
        )

        if records:
            record_labels = {
                r["inquiry_id"]: (
                    f'{r["inquiry"].get("company_name", "Unknown")} — '
                    f'{r["inquiry"].get("project_title", "Untitled")} '
                    f'({r["inquiry_id"]})'
                )
                for r in records
            }

            selected_ids = st.multiselect(
                "Select projects to delete",
                options=list(record_labels.keys()),
                format_func=lambda item: record_labels[item],
                key="delete_selected_project_ids",
                placeholder="Choose one or more projects..."
            )

            confirm_delete = st.checkbox(
                "I understand that deleting these projects cannot be undone.",
                key="confirm_project_deletion"
            )

            if selected_ids:
                st.warning(
                    f"{len(selected_ids)} project record(s) selected for deletion."
                )

            if st.button(
                "Delete Selected Projects",
                type="primary",
                disabled=not selected_ids or not confirm_delete,
                key="delete_selected_projects_button"
            ):
                try:
                    deleted_count = db.delete_blueprints(selected_ids)

                    st.session_state["delete_selected_project_ids"] = []
                    st.session_state["confirm_project_deletion"] = False

                    if deleted_count:
                        st.success(
                            f"Successfully deleted {deleted_count} project record(s)."
                        )
                    else:
                        st.info("No matching project records were deleted.")

                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not delete selected projects: {exc}")
        else:
            st.info("There are no project records to manage.")

    elif page == "Page 2: New Project Inquiry":
        st.markdown('<div class="ab-eyebrow">PROJECT INTAKE</div>', unsafe_allow_html=True)
        st.title("Start with the customer brief")
        st.caption("Capture the business need. AutoBlueprint will research, design and review a first technical blueprint.")
        samples = load_sample_inquiries()
        sample_titles = ["Custom Inquiry (Blank Form)"] + [
            f"{s['company_name']} — {s['project_title']}" for s in samples
        ]
        chosen_sample = st.selectbox("Load Sample Synthetic Inquiry", sample_titles)
        defaults = (
            next(s for s in samples if f"{s['company_name']} — {s['project_title']}" == chosen_sample)
            if chosen_sample != "Custom Inquiry (Blank Form)"
            else {}
        )

        with st.form("inquiry_form"):
            col1, col2 = st.columns(2)
            with col1:
                customer_name = st.text_input("Customer Name *", value=defaults.get("customer_name", ""))
                company_name = st.text_input("Company Name *", value=defaults.get("company_name", ""))
                email = st.text_input("Email Address (Optional)", value=defaults.get("email", ""))
                industry = st.text_input("Industry *", value=defaults.get("industry", ""))
            with col2:
                project_title = st.text_input("Project Title *", value=defaults.get("project_title", ""))
                budget_range = st.text_input("Budget (Optional)", value=defaults.get("budget_range", ""))
                expected_timeline = st.text_input("Expected Timeline (Optional)", value=defaults.get("expected_timeline", ""))
                simulate_defect = st.checkbox("Simulate Initial Architect Omission (Triggers Critic Revision Loop)", value=False)

            project_description = st.text_area("Project Description *", value=defaults.get("project_description", ""), height=120)
            business_objectives = st.text_area("Business Objectives", value=defaults.get("business_objectives", ""), height=70)
            additional_requirements = st.text_area("Additional Requirements", value=defaults.get("additional_requirements", ""), height=70)
            submitted = st.form_submit_button("Execute Multi-Agent AutoBlueprint Pipeline", type="primary")

        if submitted:
            payload = {
                "customer_name": customer_name,
                "company_name": company_name,
                "email": email,
                "industry": industry,
                "project_title": project_title,
                "project_description": project_description,
                "business_objectives": business_objectives,
                "budget_range": budget_range or "Not specified",
                "expected_timeline": expected_timeline or "Not specified",
                "additional_requirements": additional_requirements,
                "simulate_initial_defect": simulate_defect,
            }
            result_state = workflow.execute_inquiry(payload, is_synthetic_demo=False)
            if not result_state.get("validation_passed", False):
                st.error(f"Input validation failed: {result_state.get('validation_errors')}")
            else:
                db.save_blueprint(result_state)
                st.session_state.active_inquiry_id = result_state["inquiry_id"]
                st.success(
                    f"Inquiry `{result_state['inquiry_id']}` processed in {result_state['processing_time_ms']} ms "
                    f"(`{result_state.get('execution_mode')}`)! Navigate to Page 3 or Page 4 to inspect."
                )

    elif page == "Page 3: Multi-Agent Execution":
        st.markdown('<div class="ab-eyebrow">ORCHESTRATION</div>', unsafe_allow_html=True)
        st.title("Multi-agent execution")
        st.caption("Follow each stage of research, architecture and quality review.")
        records = db.list_blueprints()
        selected_id = st.selectbox("Select Processed Inquiry", [r["inquiry_id"] for r in records], index=0)
        active = db.get_blueprint(selected_id)
        if active:
            st.subheader(
                f"Execution Pipeline for `{active['inquiry_id']}` — {active['inquiry']['project_title']} "
                f"(`{active.get('execution_mode', 'OFFLINE_DEMO_MODE')}`)"
            )
            for step_idx, log in enumerate(active.get("agent_execution_logs", []), start=1):
                st.info(
                    f"**Stage {step_idx}: {log['agent']}** | Status: `{log['status']}` | "
                    f"Duration: `{log['duration_ms']} ms`\n\n{log['summary']}"
                )
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Researcher Agent Output & RAG Citations")
                st.json(active.get("research_findings"))
            with col_b:
                st.subheader("Critic Agent Quality Audit History")
                st.json(active.get("critic_history"))

    elif page == "Page 4: Blueprint Viewer":
        st.markdown('<div class="ab-eyebrow">DELIVERABLE</div>', unsafe_allow_html=True)
        st.title("Technical blueprint")
        st.caption("Review the generated solution and export a shareable PDF.")
        records = db.list_blueprints()
        selected_id = st.selectbox("Select Blueprint", [r["inquiry_id"] for r in records], index=0)
        active = db.get_blueprint(selected_id)
        if active:
            pdf_out = os.path.join(BASE_DIR, "data", f"{active['inquiry_id']}_Blueprint.pdf")
            export_blueprint_to_pdf(active, pdf_out)
            with open(pdf_out, "rb") as f_pdf:
                st.download_button(
                    "📄 Download 18-Section PDF Blueprint",
                    data=f_pdf.read(),
                    file_name=f"Cynbit_AutoBlueprint_{active['inquiry_id']}.pdf",
                    mime="application/pdf",
                )
            st.markdown(active.get("blueprint_markdown", ""))

    elif page == "Page 5: Approval Center":
        st.markdown('<div class="ab-eyebrow">GOVERNANCE</div>', unsafe_allow_html=True)
        st.title("Human review & approval")
        st.caption("Keep a person in control of every final decision.")
        records = db.list_blueprints()
        selected_id = st.selectbox("Select Blueprint for Review", [r["inquiry_id"] for r in records], index=0)
        active = db.get_blueprint(selected_id)
        if active:
            st.write(f"**Current Governance Status:** `{active.get('approval_status')}`")
            reviewer_name = st.text_input(
                "Reviewer Name *",
                value=active.get("reviewer_name") or "Synthetic Demo Reviewer A (Demo Principal Architect)",
            )
            reviewer_notes = st.text_area("Reviewer Audit Notes", value=active.get("reviewer_notes") or "")
            edited_md = st.text_area("Editable Technical Blueprint Markdown", value=active.get("blueprint_markdown", ""), height=320)

            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Approve Blueprint", type="primary"):
                updated = HumanApprovalManager.apply_human_decision(active, "APPROVED", reviewer_name, reviewer_notes, edited_md)
                db.save_blueprint(updated)
                st.success(f"Blueprint `{selected_id}` explicitly APPROVED by {reviewer_name}.")
            if b2.button("🔄 Request Revisions"):
                updated = HumanApprovalManager.apply_human_decision(active, "REVISION_REQUESTED", reviewer_name, reviewer_notes, edited_md)
                db.save_blueprint(updated)
                st.warning(f"Blueprint `{selected_id}` marked as REVISION_REQUESTED.")
            if b3.button("❌ Reject Blueprint"):
                updated = HumanApprovalManager.apply_human_decision(active, "REJECTED", reviewer_name, reviewer_notes, edited_md)
                db.save_blueprint(updated)
                st.error(f"Blueprint `{selected_id}` marked as REJECTED.")

    elif page == "Page 6: Blueprint History":
        st.markdown('<div class="ab-eyebrow">RECORDS</div>', unsafe_allow_html=True)
        st.title("Blueprint history")
        st.caption("Search previous runs and inspect their review trail.")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            status_filter = st.selectbox("Filter by Status", ["ALL", "PENDING_APPROVAL", "APPROVED", "REVISION_REQUESTED", "REJECTED"])
        with f_col2:
            search_q = st.text_input("Search by Company, Title, Customer, or Domain", value="")
        filtered = db.list_blueprints(status_filter=status_filter, search_query=search_q)
        st.write(f"Showing **{len(filtered)}** stored blueprint(s):")
        for item in filtered:
            with st.expander(f"[{item['inquiry_id']}] {item['inquiry']['company_name']} — {item['inquiry']['project_title']} ({item['approval_status']})"):
                st.write(
                    f"**Domain:** {(item.get('research_findings') or {}).get('classified_domain')} | "
                    f"**Execution Mode:** `{item.get('execution_mode', 'OFFLINE_DEMO_MODE')}` | "
                    f"**Updated:** {item.get('updated_at')}"
                )
                st.write(f"**Reviewer:** {item.get('reviewer_name') or 'Pending HITL Review'} | **Notes:** {item.get('reviewer_notes') or 'None'}")


def run_cli_seed_and_verify() -> None:
    db = AutoBlueprintDatabase(DEFAULT_DB_PATH)
    workflow = CynbitAutoBlueprintWorkflow(max_retries=int(os.environ.get("MAX_CRITIC_RETRIES", "2")))
    seed_initial_records_if_empty(db, workflow)
    metrics = db.get_dashboard_metrics()
    print("=== CYNBIT AUTOBLUEPRINT CLI SEED & VERIFICATION ===")
    print(f"Execution Mode: {workflow.llm_client.get_execution_mode_label()}")
    print(f"Orchestration Backend: {workflow.orchestration_backend}")
    print(f"RAG Embedding Backend: {workflow.rag_index.embedder.active_backend}")
    print(json.dumps(metrics, indent=2))
    records = db.list_blueprints()
    if records:
        sample_pdf = os.path.join(BASE_DIR, "data", "Sample_Approved_Blueprint.pdf")
        export_blueprint_to_pdf(records[0], sample_pdf)
        print(f"Exported sample PDF blueprint to {sample_pdf} ({os.path.getsize(sample_pdf)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-and-verify", action="store_true", help="Seed SQLite DB and verify PDF export")
    args, unknown = parser.parse_known_args()
    if args.seed_and_verify:
        run_cli_seed_and_verify()
    else:
        run_streamlit_ui()
