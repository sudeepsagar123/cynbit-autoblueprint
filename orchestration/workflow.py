"""Stage 5 of Core Spec: End-to-End Multi-Agent Workflow Orchestrator using LangGraph StateGraph.

Uses `create_state_graph(WorkflowStateTypedDict)` from `orchestration.graph`, which instantiates
`langgraph.graph.StateGraph` when `langgraph` is installed and falls back to `FallbackStateGraph`
in minimal offline environments.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from agents.approval import HumanApprovalManager
from agents.architect import ArchitectAgent
from agents.critic import CriticAgent
from agents.researcher import ResearcherAgent
from orchestration.graph import END, create_state_graph
from orchestration.state import ProjectInquiryInput, WorkflowStateTypedDict
from rag.retrieval import CynbitVectorRAGIndex
from services.blueprint_generator import format_blueprint_markdown
from services.llm_client import GeminiStructuredClient


class CynbitAutoBlueprintWorkflow:
    """Orchestrates the 4-stage Multi-Agent System with conditional revision loops."""

    def __init__(
        self,
        max_retries: int = 2,
        llm_client: Optional[GeminiStructuredClient] = None,
        rag_index: Optional[CynbitVectorRAGIndex] = None,
        graph_cls_override: Any = None,
    ) -> None:
        self.max_retries = max_retries
        self.llm_client = llm_client or GeminiStructuredClient()
        self.rag_index = rag_index or CynbitVectorRAGIndex(api_key=self.llm_client.api_key or None)
        self.researcher = ResearcherAgent(self.rag_index, llm_client=self.llm_client)
        self.architect = ArchitectAgent(llm_client=self.llm_client)
        self.critic = CriticAgent(llm_client=self.llm_client)
        self.graph, self.orchestration_backend = self._build_graph(graph_cls_override=graph_cls_override)

    def _validate_input_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_inq = state.get("inquiry", {})
        errors = []
        try:
            validated = ProjectInquiryInput(**raw_inq)
            clean_dict = validated.model_dump() if hasattr(validated, "model_dump") else validated.dict()
            valid_flag = True
        except Exception as exc:
            clean_dict = dict(raw_inq)
            valid_flag = False
            errors.append(str(exc))

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logs = list(state.get("agent_execution_logs", []))
        logs.append({
            "agent": "Input Validator & Security Gate",
            "status": "PASSED" if valid_flag else "FAILED",
            "execution_mode": self.llm_client.get_execution_mode_label(),
            "duration_ms": elapsed_ms,
            "summary": "Validated inquiry schema and security constraints." if valid_flag else f"Validation failed: {'; '.join(errors)}",
        })
        return {
            "inquiry": clean_dict,
            "validation_passed": valid_flag,
            "validation_errors": errors,
            "agent_execution_logs": logs,
        }

    def _researcher_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("validation_passed", True):
            return {}
        return self.researcher.run(state)

    def _architect_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("validation_passed", True):
            return {}
        updates = self.architect.run(state)
        merged = dict(state)
        merged.update(updates)
        updates["blueprint_markdown"] = format_blueprint_markdown(merged)
        return updates

    def _critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("validation_passed", True):
            return {}
        updates = self.critic.run(state)
        merged = dict(state)
        merged.update(updates)
        updates["blueprint_markdown"] = format_blueprint_markdown(merged)
        return updates

    def _hitl_gate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state.get("validation_passed", True):
            return {"approval_status": "REJECTED"}
        updates = HumanApprovalManager.stage_for_human_review(state)
        merged = dict(state)
        merged.update(updates)
        updates["blueprint_markdown"] = format_blueprint_markdown(merged)
        return updates

    def _route_after_validation(self, state: Dict[str, Any]) -> str:
        return "researcher_agent" if state.get("validation_passed", False) else END

    def _route_after_critic(self, state: Dict[str, Any]) -> str:
        review = state.get("critic_review") or {}
        if review.get("approval_recommendation") == "NEEDS_REVISION":
            return "architect_agent"
        return "human_approval_gate"

    def _build_graph(self, graph_cls_override: Any = None):
        builder, backend_name = create_state_graph(WorkflowStateTypedDict, graph_cls_override=graph_cls_override)
        builder.add_node("input_validation", self._validate_input_node)
        builder.add_node("researcher_agent", self._researcher_node)
        builder.add_node("architect_agent", self._architect_node)
        builder.add_node("critic_agent", self._critic_node)
        builder.add_node("human_approval_gate", self._hitl_gate_node)

        builder.set_entry_point("input_validation")
        builder.add_conditional_edges(
            "input_validation",
            self._route_after_validation,
            {"researcher_agent": "researcher_agent", END: END},
        )
        builder.add_edge("researcher_agent", "architect_agent")
        builder.add_edge("architect_agent", "critic_agent")
        builder.add_conditional_edges(
            "critic_agent",
            self._route_after_critic,
            {
                "architect_agent": "architect_agent",
                "human_approval_gate": "human_approval_gate",
            },
        )
        builder.add_edge("human_approval_gate", END)
        return builder.compile(), backend_name

    def execute_inquiry(self, inquiry_data: Dict[str, Any], is_synthetic_demo: bool = False) -> Dict[str, Any]:
        """Runs the complete multi-agent workflow and records end-to-end processing time."""
        t0 = time.perf_counter()
        inq_id = inquiry_data.get("inquiry_id") or f"INQ-{uuid.uuid4().hex[:8].upper()}"
        initial_state: Dict[str, Any] = {
            "inquiry_id": inq_id,
            "inquiry": {**inquiry_data, "inquiry_id": inq_id},
            "execution_mode": self.llm_client.get_execution_mode_label(),
            "orchestration_backend": self.orchestration_backend,
            "validation_passed": True,
            "validation_errors": [],
            "security_flags": [],
            "research_findings": None,
            "rag_context": "",
            "blueprint_sections": None,
            "blueprint_markdown": "",
            "mermaid_diagram": "",
            "critic_review": None,
            "critic_history": [],
            "revision_count": 0,
            "max_retries": self.max_retries,
            "approval_status": "PENDING_APPROVAL",
            "reviewer_name": None,
            "reviewer_notes": None,
            "approved_at": None,
            "is_synthetic_demo": is_synthetic_demo,
            "agent_execution_logs": [],
            "processing_time_ms": 0.0,
        }
        final_state = self.graph.invoke(initial_state)
        final_state["processing_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return final_state
