"""AGENT 4: HUMAN-IN-THE-LOOP (HITL) APPROVAL GATE
Purpose: Ensure that generated technical blueprints are reviewed by a human architect before
external delivery. Supports Approve, Reject, Request Revision, and inline Blueprint Editing.
Never auto-approves or auto-dispatches blueprints to customers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

VALID_DECISIONS = {"APPROVED", "REJECTED", "REVISION_REQUESTED", "PENDING_APPROVAL"}


class HumanApprovalManager:
    """Manages human review decisions, inline edits, and audit timestamps."""

    @staticmethod
    def stage_for_human_review(state: Dict[str, Any]) -> Dict[str, Any]:
        """Places a Critic-verified blueprint into PENDING_APPROVAL state."""
        logs = list(state.get("agent_execution_logs", []))
        logs.append({
            "agent": "Human-in-the-Loop (HITL) Gate",
            "status": "AWAITING_HUMAN_REVIEW",
            "duration_ms": 0.1,
            "summary": (
                "Blueprint staged in Approval Center with status PENDING_APPROVAL. "
                "External delivery is locked until an authorized human reviewer approves."
            ),
        })
        return {
            "approval_status": "PENDING_APPROVAL",
            "agent_execution_logs": logs,
        }

    @staticmethod
    def apply_human_decision(
        blueprint_record: Dict[str, Any],
        decision: str,
        reviewer_name: str,
        reviewer_notes: str = "",
        edited_markdown: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Applies an explicit human reviewer decision (APPROVED, REJECTED, or REVISION_REQUESTED)."""
        normalized = decision.strip().upper()
        if normalized not in VALID_DECISIONS:
            raise ValueError(f"Invalid approval decision '{decision}'. Must be one of {sorted(VALID_DECISIONS)}.")
        if not reviewer_name or not reviewer_name.strip():
            raise ValueError("Reviewer name is required for Human-in-the-Loop governance sign-off.")

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        updated = dict(blueprint_record)
        updated["approval_status"] = normalized
        updated["reviewer_name"] = reviewer_name.strip()
        updated["reviewer_notes"] = reviewer_notes.strip() or f"Decision {normalized} recorded by {reviewer_name.strip()}."
        updated["approved_at"] = now_iso if normalized == "APPROVED" else updated.get("approved_at")
        updated["updated_at"] = now_iso

        if edited_markdown and edited_markdown.strip():
            updated["blueprint_markdown"] = edited_markdown.strip()

        logs = list(updated.get("agent_execution_logs", []))
        logs.append({
            "agent": f"HITL Reviewer ({reviewer_name.strip()})",
            "status": normalized,
            "duration_ms": 0.0,
            "summary": f"Human reviewer set status to {normalized}. Notes: {updated['reviewer_notes']}",
        })
        updated["agent_execution_logs"] = logs
        return updated
