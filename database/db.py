"""SQLite Persistence Layer for Cynbit AutoBlueprint.

Stores inquiries, generated blueprints, agent execution logs, Critic reviews,
and Human-in-the-Loop approval records. Computes real-time dashboard metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


class AutoBlueprintDatabase:
    """Thread-safe SQLite storage for blueprints, inquiries, and approval audit logs."""

    def __init__(self, db_path: str = "data/autoblueprint.db") -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blueprints (
                    inquiry_id TEXT PRIMARY KEY,
                    customer_name TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    email TEXT,
                    industry TEXT NOT NULL,
                    project_title TEXT NOT NULL,
                    project_description TEXT NOT NULL,
                    classified_domain TEXT,
                    approval_status TEXT NOT NULL,
                    critic_score INTEGER,
                    requirement_coverage REAL,
                    revision_count INTEGER DEFAULT 0,
                    processing_time_ms REAL DEFAULT 0.0,
                    reviewer_name TEXT,
                    reviewer_notes TEXT,
                    approved_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_json TEXT NOT NULL
                )
                """
            )
            conn.commit()


    def delete_blueprint(self, inquiry_id: str) -> bool:
        """Delete one blueprint by its unique inquiry ID."""
        if not inquiry_id or not inquiry_id.strip():
            raise ValueError("A valid inquiry ID is required.")

        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM blueprints WHERE inquiry_id = ?",
                (inquiry_id.strip(),)
            )
            return cursor.rowcount > 0

    def delete_blueprints(self, inquiry_ids: list[str]) -> int:
        """Delete selected blueprints and return the number deleted."""
        ids = list(dict.fromkeys(
            item.strip() for item in inquiry_ids if item and item.strip()
        ))

        if not ids:
            return 0

        with self._connect() as conn:
            cursor = conn.executemany(
                "DELETE FROM blueprints WHERE inquiry_id = ?",
                [(item,) for item in ids]
            )
            return cursor.rowcount

    def save_blueprint(self, state: Dict[str, Any]) -> str:
        inq = state.get("inquiry", {})
        findings = state.get("research_findings") or {}
        critic = state.get("critic_review") or {}
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        inquiry_id = state["inquiry_id"]

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM blueprints WHERE inquiry_id = ?", (inquiry_id,)
            ).fetchone()
            created_at = existing["created_at"] if existing else now_iso

            conn.execute(
                """
                INSERT OR REPLACE INTO blueprints (
                    inquiry_id, customer_name, company_name, email, industry,
                    project_title, project_description, classified_domain,
                    approval_status, critic_score, requirement_coverage,
                    revision_count, processing_time_ms, reviewer_name,
                    reviewer_notes, approved_at, created_at, updated_at, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inquiry_id,
                    inq.get("customer_name", "Unknown"),
                    inq.get("company_name", "Unknown"),
                    inq.get("email", ""),
                    inq.get("industry", "Enterprise"),
                    inq.get("project_title", "Untitled Project"),
                    inq.get("project_description", ""),
                    findings.get("classified_domain", "Enterprise Software"),
                    state.get("approval_status", "PENDING_APPROVAL"),
                    int(critic.get("overall_quality_score", 0)),
                    float(critic.get("requirement_coverage_pct", 0.0)),
                    int(state.get("revision_count", 0)),
                    float(state.get("processing_time_ms", 0.0)),
                    state.get("reviewer_name"),
                    state.get("reviewer_notes"),
                    state.get("approved_at"),
                    created_at,
                    now_iso,
                    json.dumps(state),
                ),
            )
            conn.commit()
        return inquiry_id

    def get_blueprint(self, inquiry_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json, created_at, updated_at FROM blueprints WHERE inquiry_id = ?",
                (inquiry_id,),
            ).fetchone()
            if not row:
                return None
            data = json.loads(row["state_json"])
            data["created_at"] = row["created_at"]
            data["updated_at"] = row["updated_at"]
            return data

    def list_blueprints(
        self,
        status_filter: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM blueprints WHERE 1=1"
        params: List[Any] = []
        if status_filter and status_filter != "ALL":
            sql += " AND approval_status = ?"
            params.append(status_filter)
        if search_query and search_query.strip():
            q = f"%{search_query.strip()}%"
            sql += " AND (company_name LIKE ? OR project_title LIKE ? OR customer_name LIKE ? OR classified_domain LIKE ?)"
            params.extend([q, q, q, q])
        sql += " ORDER BY updated_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            out = []
            for r in rows:
                state = json.loads(r["state_json"])
                state["created_at"] = r["created_at"]
                state["updated_at"] = r["updated_at"]
                out.append(state)
            return out

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Computes real application statistics directly from SQLite records."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM blueprints").fetchall()

        total = len(rows)
        pending = sum(1 for r in rows if r["approval_status"] == "PENDING_APPROVAL")
        approved = sum(1 for r in rows if r["approval_status"] == "APPROVED")
        rejected = sum(1 for r in rows if r["approval_status"] == "REJECTED")
        revision_req = sum(1 for r in rows if r["approval_status"] == "REVISION_REQUESTED")
        avg_time_ms = (
            round(sum(float(r["processing_time_ms"]) for r in rows) / total, 2)
            if total > 0
            else 0.0
        )
        avg_score = (
            round(sum(int(r["critic_score"]) for r in rows) / total, 1)
            if total > 0
            else 0.0
        )

        return {
            "total_inquiries_processed": total,
            "blueprints_generated": total,
            "pending_approvals": pending,
            "approved_blueprints": approved,
            "rejected_blueprints": rejected,
            "revision_requested_blueprints": revision_req,
            "average_processing_time_ms": avg_time_ms,
            "average_critic_score": avg_score,
        }
