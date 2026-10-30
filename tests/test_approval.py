"""Automated Tests for Human-in-the-Loop (HITL) Approval, Persistence, and PDF Export."""

import os
import tempfile
import unittest

from agents.approval import HumanApprovalManager
from database.db import AutoBlueprintDatabase
from orchestration.workflow import CynbitAutoBlueprintWorkflow
from services.pdf_export import export_blueprint_to_pdf


class TestApprovalAndPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_autoblueprint.db")
        self.db = AutoBlueprintDatabase(self.db_path)
        self.workflow = CynbitAutoBlueprintWorkflow(max_retries=2)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_01_hitl_approval_lifecycle_and_sqlite_persistence(self) -> None:
        inquiry = {
            "inquiry_id": "INQ-TEST-900",
            "customer_name": "Siddharth Jain",
            "company_name": "EduSpark Academy",
            "industry": "EdTech",
            "project_title": "Adaptive Coding & AI Mock Interview Platform",
            "project_description": (
                "We want an adaptive coding platform for 5,000 students with placement assessments, "
                "AI code quality review, and mock technical interviews."
            ),
            "business_objectives": "Increase student placement rates by 35%.",
        }
        state = self.workflow.execute_inquiry(inquiry)
        self.assertEqual(state["approval_status"], "PENDING_APPROVAL")
        self.db.save_blueprint(state)

        # Verify human revision request using synthetic reviewer identifier
        rev_state = HumanApprovalManager.apply_human_decision(
            state,
            decision="REVISION_REQUESTED",
            reviewer_name="Synthetic Demo Reviewer B (Demo QA Architect)",
            reviewer_notes="Add BitElevate 16-dimension skill radar mapping note.",
            edited_markdown=state["blueprint_markdown"] + "\n\n<!-- Reviewer Edit Applied -->",
        )
        self.db.save_blueprint(rev_state)
        loaded_rev = self.db.get_blueprint("INQ-TEST-900")
        self.assertEqual(loaded_rev["approval_status"], "REVISION_REQUESTED")
        self.assertIn("Reviewer Edit Applied", loaded_rev["blueprint_markdown"])

        # Verify explicit human approval
        app_state = HumanApprovalManager.apply_human_decision(
            loaded_rev,
            decision="APPROVED",
            reviewer_name="Synthetic Demo Reviewer A (Demo Lead Architect)",
            reviewer_notes="All architectural and security checks passed. Approved.",
        )
        self.db.save_blueprint(app_state)
        loaded_app = self.db.get_blueprint("INQ-TEST-900")
        self.assertEqual(loaded_app["approval_status"], "APPROVED")
        self.assertEqual(loaded_app["reviewer_name"], "Synthetic Demo Reviewer A (Demo Lead Architect)")
        self.assertIsNotNone(loaded_app["approved_at"])

        metrics = self.db.get_dashboard_metrics()
        self.assertEqual(metrics["total_inquiries_processed"], 1)
        self.assertEqual(metrics["approved_blueprints"], 1)

        # Verify PDF export
        pdf_path = os.path.join(self.tmp_dir.name, "test_export.pdf")
        export_blueprint_to_pdf(loaded_app, pdf_path)
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 2500)

    def test_02_delete_single_blueprint_by_id(self) -> None:
        state = {
            "inquiry_id": "INQ-DELETE-001",
            "inquiry": {"customer_name": "Test User", "company_name": "Test Co",
                        "industry": "Software", "project_title": "Delete Test",
                        "project_description": "A temporary test record."},
            "research_findings": {},
            "critic_review": {},
            "approval_status": "PENDING_APPROVAL",
        }
        self.db.save_blueprint(state)
        self.assertIsNotNone(self.db.get_blueprint("INQ-DELETE-001"))
        self.assertTrue(self.db.delete_blueprint("INQ-DELETE-001"))
        self.assertIsNone(self.db.get_blueprint("INQ-DELETE-001"))
        self.assertFalse(self.db.delete_blueprint("INQ-DELETE-001"))

    def test_03_delete_selected_blueprints_only(self) -> None:
        for inquiry_id in ("INQ-KEEP-001", "INQ-DELETE-002", "INQ-DELETE-003"):
            self.db.save_blueprint({
                "inquiry_id": inquiry_id,
                "inquiry": {"customer_name": "Test User", "company_name": "Test Co",
                            "industry": "Software", "project_title": inquiry_id,
                            "project_description": "A temporary test record."},
                "research_findings": {}, "critic_review": {},
                "approval_status": "PENDING_APPROVAL",
            })

        deleted = self.db.delete_blueprints([
            "INQ-DELETE-002", "INQ-DELETE-003", "INQ-DELETE-002", "", None
        ])
        self.assertEqual(deleted, 2)
        self.assertIsNotNone(self.db.get_blueprint("INQ-KEEP-001"))
        self.assertIsNone(self.db.get_blueprint("INQ-DELETE-002"))
        self.assertIsNone(self.db.get_blueprint("INQ-DELETE-003"))
        self.assertEqual(self.db.delete_blueprints([]), 0)

    def test_04_delete_rejects_blank_identifier(self) -> None:
        with self.assertRaises(ValueError):
            self.db.delete_blueprint("   ")


if __name__ == "__main__":
    unittest.main()
