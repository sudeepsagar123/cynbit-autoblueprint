"""Automated Tests for End-to-End Multi-Agent Workflow & Bounded Revision Loop."""

import unittest

from orchestration.workflow import CynbitAutoBlueprintWorkflow


class TestWorkflowAndRevisionLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = CynbitAutoBlueprintWorkflow(max_retries=2)

    def test_01_standard_end_to_end_workflow(self) -> None:
        inquiry = {
            "customer_name": "Rohan Gupta",
            "company_name": "GlobalFin Corp",
            "industry": "FinTech",
            "project_title": "Cross-Border Payment Settlement Engine",
            "project_description": (
                "We need a high-precision cross-border payment routing engine with zero-trust compliance, "
                "idempotent ledger reconciliation, and real-time anomaly detection."
            ),
            "business_objectives": "Sub-second payment routing and 99.99% ledger accuracy.",
        }
        final_state = self.workflow.execute_inquiry(inquiry)
        self.assertTrue(final_state["validation_passed"])
        self.assertEqual(final_state["approval_status"], "PENDING_APPROVAL")
        self.assertEqual(final_state["revision_count"], 0)
        self.assertGreater(len(final_state["blueprint_markdown"]), 1500)

    def test_02_bounded_revision_loop_triggers_and_recovers(self) -> None:
        inquiry = {
            "customer_name": "Ananya Sen",
            "company_name": "OmniRetail Brands",
            "industry": "Retail & E-Commerce",
            "project_title": "Multi-Branch Customer Review & Sentiment Hub",
            "project_description": (
                "We need cross-platform review syncing across 25 branches, aspect-based sentiment analysis, "
                "automated draft responses, and manager crisis escalation alerts."
            ),
            "business_objectives": "Improve branch sentiment ratings and automate review responses.",
            "simulate_initial_defect": True,
        }
        final_state = self.workflow.execute_inquiry(inquiry)
        # First Critic pass must catch the simulated defect and trigger revision_count = 1
        self.assertEqual(final_state["revision_count"], 1)
        self.assertEqual(len(final_state["critic_history"]), 2)
        self.assertEqual(final_state["critic_history"][0]["approval_recommendation"], "NEEDS_REVISION")
        self.assertEqual(final_state["critic_history"][1]["approval_recommendation"], "READY_FOR_HUMAN_REVIEW")
        self.assertEqual(final_state["critic_review"]["requirement_coverage_pct"], 100.0)
        self.assertEqual(final_state["approval_status"], "PENDING_APPROVAL")

    def test_03_invalid_input_is_rejected_cleanly(self) -> None:
        bad_inquiry = {
            "customer_name": "A",
            "company_name": "",
            "industry": "",
            "project_title": "Hi",
            "project_description": "Too short",
        }
        final_state = self.workflow.execute_inquiry(bad_inquiry)
        self.assertFalse(final_state["validation_passed"])
        self.assertGreater(len(final_state["validation_errors"]), 0)


if __name__ == "__main__":
    unittest.main()
