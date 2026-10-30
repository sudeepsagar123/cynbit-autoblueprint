"""Automated Tests for RAG Pipeline:
- Document ingestion & HTML/text cleaning
- Semantic chunking & metadata preservation
- Embedding generation & vector normalization
- Case-study relevance & source attribution
- Unmatched query handling (anti-hallucination check)
"""

import unittest

from rag.chunking import chunk_documents
from rag.embeddings import HybridEmbeddingEngine
from rag.ingestion import clean_html_and_text, load_verified_corpus
from rag.retrieval import CynbitVectorRAGIndex


class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = load_verified_corpus()
        cls.rag = CynbitVectorRAGIndex(min_similarity=0.08)

    def test_01_document_ingestion_and_cleaning(self) -> None:
        self.assertGreaterEqual(len(self.corpus), 15)
        raw_html = "<div>Cynbit &amp; Partners   <b>AI Triage</b>\n\nSystem</div>"
        self.assertEqual(clean_html_and_text(raw_html), "Cynbit & Partners AI Triage System")
        for doc in self.corpus:
            self.assertTrue(doc.url.startswith("https://www.cynbit.in"))
            self.assertTrue(doc.verified_public)

    def test_02_chunking_preserves_metadata(self) -> None:
        chunks = chunk_documents(self.corpus, chunk_size_words=50, chunk_overlap_words=10)
        self.assertGreaterEqual(len(chunks), len(self.corpus))
        for ch in chunks:
            self.assertTrue(ch.chunk_id.startswith(ch.doc_id))
            self.assertTrue(ch.url.startswith("https://www.cynbit.in"))
            self.assertIn(ch.client_or_entity, ch.text)

    def test_03_embeddings_are_unit_normalized(self) -> None:
        engine = HybridEmbeddingEngine(latent_dim=8).fit([d.raw_content for d in self.corpus])
        vecs = engine.encode(["AI clinical triage for hospital patients", "Smart property rent management"])
        self.assertEqual(vecs.shape[0], 2)
        for v in vecs:
            self.assertAlmostEqual(float((v @ v) ** 0.5), 1.0, places=5)

    def test_04_case_study_relevance_and_attribution(self) -> None:
        # Query matching WizHealth
        res_health = self.rag.retrieve(
            "AI diagnostic triage assistant and patient symptom intake for medical clinics",
            top_k=3,
            source_types=["case_study"],
        )
        self.assertTrue(res_health["has_verified_matches"])
        top_ids = [r["doc_id"] for r in res_health["references"]]
        self.assertIn("CS-WIZHEALTH", top_ids)

        # Query matching Evento
        res_event = self.rag.retrieve(
            "Scaling real-time ticketing platform for 50k concurrent users in entertainment",
            top_k=3,
            source_types=["case_study"],
        )
        self.assertEqual(res_event["references"][0]["doc_id"], "CS-EVENTO")

        # Query matching GlobalPay
        res_pay = self.rag.retrieve(
            "Cross-border FinTech payments routing with zero-trust compliance",
            top_k=3,
            source_types=["case_study"],
        )
        self.assertEqual(res_pay["references"][0]["doc_id"], "CS-GLOBALPAY")

    def test_05_unmatched_query_communicates_clearly(self) -> None:
        res_none = self.rag.retrieve("zzzzxxxxqqqq11112222nonexistenttoken", top_k=3)
        self.assertFalse(res_none["has_verified_matches"])
        self.assertEqual(len(res_none["references"]), 0)
        self.assertIn("No verified reference was found", res_none["status_message"])

    def test_06_benchmark_precision_and_recall(self) -> None:
        eval_metrics = self.rag.evaluate_benchmark(k=3)
        self.assertGreaterEqual(eval_metrics["mean_precision@3"], 0.65)
        self.assertGreaterEqual(eval_metrics["mean_recall@3"], 0.75)
        self.assertEqual(eval_metrics["source_attribution_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
