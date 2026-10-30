"""Stage 7-10 of RAG Pipeline: Vector Storage, Semantic Retrieval, Grounded Context & Evaluation.

Uses `SemanticEmbeddingRouter` (Gemini `text-embedding-004` / `SentenceTransformers`) with automatic
fallback to the offline `HybridEmbeddingEngine` if live embedding API calls fail during indexing or query time.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rag.chunking import DocumentChunk, chunk_documents
from rag.embeddings import SemanticEmbeddingRouter
from rag.ingestion import KnowledgeDocument, load_verified_corpus


@dataclass
class RetrievedReference:
    doc_id: str
    chunk_id: str
    title: str
    client_or_entity: str
    source_type: str
    category: str
    url: str
    similarity_score: float
    excerpt: str
    tech_stack: List[str]
    verified_metric: str
    embedding_backend: str = "OFFLINE_HYBRID_TFIDF_SVD"
    is_verified_evidence: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CynbitVectorRAGIndex:
    """Vector Database supporting Genuine Semantic Embeddings + Offline Hybrid Fallback."""

    def __init__(
        self,
        min_similarity: Optional[float] = None,
        api_key: Optional[str] = None,
        remote_embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        configured_threshold = os.environ.get("RAG_SIMILARITY_THRESHOLD", "0.25")
        self.min_similarity = float(
            configured_threshold if min_similarity is None else min_similarity
        )
        self.documents: List[KnowledgeDocument] = load_verified_corpus()
        self.chunks: List[DocumentChunk] = chunk_documents(self.documents)
        self.embedder = SemanticEmbeddingRouter(api_key=api_key, remote_embed_fn=remote_embed_fn)
        self.chunk_embeddings: np.ndarray = np.array([])
        self.offline_chunk_embeddings: np.ndarray = np.array([])
        self._build_index()

    def _build_index(self) -> None:
        corpus_texts = [c.text for c in self.chunks]
        self.embedder.fit(corpus_texts)
        self.offline_chunk_embeddings = self.embedder.offline_engine.encode(corpus_texts)
        self.chunk_embeddings = self.embedder.encode(corpus_texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        source_types: Optional[List[str]] = None,
        min_similarity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieves top-k unique documents by semantic cosine similarity with strict threshold gating."""
        # Enforce the calibrated minimum relevance threshold.
        requested_threshold = (
            self.min_similarity if min_similarity is None else min_similarity
        )
        threshold = max(requested_threshold, 0.25)
        if not query or not query.strip():
            return {
                "references": [],
                "has_verified_matches": False,
                "embedding_backend": self.embedder.active_backend,
                "status_message": "No query text provided; no verified reference was retrieved.",
                "grounded_context": "No verified Cynbit reference found.",
            }

        q_emb = self.embedder.encode([query])[0]
        # Ensure dimensionality matches if a live API call fell back mid-flight
        active_matrix = (
            self.chunk_embeddings
            if self.chunk_embeddings.shape[1] == q_emb.shape[0]
            else self.offline_chunk_embeddings
        )

        if np.linalg.norm(q_emb) == 0:
            return {
                "references": [],
                "has_verified_matches": False,
                "embedding_backend": self.embedder.active_backend,
                "status_message": "No verified reference was found matching the submitted terminology.",
                "grounded_context": "No verified Cynbit reference found.",
            }

        raw_scores = active_matrix @ q_emb
        ranked_indices = np.argsort(raw_scores)[::-1]

        seen_docs = set()
        results: List[RetrievedReference] = []

        for idx in ranked_indices:
            score = float(raw_scores[int(idx)])
            if score < threshold:
                continue
            chunk = self.chunks[int(idx)]
            if source_types and chunk.source_type not in source_types:
                continue
            if chunk.doc_id in seen_docs:
                continue
            seen_docs.add(chunk.doc_id)
            results.append(
                RetrievedReference(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    client_or_entity=chunk.client_or_entity,
                    source_type=chunk.source_type,
                    category=chunk.category,
                    url=chunk.url,
                    similarity_score=round(score, 4),
                    excerpt=chunk.text,
                    tech_stack=list(chunk.tech_stack),
                    verified_metric=chunk.verified_metric,
                    embedding_backend=self.embedder.active_backend,
                    is_verified_evidence=True,
                )
            )
            if len(results) >= top_k:
                break

        if not results:
            return {
                "references": [],
                "has_verified_matches": False,
                "embedding_backend": self.embedder.active_backend,
                "status_message": (
                    "No verified reference was found in the public Cynbit knowledge base above the "
                    f"relevance threshold ({threshold:.2f}). Recommendations below are generated architectural "
                    "proposals rather than retrieved historical case studies."
                ),
                "grounded_context": "No verified Cynbit reference found above relevance threshold.",
            }

        context_lines = [
            f"=== RETRIEVED VERIFIED CYNBIT EVIDENCE (Backend: {self.embedder.active_backend}) ==="
        ]
        for i, ref in enumerate(results, start=1):
            context_lines.append(
                f"{i}. [{ref.doc_id}] {ref.title} ({ref.client_or_entity}) | "
                f"Similarity: {ref.similarity_score:.3f} | URL: {ref.url} | "
                f"Verified Metric: {ref.verified_metric}"
            )

        return {
            "references": [r.to_dict() for r in results],
            "has_verified_matches": True,
            "embedding_backend": self.embedder.active_backend,
            "status_message": (
                f"Retrieved {len(results)} verified Cynbit reference(s) via {self.embedder.active_backend}."
            ),
            "grounded_context": "\n".join(context_lines),
        }

    def evaluate_benchmark(self, k: int = 3) -> Dict[str, Any]:
        """Evaluate accepted-result precision and pre-threshold ranked recall at K."""
        benchmark_queries = [
            {
                "query": "AI patient symptom intake and clinical diagnostic triage workflow for multi-clinic healthcare hospital",
                "relevant_doc_ids": {"CS-WIZHEALTH", "CS-CAREPLUS"},
            },
            {
                "query": "Smart property rental unit tracking, tenant digital ID verification, and automated monthly rent payment reminders",
                "relevant_doc_ids": {"PROD-TENANTEASE", "CS-GLOBALPAY"},
            },
            {
                "query": "High-concurrency live event ticketing platform handling 50,000 simultaneous users with zero latency",
                "relevant_doc_ids": {"CS-EVENTO"},
            },
            {
                "query": "Multi-branch customer review synchronization across Google and Instagram with sentiment analysis and crisis detection",
                "relevant_doc_ids": {"PROD-REVIO", "CS-MARKETINGADDICT"},
            },
            {
                "query": "Adaptive coding curriculum, student skill-gap assessment, AI code quality evaluation, and mock interviews for universities",
                "relevant_doc_ids": {"PROD-STUDIO", "CS-ECONTENTPREP"},
            },
            {
                "query": "Cross-border FinTech payment routing with low latency ledger consistency and zero-trust security compliance",
                "relevant_doc_ids": {"CS-GLOBALPAY"},
            },
        ]

        precisions: List[float] = []
        recalls: List[float] = []
        attribution_valid_count = 0
        total_retrieved = 0
        per_query_results = []

        valid_urls = {d.url for d in self.documents}
        valid_titles = {d.title for d in self.documents}

        for item in benchmark_queries:
            # Precision measures the quality of references that pass the production
            # confidence gate. Recall measures whether relevant documents appear in
            # the top-k ranking before that gate; otherwise thresholding would hide
            # relevant candidates and conflate ranking quality with abstention.
            res = self.retrieve(item["query"], top_k=k)
            accepted_ids = [r["doc_id"] for r in res["references"]]

            query_vector = self.embedder.encode([item["query"]])[0]
            active_matrix = (
                self.chunk_embeddings
                if self.chunk_embeddings.shape[1] == query_vector.shape[0]
                else self.offline_chunk_embeddings
            )
            ranked_indices = np.argsort(active_matrix @ query_vector)[::-1]
            ranked_ids: List[str] = []
            seen_ranked = set()
            for idx in ranked_indices:
                chunk = self.chunks[int(idx)]
                if chunk.doc_id in seen_ranked:
                    continue
                seen_ranked.add(chunk.doc_id)
                ranked_ids.append(chunk.doc_id)
                if len(ranked_ids) >= k:
                    break

            rel_set = item["relevant_doc_ids"]
            accepted_hits = sum(1 for doc_id in accepted_ids if doc_id in rel_set)
            ranked_hits = sum(1 for doc_id in ranked_ids if doc_id in rel_set)
            p_at_k = accepted_hits / max(1, len(accepted_ids))
            r_at_k = ranked_hits / max(1, len(rel_set))
            precisions.append(p_at_k)
            recalls.append(r_at_k)

            for r in res["references"]:
                total_retrieved += 1
                if r["url"] in valid_urls and r["title"] in valid_titles and r["url"].startswith("https://www.cynbit.in"):
                    attribution_valid_count += 1

            per_query_results.append({
                "query": item["query"],
                "expected_relevant": sorted(list(rel_set)),
                "confidence_accepted_ids": accepted_ids,
                "ranked_top_k": ranked_ids,
                "accepted_hits": accepted_hits,
                "ranked_hits": ranked_hits,
                f"precision@{k}": round(p_at_k, 3),
                f"ranked_recall@{k}": round(r_at_k, 3),
            })

        return {
            "k": k,
            "embedding_backend": self.embedder.active_backend,
            "num_queries": len(benchmark_queries),
            f"mean_precision@{k}": round(float(np.mean(precisions)), 4),
            f"mean_recall@{k}": round(float(np.mean(recalls)), 4),
            f"mean_ranked_recall@{k}": round(float(np.mean(recalls)), 4),
            "precision_definition": "Precision among references accepted by the production confidence threshold.",
            "recall_definition": "Recall among the top-k ranked unique documents before confidence thresholding.",
            "source_attribution_accuracy": round(attribution_valid_count / max(1, total_retrieved), 4),
            "per_query_breakdown": per_query_results,
        }
