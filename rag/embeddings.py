"""Stage 6 of RAG Pipeline: Genuine Semantic Embeddings (Gemini API & SentenceTransformers)
with Automatic Offline Hybrid TF-IDF + SVD Fallback.

Supports three embedding backends in priority order:
1. `GEMINI_TEXT_EMBEDDING_004`: Calls Gemini's `text-embedding-004` model via `google-genai` SDK,
   `google-generativeai` SDK, or Gemini v1beta REST API (`batchEmbedContents`) when `GEMINI_API_KEY` is configured.
2. `SENTENCE_TRANSFORMERS_MINILM`: Uses `sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')`
   when `sentence-transformers` is installed locally.
3. `OFFLINE_HYBRID_TFIDF_SVD`: Deterministic sublinear TF-IDF + domain-synonym expansion + Truncated SVD
   dense vector engine retained as the offline fallback when neither live API credentials nor local
   neural weights are available.
"""

from __future__ import annotations

from collections import Counter
import json
import math
import os
import re
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over", "across",
    "want", "need", "our", "are", "still", "using", "like", "plus", "which", "before",
    "was", "were", "have", "has", "had", "will", "would", "could", "should", "their",
}

DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "medical": ["healthcare", "clinical", "patient", "triage", "diagnostic", "hospital", "clinic", "hipaa"],
    "healthcare": ["medical", "clinical", "patient", "triage", "vitals", "health"],
    "triage": ["diagnostic", "symptom", "routing", "intake", "medical"],
    "property": ["tenantease", "rental", "tenant", "rent", "building", "unit", "occupancy", "proptech", "lease"],
    "tenant": ["tenantease", "property", "rent", "payment", "unit", "lease"],
    "rent": ["tenantease", "payment", "tenant", "property", "collection"],
    "review": ["revio", "sentiment", "reputation", "feedback", "crisis", "branch", "rating"],
    "sentiment": ["revio", "review", "reputation", "feedback", "crisis"],
    "ticketing": ["evento", "concurrent", "entertainment", "event", "latency", "high_concurrency"],
    "erp": ["sofiasa", "legacy", "modernization", "migration", "monolith", "enterprise"],
    "payment": ["globalpay", "fintech", "cross_border", "transaction", "ledger", "rent"],
    "fintech": ["globalpay", "payment", "cross_border", "transaction", "zero_trust"],
    "learning": ["econtentprep", "studio", "bitelevate", "edtech", "student", "curriculum", "adaptive", "coding"],
    "edtech": ["econtentprep", "studio", "bitelevate", "learning", "student", "adaptive"],
    "iot": ["pragya", "telemetry", "edge", "remote", "monitoring", "vitals", "sensors"],
    "retail": ["urban", "wardrobe", "fashion", "ecommerce", "try_on", "branch"],
    "marketing": ["campaign", "seo", "crm", "leads", "organic", "funnel"],
}


def extract_semantic_tokens(text: str) -> List[str]:
    """Tokenizes text into unigrams, bigrams, and domain-synonym expansions for offline fallback."""
    raw_words = [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in STOP_WORDS
    ]
    expanded = list(raw_words)
    for w in raw_words:
        if w in DOMAIN_SYNONYMS:
            expanded.extend(DOMAIN_SYNONYMS[w])
    bigrams = [f"{raw_words[i]}_{raw_words[i+1]}" for i in range(len(raw_words) - 1)]
    return expanded + bigrams


class HybridEmbeddingEngine:
    """Offline fallback engine combining sublinear TF-IDF and non-negative SVD latent concepts."""

    def __init__(self, latent_dim: int = 32) -> None:
        self.latent_dim = latent_dim
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.vt_proj: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, texts: List[str]) -> "HybridEmbeddingEngine":
        tokenized = [extract_semantic_tokens(t) for t in texts]
        vocab_list = sorted({tok for doc in tokenized for tok in doc})
        self.vocab = {tok: idx for idx, tok in enumerate(vocab_list)}
        n_docs = len(texts)
        df = np.zeros(len(self.vocab), dtype=float)
        for doc in tokenized:
            for tok in set(doc):
                df[self.vocab[tok]] += 1.0
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        tfidf_mat = np.vstack([self._tfidf_vec(doc) for doc in tokenized])
        _, _, vt = np.linalg.svd(tfidf_mat, full_matrices=False)
        k = min(self.latent_dim, vt.shape[0])
        self.vt_proj = vt[:k, :].T
        self.is_fitted = True
        return self

    def _tfidf_vec(self, tokens: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=float)
        counts = Counter(tokens)
        for tok, cnt in counts.items():
            if tok in self.vocab:
                vec[self.vocab[tok]] = (1.0 + math.log(cnt)) * self.idf[self.vocab[tok]]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HybridEmbeddingEngine must be fitted before calling encode().")
        out_vecs = []
        for t in texts:
            tokens = extract_semantic_tokens(t)
            lexical = self._tfidf_vec(tokens)
            if np.linalg.norm(lexical) == 0:
                combined = np.zeros(len(self.vocab) + self.vt_proj.shape[1], dtype=float)
            else:
                latent = np.abs(lexical @ self.vt_proj)
                l_norm = np.linalg.norm(latent)
                if l_norm > 0:
                    latent = latent / l_norm
                combined = np.concatenate([0.85 * lexical, 0.15 * latent])
                c_norm = np.linalg.norm(combined)
                if c_norm > 0:
                    combined = combined / c_norm
            out_vecs.append(combined)
        return np.vstack(out_vecs)


class SemanticEmbeddingRouter:
    """Routes document and query embedding requests to Gemini Embeddings, SentenceTransformers, or Offline Fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        remote_embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "")
        ).strip()
        if self.api_key == "your_gemini_api_key_here":
            self.api_key = ""
        self.embedding_model = embedding_model or os.environ.get("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        self.remote_embed_fn = remote_embed_fn
        self.offline_engine = HybridEmbeddingEngine(latent_dim=16)
        self.st_model = None
        self.active_backend = "OFFLINE_HYBRID_TFIDF_SVD"
        self.last_fallback_reason: Optional[str] = None

        # Try loading SentenceTransformers if installed and no live Gemini key is set
        if not self.api_key and self.remote_embed_fn is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.active_backend = "SENTENCE_TRANSFORMERS_MINILM"
            except Exception:
                self.st_model = None

    def _normalize_rows(self, mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def _call_gemini_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.remote_embed_fn is not None:
            raw_lists = self.remote_embed_fn(texts)
            return self._normalize_rows(np.array(raw_lists, dtype=float))

        if not self.api_key:
            raise RuntimeError("No GEMINI_API_KEY configured for Gemini semantic embeddings.")

        # 1. Try google-genai SDK
        try:
            from google import genai  # type: ignore
            client = genai.Client(api_key=self.api_key)
            resp = client.models.embed_content(model=self.embedding_model, contents=texts)
            vecs = [e.values for e in resp.embeddings]
            return self._normalize_rows(np.array(vecs, dtype=float))
        except ImportError:
            pass

        # 2. Try google-generativeai SDK
        try:
            import google.generativeai as legacy_genai  # type: ignore
            legacy_genai.configure(api_key=self.api_key)
            model_path = self.embedding_model if self.embedding_model.startswith("models/") else f"models/{self.embedding_model}"
            res = legacy_genai.embed_content(model=model_path, content=texts)
            emb = res["embedding"]
            return self._normalize_rows(np.array(emb, dtype=float))
        except ImportError:
            pass

        # 3. Direct Gemini v1beta REST batchEmbedContents call
        model_path = self.embedding_model if self.embedding_model.startswith("models/") else f"models/{self.embedding_model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:batchEmbedContents?key={self.api_key}"
        requests_payload = [
            {"model": model_path, "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
        req = urllib.request.Request(
            url,
            data=json.dumps({"requests": requests_payload}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        vecs = [item["values"] for item in body.get("embeddings", [])]
        if len(vecs) != len(texts):
            raise RuntimeError("Gemini batchEmbedContents returned mismatched vector count.")
        return self._normalize_rows(np.array(vecs, dtype=float))

    def fit(self, corpus_texts: List[str]) -> "SemanticEmbeddingRouter":
        # Always fit the offline fallback engine so fallback is instantaneous if an API error occurs
        self.offline_engine.fit(corpus_texts)
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.api_key or self.remote_embed_fn is not None:
            try:
                vecs = self._call_gemini_embeddings(texts)
                self.active_backend = "GEMINI_TEXT_EMBEDDING_004"
                self.last_fallback_reason = None
                return vecs
            except Exception as exc:
                self.active_backend = "OFFLINE_HYBRID_TFIDF_SVD"
                self.last_fallback_reason = f"Gemini embedding API failed ({exc}); fell back to offline Hybrid TF-IDF/SVD."
                return self.offline_engine.encode(texts)

        if self.st_model is not None:
            try:
                vecs = self.st_model.encode(texts, normalize_embeddings=True)
                self.active_backend = "SENTENCE_TRANSFORMERS_MINILM"
                return np.array(vecs, dtype=float)
            except Exception as exc:
                self.active_backend = "OFFLINE_HYBRID_TFIDF_SVD"
                self.last_fallback_reason = f"SentenceTransformers failed ({exc}); fell back to offline engine."

        self.active_backend = "OFFLINE_HYBRID_TFIDF_SVD"
        return self.offline_engine.encode(texts)
