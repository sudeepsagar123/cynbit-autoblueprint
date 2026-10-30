# Cynbit AutoBlueprint — Executed Testing & Evaluation Report

## 1. Automated Test Execution
The current audit ZIP was tested with `python -m pytest -q`. **22 / 22 tests passed**. The suite covers RAG ingestion, chunking, embedding normalization, source attribution, unmatched-query abstention, agent behavior, prompt-injection sanitization, 18-section blueprint generation, workflow revision loops, SQLite persistence, approval lifecycle, PDF export, database deletion, Gemini structured-output validation with a mocked transport, API failure fallback, and LangGraph wiring.

**Live API qualification:** `test_01_live_gemini_structured_output_end_to_end_with_mock` uses a mocked Gemini transport. It validates the structured-output contract and agent integration, but is not a live external Gemini API test. A separate live API smoke test must be run with a valid key before claiming live Gemini operation has been verified in deployment.

## 2. RAG Benchmark
- Benchmark queries: 6 (Healthcare, PropTech, Ticketing, Reputation, EdTech, FinTech)
- Embedding backend in the audit environment: `OFFLINE_HYBRID_TFIDF_SVD`
- Mean Precision@3: **83.33%** among references accepted by the production confidence threshold.
- Mean Ranked Recall@3: **91.67%** across the top-three unique-document ranking before confidence thresholding.
- Source URL attribution accuracy: **100.0%** among confidence-accepted references.

### Metric definitions
Production retrieval uses a confidence threshold and can return fewer than `k` documents when evidence is weak. Precision is calculated over the accepted references. Recall is calculated over the top-k ranked unique documents before thresholding, so the benchmark can measure candidate retrieval even when production confidence gating abstains. These metrics represent two different stages and must be reported with these definitions.

## 3. Benchmark Limitations
The benchmark contains six hand-authored queries and relevance labels over the bundled public Cynbit corpus. It is a small offline evaluation set, not a statistically representative production dataset. In particular, the FinTech query's highest-ranked candidate is currently `PROD-TENANTEASE`, while `CS-GLOBALPAY` appears in the top three. This indicates a ranking-quality issue worth improving before production use, despite the aggregate recall score.

## 4. Workflow Evaluation
The bounded revision-loop test simulates an initial architecture omission. The Critic flags the draft for revision and the next Architect pass restores the required coverage. This is a deterministic synthetic test, not a measured customer outcome.

## 5. Reproducibility
Run from the repository root:
```bash
python -m pytest -q
python - <<'PY'
from rag.retrieval import CynbitVectorRAGIndex
print(CynbitVectorRAGIndex().evaluate_benchmark(k=3))
PY
```
