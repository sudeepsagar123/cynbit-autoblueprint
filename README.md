# Cynbit AutoBlueprint — Multi-Agent AI Discovery & Technical Blueprint Generation System

**Proof of Work — AI/ML Engineer Intern Technical Assessment ([Cynbit Technologies](https://www.cynbit.in/))**  
**Candidate:** Sudeep Sagar

## 1. Project Overview
**Cynbit AutoBlueprint** is a complete, multi-agent AI discovery and technical blueprint generation platform built for the AI/ML Engineer Intern technical assessment at [Cynbit Technologies](https://www.cynbit.in/). Directly implementing Opportunity 1 from `docs/Cynbit_AI_Opportunity_Report.pdf`, the system analyzes incoming project inquiries, retrieves verified public Cynbit references via RAG, synthesizes an 18-section Technical Blueprint, audits quality via a Critic Agent with a bounded revision loop, and enforces Human-in-the-Loop (HITL) approval before delivery.

## 2. Business Problem & Proposed Solution
Prospective enterprise clients submitting inquiries via [cynbit.in/contact](https://www.cynbit.in/contact) expect a tailored architectural blueprint within Cynbit's **24-Hour Response Sync** window. Manually parsing unstructured requirements, cross-referencing past case studies, designing database/API schemas, and estimating phased timelines requires significant architect effort.

**Cynbit AutoBlueprint** automates initial discovery and blueprint drafting through a 4-stage stateful pipeline (**Researcher Agent → Architect Agent → Critic Agent → Human-in-the-Loop Approval Center**) grounded in verified public Cynbit documentation. This is a proposed prototype using only publicly accessible Cynbit data and synthetic demonstration inquiries.

## 3. Core Architecture & Dual-Mode Execution
1. **Genuine Gemini API Structured Output Mode (`services/llm_client.py`)**:
   - When `GEMINI_API_KEY` is configured, the Researcher, Architect, and Critic agents invoke Google Gemini (`gemini-3.8-flash`) via the `google-genai` SDK, `google-generativeai` SDK, or direct Gemini v1beta REST API (`response_mime_type="application/json"`) and validate responses strictly against Pydantic schemas (`ResearcherLLMOutput`, `ArchitectLLMOutput`, `CriticLLMOutput`).
   - When `GEMINI_API_KEY` is omitted or an API error occurs, the system automatically runs in a clearly labeled **`OFFLINE_DEMO_MODE`** deterministic fallback so every feature, revision loop, and test works offline without paid credentials.
2. **LangGraph StateGraph Orchestration (`orchestration/graph.py`, `orchestration/workflow.py`)**:
   - Uses `langgraph.graph.StateGraph(WorkflowStateTypedDict)` when `langgraph` is installed, with an API-compatible `FallbackStateGraph` for minimal offline environments.
   - Implements a bounded Critic → Architect revision loop (`max_retries=2`) to prevent infinite agent execution.
3. **Semantic & Hybrid RAG Pipeline (`rag/ingestion.py`, `rag/chunking.py`, `rag/embeddings.py`, `rag/retrieval.py`)**:
   - Supports live Gemini semantic embeddings (`text-embedding-004`) and local neural embeddings (`sentence-transformers` `all-MiniLM-L6-v2`), retaining the offline `HybridEmbeddingEngine` (sublinear TF-IDF + domain synonyms + Truncated SVD) as a fallback.
   - Indexes **16 strictly verified public records** from `https://www.cynbit.in/`, `https://www.cynbit.in/services`, `https://www.cynbit.in/products`, `https://www.cynbit.in/case-studies`, `https://www.cynbit.in/bitelevate/paths`, `https://www.cynbit.in/impact`, and `https://www.cynbit.in/blog/ai-agents-business-operations`. Does not fabricate confidential per-case-study stacks or metrics.
4. **Human-in-the-Loop (HITL) Governance (`agents/approval.py`, `database/db.py`, `services/pdf_export.py`)**:
   - Stages blueprints as `PENDING_APPROVAL` in SQLite (`data/autoblueprint.db`).
   - All seeded demonstration approvals use clearly labeled synthetic reviewer identifiers (`Synthetic Demo Reviewer A` / `Synthetic Demo Reviewer B`) and never claim real approvals by Cynbit leadership.
   - Exports multi-page 18-section PDF blueprints via ReportLab (`services/pdf_export.py`).

## 4. Repository Structure
```text
cynbit-autoblueprint/
├── app.py                              # 6-Page Streamlit SaaS Application & CLI seed/verification runner
├── requirements.txt                    # Full dependencies (numpy, pydantic, streamlit, google-genai, langgraph, etc.)
├── README.md                           # Installation, configuration, execution, testing & deployment guide
├── .env.example                        # Environment variable template (GEMINI_API_KEY, model names, DB path)
├── .gitignore                          # Git ignore rules
├── agents/
│   ├── researcher.py                   # Agent 1: Input sanitization, requirement extraction & RAG retrieval
│   ├── architect.py                    # Agent 2: 18-section blueprint & Mermaid diagram synthesis
│   ├── critic.py                       # Agent 3: Requirement coverage, security & consistency audit
│   └── approval.py                     # Agent 4: Human-in-the-Loop (HITL) approval state manager
├── orchestration/
│   ├── state.py                        # WorkflowStateTypedDict & Pydantic state schemas
│   ├── graph.py                        # LangGraph StateGraph integration + OfflineFallbackStateGraph
│   └── workflow.py                     # End-to-end 4-stage pipeline with bounded revision loop
├── rag/
│   ├── ingestion.py                    # 16 strictly verified public Cynbit documents & HTML cleaner
│   ├── chunking.py                     # Sliding-window semantic chunker with metadata preservation
│   ├── embeddings.py                   # SemanticEmbeddingRouter (Gemini / SentenceTransformers / Offline SVD)
│   └── retrieval.py                    # Cosine vector search, threshold gating & benchmark evaluator
├── prompts/
│   ├── researcher.txt                  # System prompt for Researcher Agent
│   ├── architect.txt                   # System prompt for Architect Agent
│   └── critic.txt                      # System prompt for Critic Agent
├── database/
│   └── db.py                           # SQLite persistence & real-time dashboard metric aggregation
├── services/
│   ├── llm_client.py                   # Gemini API structured output client & Pydantic validator
│   ├── blueprint_generator.py          # 18-section Markdown formatter
│   └── pdf_export.py                   # ReportLab multi-page PDF exporter
├── data/
│   ├── sample_inquiries.json           # 4 synthetic demonstration inquiries
│   └── Sample_Approved_Blueprint.pdf   # Curated 18-section PDF blueprint example (local DB is generated at runtime)
├── tests/
│   ├── test_rag.py                     # 6 RAG ingestion, chunking, embedding & attribution tests
│   ├── test_agents.py                  # 3 agent extraction, prompt-injection & 18-section tests
│   ├── test_workflow.py                # 3 end-to-end workflow & bounded revision loop tests
│   ├── test_approval.py                # 1 HITL lifecycle, SQLite persistence & PDF export test
│   └── test_llm_integration.py         # 6 Gemini mock, API failure, validation, embedding & LangGraph tests
├── screenshots/                        # 6 captured screenshots of all application pages
└── docs/
    ├── architecture.md                 # System architecture specification
    ├── evaluation.md                   # Executed test & RAG benchmark evaluation report
    ├── Cynbit_AI_Opportunity_Report.pdf# Deliverable 1: AI Opportunity Report
    └── Cynbit_Technical_Approach_Report.pdf # Deliverable 3: Updated Technical Approach Report
```

## 5. Installation & Configuration
1. **Clone the repository and install dependencies**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cynbit-autoblueprint.git
   cd cynbit-autoblueprint
   pip install -r requirements.txt
   ```
2. **Configure environment variables (Optional for Live Gemini Mode)**:
   ```bash
   cp .env.example .env
   # Edit .env and set GEMINI_API_KEY=your_actual_gemini_api_key
   ```
   - With `GEMINI_API_KEY` set: runs in **`LIVE_GEMINI_API`** mode using the configured `GEMINI_MODEL` (default: `gemini-3.8-flash`) and `GEMINI_EMBEDDING_MODEL` (default: `text-embedding-004`).
   - Without `GEMINI_API_KEY`: runs automatically in **`OFFLINE_DEMO_MODE`** using the deterministic engine and `OFFLINE_HYBRID_TFIDF_SVD` embeddings.

## 6. Application Execution
1. **Seed Database & Verify PDF Export via CLI**:
   ```bash
   python3 app.py --seed-and-verify
   ```
2. **Launch the 6-Page Streamlit SaaS Application**:
   ```bash
   streamlit run app.py
   ```

## 7. Running Automated Tests (`22 / 22 Passed` )
Execute the full unit and integration test suite via `unittest` or `pytest`:
```bash
python3 -m unittest discover -s tests -t . -v
# or
pytest -v
```

## 8. Executed Evaluation Summary
- **Automated Test Suite**: `22 / 22 tests PASSED` in the uploaded audit environment.
- **RAG Mean Precision@3**: `83.33%` among references accepted by the production confidence threshold.
- **RAG Mean Ranked Recall@3**: `91.67%` across the top-three unique-document ranking before confidence thresholding.
- **Source URL Attribution Accuracy**: `100.0%` among confidence-accepted retrieved references; each is matched to the verified public corpus.
- **Bounded Revision Loop Recovery**: `100.0%` in the synthetic test scenario (a simulated initial omission triggers `NEEDS_REVISION`, then the next pass restores full requirement coverage).

**Metric note:** Precision and recall intentionally describe different stages. Production retrieval abstains on low-confidence matches; the benchmark therefore measures precision on accepted results and recall on the unthresholded top-k ranking. These figures should not be presented as a single conventional precision/recall pair without this definition. The automated Gemini workflow test uses a mocked transport; it verifies schema and agent integration, not a live external Gemini call.


## 9. Deploy to Streamlit Community Cloud
1. Push this repository to a public GitHub repository (or a private repository supported by your Streamlit account).
2. In Streamlit Community Cloud, create an app from that repository and set the main file path to `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   GEMINI_MODEL = "gemini-3.8-flash"
   GEMINI_EMBEDDING_MODEL = "text-embedding-004"
   ```
4. Save and reboot the app. The application reads these values from `st.secrets`; no API key should be committed to GitHub.
5. If no key is configured, the app starts in the clearly labeled offline demo mode.

The SQLite database on Streamlit Community Cloud is local to the app container and may not be durable across restarts or redeployments. This prototype uses SQLite for demonstration; use managed persistent storage before treating it as a production multi-user service.

## 10. Security and Prototype Scope
- Never commit `.env`, `.streamlit/secrets.toml`, API keys, or local SQLite files.
- The app uses a local SQLite database and is intended as an assessment prototype, not a hardened production service.
- Human approval is an application workflow gate; it does not provide user authentication or role-based access control by itself.
- Estimates and architecture recommendations are preliminary and must be validated by a human architect with the customer.
