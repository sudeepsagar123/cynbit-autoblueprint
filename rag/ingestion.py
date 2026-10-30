"""Stage 1-3 & 5 of RAG Pipeline: Strictly Verified Cynbit Public Document Ingestion.

IMPORTANT GROUNDING & ANTI-FABRICATION POLICY:
Every document in this corpus contains ONLY verbatim or directly verifiable facts from public
pages on https://www.cynbit.in/:
- https://www.cynbit.in/ (Homepage, 6-Stage Architectural Lifecycle, Client Quotes)
- https://www.cynbit.in/services (Services Hub & Company-Wide Production Tech Stack)
- https://www.cynbit.in/products (Proprietary SaaS Suite: Cynbit Studio, TenantEase, Revio)
- https://www.cynbit.in/case-studies (Public Case Study Cards: Client, Category, and Title)
- https://www.cynbit.in/bitelevate/paths (BitElevate 16-Dimension Adaptive Learning Paths)
- https://www.cynbit.in/impact (Social Impact: BitElevate 2.0 & Manav Seva Trust)
- https://www.cynbit.in/contact (Contact Hub & 24-Hour Response Sync Blueprint Inquiry)
- https://www.cynbit.in/blog/ai-agents-business-operations (MAS & HITL Governance Blog)

Because https://www.cynbit.in/case-studies publishes case-study titles, client names, categories,
and aggregate/featured quotes without disclosing confidential per-project internal tech stacks,
case-study records below do NOT fabricate per-case-study tech stacks. Company-wide stack
capabilities are attributed strictly to https://www.cynbit.in/services.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import html
import re
from typing import Any, Dict, List


@dataclass
class KnowledgeDocument:
    doc_id: str
    title: str
    source_type: str  # "case_study" | "saas_product" | "service_capability" | "engineering_article"
    category: str
    client_or_entity: str
    url: str
    raw_content: str
    tech_stack: List[str] = field(default_factory=list)
    verified_metric: str = ""
    verified_public: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clean_html_and_text(raw_text: str) -> str:
    """Strips HTML tags, decodes entities, removes control characters, and normalizes whitespace."""
    if not raw_text:
        return ""
    unescaped = html.unescape(raw_text)
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    normalized = re.sub(r"\s+", " ", no_tags).strip()
    return normalized


VERIFIED_CYNBIT_CORPUS: List[KnowledgeDocument] = [
    # --- VERIFIED PUBLIC CASE STUDIES ON https://www.cynbit.in/case-studies & https://www.cynbit.in/ ---
    KnowledgeDocument(
        doc_id="CS-WIZHEALTH",
        title="AI Diagnostic Triage: Enhancing Medical Efficiency",
        source_type="case_study",
        category="Software Development",
        client_or_entity="WizHealth",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies: 'Cynbit Technologies Case Study: "
            "AI Diagnostic Triage: Enhancing Medical Efficiency. for WizHealth — Software Development.' "
            "Listed under Cynbit's Success Blueprints library ('From Challenge to Scalable Solutions')."
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies (internal project stack not publicly disclosed)"],
        verified_metric="Published on cynbit.in/case-studies (Page header benchmarks: 99.99% Uptime, +140% Growth)",
    ),
    KnowledgeDocument(
        doc_id="CS-PRAGYA",
        title="IoT Remote Patient Monitoring for Rural Health",
        source_type="case_study",
        category="Software Development",
        client_or_entity="Pragya Healths",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies: 'Cynbit Technologies Case Study: "
            "IoT Remote Patient Monitoring for Rural Health. for Pragya Healths — Software Development.' "
            "Listed under Cynbit's Success Blueprints library."
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies (internal project stack not publicly disclosed)"],
        verified_metric="Published on cynbit.in/case-studies (IoT Remote Patient Monitoring for Rural Health)",
    ),
    KnowledgeDocument(
        doc_id="CS-EVENTO",
        title="Scaling Real-time Ticketing for Major Entertainment",
        source_type="case_study",
        category="Software Development & Enterprise SaaS",
        client_or_entity="Evento",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies and https://www.cynbit.in/: "
            "'Cynbit Technologies Case Study: Scaling Real-time Ticketing for Major Entertainment. for Evento — "
            "Software Development.' Verbatim client testimonial on https://www.cynbit.in/: '50k+ Users: The AI "
            "automation handled 50k+ users without a single flicker of latency. True engineering rigor. — Evento, Enterprise SaaS.'"
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies & cynbit.in (AI automation & high-concurrency ticketing)"],
        verified_metric="50k+ Users handled without latency flicker (Verbatim quote on cynbit.in)",
    ),
    KnowledgeDocument(
        doc_id="CS-SOFIASA",
        title="Modernizing Legacy ERP for the Digital Era",
        source_type="case_study",
        category="Software Development",
        client_or_entity="Sofiasa",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies: 'Cynbit Technologies Case Study: "
            "Modernizing Legacy ERP for the Digital Era. for Sofiasa — Software Development.' "
            "Listed under Cynbit's Success Blueprints library."
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies (internal project stack not publicly disclosed)"],
        verified_metric="Published on cynbit.in/case-studies (Legacy ERP Modernization)",
    ),
    KnowledgeDocument(
        doc_id="CS-GLOBALPAY",
        title="Revolutionizing Cross-Border Payments for the Next Billion Users",
        source_type="case_study",
        category="Software Development",
        client_or_entity="GlobalPay",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies: 'Cynbit Technologies Case Study: "
            "Revolutionizing cross-border payments for the next billion users. for GlobalPay — Software Development.' "
            "Listed under Cynbit's Success Blueprints library."
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies (internal project stack not publicly disclosed)"],
        verified_metric="Published on cynbit.in/case-studies (Cross-Border Payments)",
    ),
    KnowledgeDocument(
        doc_id="CS-ECONTENTPREP",
        title="AI-Driven Adaptive Learning Pathways for EdTech",
        source_type="case_study",
        category="Software Development",
        client_or_entity="EContentPrep",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies: 'Cynbit Technologies Case Study: "
            "AI-Driven Adaptive Learning Pathways for EdTech. for EContentPrep — Software Development.' "
            "Listed under Cynbit's Success Blueprints library."
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies (internal project stack not publicly disclosed)"],
        verified_metric="Published on cynbit.in/case-studies (AI-Driven Adaptive Learning Pathways)",
    ),
    KnowledgeDocument(
        doc_id="CS-URBANWARDROBE",
        title="AR Virtual Try-On: Redefining the Fashion Experience",
        source_type="case_study",
        category="Software Development & Global Retail",
        client_or_entity="Urban Wardrobe",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies and https://www.cynbit.in/: "
            "'Cynbit Technologies Case Study: AR Virtual Try-On: Redefining the Fashion Experience. — Urban Wardrobe, "
            "Software Development.' Verbatim client testimonial on https://www.cynbit.in/: '99.9% Uptime: Cynbit's "
            "cloud architecture gave us the global stability we needed to expand our digital footprint rapidly. — "
            "Urban Wardrobe, Global Retail.'"
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies & cynbit.in (Cloud architecture & AR Virtual Try-On)"],
        verified_metric="99.9% Uptime (Verbatim quote on cynbit.in)",
    ),
    KnowledgeDocument(
        doc_id="CS-MARKETINGADDICT",
        title="AI-Powered Campaign Optimization for Global Marketing",
        source_type="case_study",
        category="Digital Marketing",
        client_or_entity="Marketing Addict",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies and https://www.cynbit.in/: "
            "'Cynbit Technologies Case Study: AI-Powered Campaign Optimization for Global Marketing. for Marketing Addict — "
            "Digital Marketing.' Verbatim client testimonial on https://www.cynbit.in/: '+40% Growth: Cynbit transformed "
            "our technical infrastructure, resulting in a 40% increase in organic leads. Their approach is revolutionary. — "
            "Marketing Addict, Strategic Partner.'"
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies & cynbit.in (AI Campaign Optimization & Technical SEO)"],
        verified_metric="+40% Increase in Organic Leads (Verbatim quote on cynbit.in)",
    ),
    KnowledgeDocument(
        doc_id="CS-CAREPLUS",
        title="Migrating Critical Health Data to the Cloud with Zero Downtime",
        source_type="case_study",
        category="Cloud Strategy",
        client_or_entity="CarePlus Health",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies: 'Cynbit Technologies Case Study: "
            "Migrating critical health data to the cloud with zero downtime. — CarePlus Health, Cloud Strategy.'"
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies (Cloud Strategy)"],
        verified_metric="Zero-Downtime Critical Health Data Cloud Migration (cynbit.in/case-studies)",
    ),
    KnowledgeDocument(
        doc_id="CS-WILDBASIL",
        title="Blockchain-Lite Transparency for Sustainable Food",
        source_type="case_study",
        category="Software Development",
        client_or_entity="Wild basil",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies: 'Cynbit Technologies Case Study: "
            "Blockchain-Lite Transparency for Sustainable Food. for Wild basil — Software Development.'"
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies (internal project stack not publicly disclosed)"],
        verified_metric="Published on cynbit.in/case-studies (Sustainable Food Transparency)",
    ),
    KnowledgeDocument(
        doc_id="CS-CLOUDYTALES",
        title="Real-time Collaborative Engine for Digital Storytellers",
        source_type="case_study",
        category="Cloud Strategy",
        client_or_entity="Cloudy Tales",
        url="https://www.cynbit.in/case-studies",
        raw_content=(
            "Public Case Study on https://www.cynbit.in/case-studies and https://www.cynbit.in/: "
            "'Cynbit Technologies Case Study: Real-time Collaborative Engine for Digital Storytellers. for Cloudy Tales — "
            "Cloud Strategy.' Verbatim client testimonial on https://www.cynbit.in/: '100% Success: From the first "
            "blueprint to final deployment, the Cynbit team acted as true architects of our story. — Cloudy Tales, Creative Studio.'"
        ),
        tech_stack=["Verified public listing on cynbit.in/case-studies & cynbit.in (Cloud Strategy)"],
        verified_metric="100% Success (Verbatim quote on cynbit.in)",
    ),
    # --- PROPRIETARY SAAS SUITE ON https://www.cynbit.in/products ---
    KnowledgeDocument(
        doc_id="PROD-TENANTEASE",
        title="TenantEase: Smart Property & Tenant Management Platform",
        source_type="saas_product",
        category="PropTech & Smart Property Management",
        client_or_entity="TenantEase (Cynbit Proprietary SaaS)",
        url="https://www.cynbit.in/products",
        raw_content=(
            "Public Product Description on https://www.cynbit.in/products: 'TenantEase — Smart Property & Tenant "
            "Management Platform. Replaces manual chaos with a single smart system. Manage properties, track tenants, "
            "and automate rent collection in one centralized dashboard. Features Rent & Payment Intelligence (Auto "
            "Reminders for system-driven chasing, Payment Audit for full digital history, Collection efficiency 100% "
            "Target), Unit Tracking (add buildings and rooms, track availability instantly), Tenant Profiles "
            "(structured digital profiles with ID verification, contacts, and historical data), and Business Dashboard "
            "(occupancy rates and revenue trends).'"
        ),
        tech_stack=["Proprietary SaaS on cynbit.in/products"],
        verified_metric="100% Collection Efficiency Target (cynbit.in/products)",
    ),
    KnowledgeDocument(
        doc_id="PROD-REVIO",
        title="Revio: The Definitive Review Management System",
        source_type="saas_product",
        category="Multi-Location Reputation AI & Sentiment Hub",
        client_or_entity="Revio (Cynbit Proprietary SaaS)",
        url="https://www.cynbit.in/products",
        raw_content=(
            "Public Product Description on https://www.cynbit.in/products: 'Revio — AI Reputation & Sentiment Hub. "
            "Revio is an AI-first reputation engine built to automate customer feedback management, sync reviews "
            "across platforms, and protect brand prestige through intelligent sentiment analysis. Features Cross-Platform "
            "Sync (Google, social media, industry sites), AI Response Engine, Sentiment Intelligence (Sentiment Score 94%), "
            "Crisis Detection, Multi-Location Control for brands with multiple branches, and Role-Based Access Hierarchy "
            "(Admin for System Governance, Manager for Brand Prestige, Advisor for Branch Operations).'"
        ),
        tech_stack=["Proprietary AI SaaS on cynbit.in/products"],
        verified_metric="94% Sentiment Score displayed on cynbit.in/products",
    ),
    KnowledgeDocument(
        doc_id="PROD-STUDIO",
        title="Cynbit Studio & BitElevate: AI-Powered Career Growth Lab",
        source_type="saas_product",
        category="EdTech, Coding Assessment & Career Intelligence",
        client_or_entity="Cynbit Studio & BitElevate",
        url="https://www.cynbit.in/products",
        raw_content=(
            "Public Product Description on https://www.cynbit.in/products and https://www.cynbit.in/bitelevate/paths: "
            "'Cynbit Studio helps students and developers move from learning to employability through adaptive coding "
            "paths, AI-evaluated missions, and institutional-grade career tools: Placement Assessment, AI Coding Missions, "
            "AI Code Evaluation (feedback on code quality instead of simple output matching), AI Mentor Support, "
            "AI Mock Interviews, AI Resume Engine, Job Matcher Analysis, Legacy Profiles, and Skill Radar.' "
            "BitElevate (https://www.cynbit.in/bitelevate/paths) covers 16 core dimensions and BitElevate 2.0 "
            "(https://www.cynbit.in/impact) addresses student mental wellness and Manav Seva Trust rural IT literacy."
        ),
        tech_stack=["Proprietary EdTech SaaS on cynbit.in/products & cynbit.in/bitelevate/paths"],
        verified_metric="1000+ Certified Architects (cynbit.in) | 16 Core Learning Dimensions (cynbit.in/bitelevate/paths)",
    ),
    # --- CYNBIT SERVICES HUB & TECH STACK ON https://www.cynbit.in/services ---
    KnowledgeDocument(
        doc_id="SVC-STACK",
        title="Cynbit Services Hub: AI Automation, Custom Software & Architectural Tech Stack",
        source_type="service_capability",
        category="Enterprise Software & AI Automation Architecture",
        client_or_entity="Cynbit Technologies Services Hub",
        url="https://www.cynbit.in/services",
        raw_content=(
            "Public Service & Stack Description on https://www.cynbit.in/services: "
            "1. AI Automation Suite: Autonomous agents (lead management & coordination), LLM automation (manual data "
            "entry & document handling), AI Voice Agents, and AI stack integration. "
            "2. Custom IT Services: Full-stack web & mobile dev, backend systems (99.9% uptime), Cloud infrastructure "
            "(AWS & Azure zero-trust security), and Legacy migration (monoliths to distributed systems). "
            "3. Digital Marketing: Performance ads, Technical SEO, CRM automation, and Funnel optimization. "
            "4. Published Architectural Tech Stack — Languages: TypeScript, Python, Go, Rust, Java, C++. "
            "Frameworks: React, Next.js, NestJS, FastAPI, Django, Tailwind. "
            "AI & Data: OpenAI, LangChain, PyTorch, TensorFlow, Pinecone, Spark. "
            "Cloud & Ops: AWS, Azure, Docker, Kubernetes, Terraform, CI/CD. "
            "Databases: PostgreSQL, MongoDB, Redis, Prisma, Supabase, Neo4j."
        ),
        tech_stack=[
            "TypeScript", "Python", "Go", "Rust", "Java", "C++", "React", "Next.js", "NestJS",
            "FastAPI", "Django", "Tailwind", "OpenAI", "LangChain", "PyTorch", "TensorFlow",
            "Pinecone", "Spark", "AWS", "Azure", "Docker", "Kubernetes", "Terraform", "CI/CD",
            "PostgreSQL", "MongoDB", "Redis", "Prisma", "Supabase", "Neo4j",
        ],
        verified_metric="99.9% Uptime | 98% Retention | 3x ROI Uplift (cynbit.in)",
    ),
    # --- CYNBIT ENGINEERING BLOG ON https://www.cynbit.in/blog/ai-agents-business-operations ---
    KnowledgeDocument(
        doc_id="BLOG-MAS-HITL",
        title="How AI Agents Are Transforming Business Operations in 2026",
        source_type="engineering_article",
        category="Agentic AI Governance & Multi-Agent Systems",
        client_or_entity="Cynbit Engineering Core",
        url="https://www.cynbit.in/blog/ai-agents-business-operations",
        raw_content=(
            "Public Engineering Article on https://www.cynbit.in/blog/ai-agents-business-operations (Published Oct 13, 2025): "
            "Describes Cynbit's Human-in-the-Loop (HITL) architecture where agents operate in a sandbox to perform research "
            "and draft actions, while high-stakes operations (like spending money or deleting data) require explicit approval "
            "from a human operator. Details Multi-Agent Systems (MAS) with an Architect Agent to plan the project, a "
            "Researcher Agent to gather data, a Writer Agent to create the report, and a Critic Agent to check for errors, "
            "and highlights three risks to avoid: Over-Reliance (hallucinations), Opaque Black-Box Logic, and Infinite Loops."
        ),
        tech_stack=["Python", "LangChain", "OpenAI"],
        verified_metric="HITL Sandbox Governance & 4-Agent MAS Pattern (cynbit.in/blog/ai-agents-business-operations)",
    ),
]


def load_verified_corpus() -> List[KnowledgeDocument]:
    """Returns cleaned, strictly verified Cynbit knowledge base documents."""
    cleaned_docs: List[KnowledgeDocument] = []
    for doc in VERIFIED_CYNBIT_CORPUS:
        cleaned = KnowledgeDocument(
            doc_id=doc.doc_id,
            title=clean_html_and_text(doc.title),
            source_type=doc.source_type,
            category=clean_html_and_text(doc.category),
            client_or_entity=clean_html_and_text(doc.client_or_entity),
            url=doc.url.strip(),
            raw_content=clean_html_and_text(doc.raw_content),
            tech_stack=list(doc.tech_stack),
            verified_metric=clean_html_and_text(doc.verified_metric),
            verified_public=True,
        )
        cleaned_docs.append(cleaned)
    return cleaned_docs
