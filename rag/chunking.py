"""Stage 4 & 5 of RAG Pipeline: Semantic Document Chunking & Metadata Preservation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from rag.ingestion import KnowledgeDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    source_type: str
    category: str
    client_or_entity: str
    url: str
    chunk_index: int
    text: str
    tech_stack: List[str]
    verified_metric: str
    verified_public: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_documents(
    documents: List[KnowledgeDocument],
    chunk_size_words: int = 90,
    chunk_overlap_words: int = 20,
) -> List[DocumentChunk]:
    """Splits KnowledgeDocuments into overlapping semantic chunks while preserving full source attribution."""
    chunks: List[DocumentChunk] = []
    step = max(1, chunk_size_words - chunk_overlap_words)

    for doc in documents:
        # Prepend title, entity, and category context to every chunk for strong semantic grounding
        header_prefix = (
            f"[{doc.source_type.upper()}] {doc.title} | Entity: {doc.client_or_entity} | "
            f"Category: {doc.category} | Tech: {', '.join(doc.tech_stack)} | "
        )
        words = doc.raw_content.split()
        if len(words) <= chunk_size_words:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}#c0",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_type=doc.source_type,
                    category=doc.category,
                    client_or_entity=doc.client_or_entity,
                    url=doc.url,
                    chunk_index=0,
                    text=header_prefix + doc.raw_content,
                    tech_stack=list(doc.tech_stack),
                    verified_metric=doc.verified_metric,
                    verified_public=doc.verified_public,
                )
            )
        else:
            idx = 0
            for start in range(0, len(words), step):
                window = " ".join(words[start : start + chunk_size_words])
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.doc_id}#c{idx}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        source_type=doc.source_type,
                        category=doc.category,
                        client_or_entity=doc.client_or_entity,
                        url=doc.url,
                        chunk_index=idx,
                        text=header_prefix + window,
                        tech_stack=list(doc.tech_stack),
                        verified_metric=doc.verified_metric,
                        verified_public=doc.verified_public,
                    )
                )
                idx += 1
                if start + chunk_size_words >= len(words):
                    break
    return chunks
