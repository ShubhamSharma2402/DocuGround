"""CLI entrypoint for the core RAG library (Phase 1 demo).

Usage:
    python -m app.rag_cli --pdf path/to/file.pdf --query "your question"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import structlog

from app.core.logging import configure_logging
from app.pipeline.chunk import chunk_pages
from app.pipeline.parse import parse_pdf
from app.services.embedder import FakeEmbedder, SentenceTransformerEmbedder
from app.services.llm import FakeLLMProvider, REFUSAL_MESSAGE
from app.services.query_rewriter import rewrite_query
from app.services.reranker import FakeReranker, CrossEncoderReranker
from app.services.retriever import HybridRetriever
from app.services.vector_store import ChunkPayload, VectorStore

log = structlog.get_logger(__name__)

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.txt"


def _build_system_prompt(chunks_text: list[str]) -> str:
    template = SYSTEM_PROMPT_PATH.read_text()
    numbered = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(chunks_text))
    return template.replace("{context}", numbered)


async def run_rag(
    pdf_path: Path,
    query: str,
    *,
    use_real_models: bool = False,
    doc_id: str = "cli-doc",
    user_id: str = "cli-user",
) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    configure_logging()

    print(f"\n[PARSE] Parsing: {pdf_path.name}")
    pages = parse_pdf(pdf_path)
    print(f"   -> {len(pages)} pages")

    chunks = chunk_pages(pages, doc_id=doc_id, user_id=user_id)
    print(f"   -> {len(chunks)} chunks")

    if use_real_models:
        embedder = SentenceTransformerEmbedder()
        reranker = CrossEncoderReranker()
        llm = FakeLLMProvider("[Real LLM not connected in CLI mode]")
    else:
        embedder = FakeEmbedder()
        reranker = FakeReranker()
        llm = FakeLLMProvider("Test answer based on the provided context.")

    print("\n[EMBED] Embedding chunks...")
    vectors = await embedder.aembed_texts([c.text for c in chunks])

    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams

    print("[INDEX] Indexing into in-memory Qdrant...")
    client = AsyncQdrantClient(":memory:")
    vec_size = len(vectors[0]) if vectors else 768
    await client.create_collection(
        collection_name="cli_test",
        vectors_config={"dense": VectorParams(size=vec_size, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    vs = VectorStore(client)
    # Override collection name for CLI test
    from app.core.config import get_settings
    original_collection = get_settings().qdrant_collection
    get_settings().__dict__["qdrant_collection"] = "cli_test"

    payloads = [
        ChunkPayload(doc_id=c.doc_id, user_id=c.user_id, page=c.page,
                     chunk_index=c.chunk_index, text=c.text)
        for c in chunks
    ]
    await vs.upsert_chunks(payloads, vectors)

    print(f"\n[SEARCH] Query: {query!r}")
    retriever = HybridRetriever(vs, embedder, reranker)

    result = await retriever.retrieve(
        query=query, user_id=user_id, doc_ids=[doc_id],
        dense_cutoff=0.0, reranker_cutoff=0.0,
    )

    # Restore collection name
    get_settings().__dict__["qdrant_collection"] = original_collection

    if not result.gate_passed:
        print(f"\n[GATE] Refusal gate: {result.refusal_reason}")
        print(f"\n{REFUSAL_MESSAGE}")
        return

    print(f"   -> {len(result.chunks)} chunks retrieved")
    for i, c in enumerate(result.chunks, 1):
        print(f"   [{i}] page={c.page} score={c.dense_score:.3f}: {c.text[:80]}...")

    system = _build_system_prompt([c.text for c in result.chunks])
    print("\n[LLM] Answer:\n")
    async for token in llm.stream_complete(system=system,
                                            messages=[{"role": "user", "content": query}],
                                            temperature=0.1):
        print(token, end="", flush=True)
    print("\n\n[OK] Done.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="DocuGround RAG CLI")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--real-models", action="store_true",
                        help="Use real SentenceTransformer + BGE models")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_rag(pdf_path=pdf_path, query=args.query, use_real_models=args.real_models))


if __name__ == "__main__":
    main()
