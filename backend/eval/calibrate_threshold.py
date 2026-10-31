"""Threshold calibration script for DocuGround.

Sweeps dense similarity cutoffs and cross-encoder reranker cutoffs across
the 50 labeled QA dataset to identify optimal separation thresholds that maximize:
  Score = Correct Refusal Rate + (1 - False Refusal Rate)
subject to False Refusal Rate <= 0.05.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, SparseVectorParams, VectorParams

from app.core.logging import configure_logging, get_logger
from app.pipeline.chunk import chunk_pages
from app.pipeline.parse import parse_pdf
from app.services.embedder import Embedder, FakeEmbedder, SentenceTransformerEmbedder
from app.services.reranker import CrossEncoderReranker, FakeReranker, Reranker
from app.services.retriever import reciprocal_rank_fusion
from app.services.vector_store import ChunkPayload, VectorStore

log = get_logger(__name__)

DATASET_DIR = Path(__file__).parent / "dataset"
DEFAULT_PDF = DATASET_DIR / "docuground_sample_report.pdf"
DEFAULT_QA = DATASET_DIR / "qa_pairs.jsonl"
REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class QuestionScoreRecord:
    id: str
    question: str
    answerable: bool
    best_dense: float
    best_rerank: float


async def index_test_document(
    pdf_path: Path,
    embedder: Embedder,
) -> tuple[VectorStore, AsyncQdrantClient, int]:
    """Parse, chunk, embed, and index the evaluation PDF into in-memory Qdrant."""
    pages = parse_pdf(pdf_path)
    chunks = chunk_pages(pages, doc_id="eval-doc", user_id="eval-user")
    vectors = await embedder.aembed_texts([c.text for c in chunks])

    client = AsyncQdrantClient(":memory:")
    vec_dim = len(vectors[0]) if vectors else 768

    await client.create_collection(
        collection_name="eval_collection",
        vectors_config={"dense": VectorParams(size=vec_dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    vs = VectorStore(client, collection_name="eval_collection")

    payloads = [
        ChunkPayload(
            doc_id=c.doc_id,
            user_id=c.user_id,
            page=c.page,
            chunk_index=c.chunk_index,
            text=c.text,
        )
        for c in chunks
    ]
    await vs.upsert_chunks(payloads, vectors)
    return vs, client, len(chunks)


async def collect_scores(
    qa_path: Path,
    vs: VectorStore,
    embedder: Embedder,
    reranker: Reranker,
    top_k: int = 20,
    rerank_n: int = 5,
) -> list[QuestionScoreRecord]:
    """Run retrieval for all QA pairs and record dense and rerank scores."""
    records: list[QuestionScoreRecord] = []

    with qa_path.open("r", encoding="utf-8") as f:
        qa_items = [json.loads(line) for line in f if line.strip()]

    for item in qa_items:
        q = item["question"]
        q_vec = await embedder.aembed_query(q)

        dense_res = await vs.dense_search(q_vec, user_id="eval-user", doc_ids=["eval-doc"], top_k=top_k)
        sparse_res = await vs.sparse_search(q, user_id="eval-user", doc_ids=["eval-doc"], top_k=top_k)

        best_dense = max([p.score or 0.0 for p in dense_res], default=0.0)

        fused = reciprocal_rank_fusion(dense_res, sparse_res, k=60)[:rerank_n]
        point_id_map = {str(p.id): p.payload.get("text", "") for p in dense_res + sparse_res if p.payload}

        candidate_texts = [point_id_map.get(pid, "") for pid, _, _, _ in fused if pid in point_id_map]

        if candidate_texts:
            rerank_scores = await reranker.arerank(q, candidate_texts)
            best_rerank = max(rerank_scores) if rerank_scores else 0.0
        else:
            best_rerank = 0.0

        records.append(
            QuestionScoreRecord(
                id=item["id"],
                question=q,
                answerable=item["answerable"],
                best_dense=float(best_dense),
                best_rerank=float(best_rerank),
            )
        )

    return records


def run_calibration_sweep(records: list[QuestionScoreRecord]) -> dict[str, Any]:
    """Sweep threshold combinations and find optimal separating cutoffs."""
    ans = [r for r in records if r.answerable]
    unans = [r for r in records if not r.answerable]

    dense_ans_scores = [r.best_dense for r in ans]
    dense_unans_scores = [r.best_dense for r in unans]
    rerank_ans_scores = [r.best_rerank for r in ans]
    rerank_unans_scores = [r.best_rerank for r in unans]

    dense_candidates = np.linspace(0.1, 0.85, 31)
    rerank_candidates = np.linspace(0.05, 0.8, 31)

    best_combined_score = -1.0
    best_params: dict[str, Any] = {}

    for d_cut in dense_candidates:
        for r_cut in rerank_candidates:
            # Gate decision: refuse if best_dense < d_cut OR best_rerank < r_cut
            ans_refused = sum(1 for r in ans if r.best_dense < d_cut or r.best_rerank < r_cut)
            unans_refused = sum(1 for r in unans if r.best_dense < d_cut or r.best_rerank < r_cut)

            false_refusal_rate = ans_refused / len(ans) if ans else 0.0
            correct_refusal_rate = unans_refused / len(unans) if unans else 1.0

            # Combined objective: maximize correct refusals while keeping false refusals low
            combined_score = correct_refusal_rate + (1.0 - false_refusal_rate)

            # Prioritize candidates meeting the target: false_refusal_rate <= 0.05
            is_valid = false_refusal_rate <= 0.05
            effective_score = combined_score if is_valid else (combined_score - 1.0)

            if effective_score > best_combined_score:
                best_combined_score = effective_score
                best_params = {
                    "dense_score_cutoff": round(float(d_cut), 3),
                    "reranker_score_cutoff": round(float(r_cut), 3),
                    "correct_refusal_rate": round(float(correct_refusal_rate), 4),
                    "false_refusal_rate": round(float(false_refusal_rate), 4),
                    "combined_score": round(float(combined_score), 4),
                }

    summary = {
        "dataset_size": len(records),
        "answerable_count": len(ans),
        "unanswerable_count": len(unans),
        "score_distributions": {
            "dense": {
                "answerable": {
                    "mean": round(float(np.mean(dense_ans_scores)), 4) if dense_ans_scores else 0,
                    "min": round(float(np.min(dense_ans_scores)), 4) if dense_ans_scores else 0,
                    "max": round(float(np.max(dense_ans_scores)), 4) if dense_ans_scores else 0,
                    "p50": round(float(np.median(dense_ans_scores)), 4) if dense_ans_scores else 0,
                },
                "unanswerable": {
                    "mean": round(float(np.mean(dense_unans_scores)), 4) if dense_unans_scores else 0,
                    "min": round(float(np.min(dense_unans_scores)), 4) if dense_unans_scores else 0,
                    "max": round(float(np.max(dense_unans_scores)), 4) if dense_unans_scores else 0,
                    "p50": round(float(np.median(dense_unans_scores)), 4) if dense_unans_scores else 0,
                },
            },
            "rerank": {
                "answerable": {
                    "mean": round(float(np.mean(rerank_ans_scores)), 4) if rerank_ans_scores else 0,
                    "min": round(float(np.min(rerank_ans_scores)), 4) if rerank_ans_scores else 0,
                    "max": round(float(np.max(rerank_ans_scores)), 4) if rerank_ans_scores else 0,
                    "p50": round(float(np.median(rerank_ans_scores)), 4) if rerank_ans_scores else 0,
                },
                "unanswerable": {
                    "mean": round(float(np.mean(rerank_unans_scores)), 4) if rerank_unans_scores else 0,
                    "min": round(float(np.min(rerank_unans_scores)), 4) if rerank_unans_scores else 0,
                    "max": round(float(np.max(rerank_unans_scores)), 4) if rerank_unans_scores else 0,
                    "p50": round(float(np.median(rerank_unans_scores)), 4) if rerank_unans_scores else 0,
                },
            },
        },
        "recommended_thresholds": best_params,
    }
    return summary


async def main_async(args: argparse.Namespace) -> None:
    configure_logging()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(args.pdf)
    qa_path = Path(args.qa)

    if not pdf_path.exists() or not qa_path.exists():
        from eval.dataset.generate_dataset import main as gen_main
        log.info("Generating evaluation dataset...")
        gen_main()

    if args.fake_models:
        log.info("Running calibration using deterministic Fake models")
        embedder = FakeEmbedder()
        reranker = FakeReranker()
    else:
        log.info("Running calibration using real SentenceTransformer + BGE cross-encoder")
        embedder = SentenceTransformerEmbedder()
        reranker = CrossEncoderReranker()

    print("\n--- 1. Indexing Evaluation Document ---")
    vs, client, chunk_count = await index_test_document(pdf_path, embedder)
    print(f"Indexed {chunk_count} chunks into in-memory Qdrant")

    print("\n--- 2. Collecting Scores Across 50 QA Pairs ---")
    records = await collect_scores(qa_path, vs, embedder, reranker)
    print(f"Collected scores for {len(records)} questions")

    print("\n--- 3. Running Threshold Optimization Sweep ---")
    results = run_calibration_sweep(records)

    # Save reports
    json_path = REPORTS_DIR / "calibration_report.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = REPORTS_DIR / "calibration_report.md"
    rec = results["recommended_thresholds"]
    dense_dist = results["score_distributions"]["dense"]
    rerank_dist = results["score_distributions"]["rerank"]

    md_content = f"""# DocuGround Threshold Calibration Report

## Recommended Thresholds
| Parameter | Value |
|---|---|
| **DENSE_SCORE_CUTOFF** | `{rec.get('dense_score_cutoff')}` |
| **RERANKER_SCORE_CUTOFF** | `{rec.get('reranker_score_cutoff')}` |
| **Correct Refusal Rate** | `{rec.get('correct_refusal_rate', 0) * 100:.1f}%` (Target: >= 95%) |
| **False Refusal Rate** | `{rec.get('false_refusal_rate', 0) * 100:.1f}%` (Target: <= 5%) |

## Score Distributions
### Dense Cosine Similarity
- **Answerable Questions**: Mean={dense_dist['answerable']['mean']:.3f}, Median={dense_dist['answerable']['p50']:.3f}, Range=[{dense_dist['answerable']['min']:.3f}, {dense_dist['answerable']['max']:.3f}]
- **Unanswerable Questions**: Mean={dense_dist['unanswerable']['mean']:.3f}, Median={dense_dist['unanswerable']['p50']:.3f}, Range=[{dense_dist['unanswerable']['min']:.3f}, {dense_dist['unanswerable']['max']:.3f}]

### Reranker Cross-Encoder Score
- **Answerable Questions**: Mean={rerank_dist['answerable']['mean']:.3f}, Median={rerank_dist['answerable']['p50']:.3f}, Range=[{rerank_dist['answerable']['min']:.3f}, {rerank_dist['answerable']['max']:.3f}]
- **Unanswerable Questions**: Mean={rerank_dist['unanswerable']['mean']:.3f}, Median={rerank_dist['unanswerable']['p50']:.3f}, Range=[{rerank_dist['unanswerable']['min']:.3f}, {rerank_dist['unanswerable']['max']:.3f}]
"""
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print("\nCalibration Complete!")
    print(f"Recommended DENSE_SCORE_CUTOFF:    {rec.get('dense_score_cutoff')}")
    print(f"Recommended RERANKER_SCORE_CUTOFF: {rec.get('reranker_score_cutoff')}")
    print(f"Correct Refusal Rate:              {rec.get('correct_refusal_rate', 0)*100:.1f}%")
    print(f"False Refusal Rate:                {rec.get('false_refusal_rate', 0)*100:.1f}%")
    print(f"Reports saved to {json_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate refusal gate thresholds")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Path to evaluation PDF")
    parser.add_argument("--qa", default=str(DEFAULT_QA), help="Path to QA pairs JSONL")
    parser.add_argument("--fake-models", action="store_true", help="Use fast fake models for testing")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
