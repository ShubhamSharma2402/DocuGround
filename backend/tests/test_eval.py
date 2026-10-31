"""Unit and integration tests for Phase 6: Dataset, Calibration, and Ablation Evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from eval.dataset.generate_dataset import create_sample_pdf, create_qa_jsonl, QA_PAIRS
from eval.calibrate_threshold import (
    index_test_document,
    collect_scores,
    run_calibration_sweep,
)
from eval.run_eval import run_single_eval
from app.services.embedder import FakeEmbedder
from app.services.reranker import FakeReranker
from app.services.llm import FakeLLMProvider


@pytest.fixture
def eval_dataset_paths(tmp_path: Path):
    pdf = create_sample_pdf()
    qa = create_qa_jsonl()
    return pdf, qa


def test_dataset_generation(eval_dataset_paths):
    """Verify that dataset generation creates 10-page PDF and exactly 50 labeled QA pairs."""
    pdf_path, qa_path = eval_dataset_paths
    assert pdf_path.exists()
    assert qa_path.exists()

    with qa_path.open("r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    assert len(items) == 50
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]

    assert len(answerable) == 35
    assert len(unanswerable) == 15

    for item in answerable:
        assert len(item["expected_pages"]) >= 1
        assert len(item["key_terms"]) >= 1
        assert item["ground_truth"] != ""

    for item in unanswerable:
        assert item["expected_pages"] == []
        assert "couldn't find this" in item["ground_truth"].lower()


@pytest.mark.asyncio
async def test_threshold_calibration_flow(eval_dataset_paths):
    """Verify end-to-end score collection and sweep on the evaluation dataset."""
    pdf_path, qa_path = eval_dataset_paths

    embedder = FakeEmbedder()
    reranker = FakeReranker()

    vs, client, chunk_count = await index_test_document(pdf_path, embedder)
    assert chunk_count > 0

    scores = await collect_scores(qa_path, vs, embedder, reranker, top_k=5, rerank_n=3)
    assert len(scores) == 50

    sweep_result = run_calibration_sweep(scores)
    assert "recommended_thresholds" in sweep_result
    assert "dense_score_cutoff" in sweep_result["recommended_thresholds"]
    assert "reranker_score_cutoff" in sweep_result["recommended_thresholds"]
    assert "score_distributions" in sweep_result


@pytest.mark.asyncio
async def test_ablation_evaluation_run(eval_dataset_paths):
    """Verify that run_single_eval computes all required metrics."""
    pdf_path, qa_path = eval_dataset_paths

    with qa_path.open("r", encoding="utf-8") as f:
        qa_items = [json.loads(line) for line in f if line.strip()]

    embedder = FakeEmbedder()
    reranker = FakeReranker()
    llm = FakeLLMProvider()

    vs, client, _ = await index_test_document(pdf_path, embedder)

    metrics = await run_single_eval(
        config_name="Test Hybrid + Reranker",
        qa_items=qa_items,
        vs=vs,
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        dense_cutoff=0.1,
        reranker_cutoff=0.2,
        use_sparse=True,
        use_reranker=True,
        use_verification=False,
    )

    assert metrics.name == "Test Hybrid + Reranker"
    assert 0.0 <= metrics.hit_rate_at_1 <= 1.0
    assert 0.0 <= metrics.hit_rate_at_3 <= 1.0
    assert 0.0 <= metrics.hit_rate_at_5 <= 1.0
    assert 0.0 <= metrics.mrr_at_5 <= 1.0
    assert 0.0 <= metrics.correct_refusal_rate <= 1.0
    assert 0.0 <= metrics.false_refusal_rate <= 1.0
    assert metrics.latency_p50_ms >= 0.0
    assert len(metrics.records) == 50
