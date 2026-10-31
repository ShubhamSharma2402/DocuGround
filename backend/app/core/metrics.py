"""Prometheus metrics collector for DocuGround."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# ── Counters ─────────────────────────────────────────────────────────────────
DOCUGROUND_UPLOADS_TOTAL = Counter(
    "docuground_uploads_total",
    "Total document uploads initiated",
)

DOCUGROUND_JOBS_TOTAL = Counter(
    "docuground_jobs_total",
    "Total ingestion jobs processed by outcome status",
    labelnames=["status"],
)

DOCUGROUND_GATE_REFUSALS_TOTAL = Counter(
    "docuground_gate_refusals_total",
    "Total chat queries refused by threshold gate",
)

DOCUGROUND_CHAT_REQUESTS_TOTAL = Counter(
    "docuground_chat_requests_total",
    "Total chat requests received",
)

# ── Histograms ───────────────────────────────────────────────────────────────
DOCUGROUND_QUERY_LATENCY_SECONDS = Histogram(
    "docuground_query_latency_seconds",
    "Latency breakdown of query steps in seconds",
    labelnames=["step"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


def export_metrics() -> tuple[bytes, str]:
    """Generate Prometheus metric scrape response."""
    return generate_latest(), CONTENT_TYPE_LATEST
