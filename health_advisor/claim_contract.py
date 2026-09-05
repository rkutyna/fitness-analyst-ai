"""Shared claim-channel vocabulary and metric ownership conventions."""
from __future__ import annotations


def is_metricless_metric(metric) -> bool:
    """Whether a claim metric value expresses metric-lessness."""
    return metric is None or (isinstance(metric, str) and not metric.strip())


def metricless_claim_instruction_sentence() -> str:
    """Return the shared model-facing encoding for a metricless claim."""
    return ("For a metric-less claim, omit the `metric` key entirely; do not "
            "send `metric: null`, `metric: \"\"`, or whitespace as a "
            "substitute for omission.")
