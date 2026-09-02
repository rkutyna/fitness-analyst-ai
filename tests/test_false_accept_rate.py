"""The false-accept harness measures the shipped grounding gate itself."""
from __future__ import annotations

from health_advisor.numeric_tokens import NUM_RE
from scripts.false_accept_rate import measure_payload, _probe_text


def test_synthetic_payload_has_hand_computed_gate_results():
    # Numeric leaves are 10, 20, 100, and a repeated 10; the bool is not a leaf.
    payload = {
        "first": 10,
        "nested": [{"second": 20.0}, {"third": 100}],
        "repeated": 10,
        "not_a_number": True,
    }
    integer_probes = (1, 10, 11, 100)
    decimal_probes = (9.9, 10.0, 10.1, 99.5)

    result = measure_payload(payload, integer_probes, decimal_probes)

    # Integers: 1 reject, 10 accept, 11 reject, 100 accept -> 2/4.
    # Decimals: 9.9 reject, 10.0 accept, 10.1 reject, 99.5 accept because
    # |99.5 - 100| = 0.5 <= max(0.05, 100 * 0.005) -> 2/4.
    assert result["number_count"] == 4
    assert result["integers"]["accepted"] == 2
    assert result["integers"]["total"] == 4
    assert result["one_decimal"]["accepted"] == 2
    assert result["one_decimal"]["total"] == 4

    # Both the generated token and complete probe prose contain exactly the
    # shared tokenizer's one intended match.
    for token in ("10", "10.0", "99.5"):
        assert NUM_RE.findall(token) == [token]
        assert NUM_RE.findall(_probe_text(token)) == [token]
