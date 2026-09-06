"""The `provider` routing block: ordered failover that cannot leave the D15 set.

Every semantic asserted here was verified live against OpenRouter on
2026-09-06, because the published docs are ambiguous on the two points that
matter. Both probes are recorded in `_openrouter_provider`'s own comment; the
tests below pin the request we build, not OpenRouter's behaviour, which is the
only half of the contract this process controls.

The privacy argument these tests exist to defend: `only` bounds the set,
`allow_fallbacks` forbids escaping it, `zdr`/`data_collection` make OpenRouter
refuse a retaining or training endpoint independently of our own allow-list,
and the throughput preference is SOFT so it can reorder that set but never
widen it.
"""
from health_advisor import llm


def _block(monkeypatch, pinned, floor=50):
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", pinned)
    monkeypatch.setattr(llm, "OPENROUTER_MIN_THROUGHPUT", floor)
    return llm._openrouter_provider()


def test_order_and_only_are_both_the_pinned_list(monkeypatch):
    """Preference and boundary are the same set, so failover stays inside it."""
    block = _block(monkeypatch, "reka/fp4,coreweave/fp8,together")
    assert block["order"] == ["reka/fp4", "coreweave/fp8", "together"]
    assert block["only"] == ["reka/fp4", "coreweave/fp8", "together"]
    # Order is a real preference: the pinned sequence is preserved verbatim
    # rather than sorted by speed or name.
    assert block["order"] == block["only"]


def test_fallbacks_stay_disabled_with_several_providers(monkeypatch):
    """A multi-provider pin must not re-enable escape to the wider market.

    OpenRouter walks the ordered list past a failing provider even with
    fallbacks off — verified live — so ordered failover needs no loosening
    here. This is the assertion that fails if someone "enables failover" by
    flipping allow_fallbacks instead.
    """
    block = _block(monkeypatch, "reka/fp4,coreweave/fp8,together")
    assert block["allow_fallbacks"] is False


def test_hard_privacy_filters_are_always_present(monkeypatch):
    """D15 must not rest solely on this process's own allow-list."""
    block = _block(monkeypatch, "reka/fp4")
    assert block["zdr"] is True
    assert block["data_collection"] == "deny"


def test_throughput_floor_is_published_as_a_p50_preference(monkeypatch):
    block = _block(monkeypatch, "reka/fp4,together", floor=50)
    assert block["preferred_min_throughput"] == {"p50": 50}


def test_throughput_floor_is_omitted_when_disabled(monkeypatch):
    """0 disables the preference without disturbing the boundary."""
    block = _block(monkeypatch, "reka/fp4,together", floor=0)
    assert "preferred_min_throughput" not in block
    assert block["only"] == ["reka/fp4", "together"]
    assert block["allow_fallbacks"] is False


def test_the_floor_never_widens_the_permitted_set(monkeypatch):
    """The soft/hard asymmetry, asserted as a property rather than a value.

    Whatever the floor is set to — including a value nothing can meet — the
    permitted set is exactly the pin. A future change that let the preference
    filter or extend `only` would fail here.
    """
    for floor in (0, 50, 100_000):
        block = _block(monkeypatch, "reka/fp4,coreweave/fp8", floor=floor)
        assert block["only"] == ["reka/fp4", "coreweave/fp8"]
        assert block["allow_fallbacks"] is False


def test_no_pin_publishes_no_boundary(monkeypatch):
    """Unpinned is refused upstream by D15; this block must not invent a set."""
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDERS", "")
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_SORT", "")
    assert llm._openrouter_provider() == {}
