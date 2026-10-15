"""Retired engine names stay retired: no alias, dead helper or dropped module comes back."""
from pathlib import Path

import pytest

from health_advisor import analysis, chat, facts, llm, normalize, plan_model

RETIRED_NAMES = [
    (analysis, "_recent_and_baseline"),
    (facts, "record_proposed"),
    (facts, "propose_fact"),
    (facts, "confirm_fact"),
    (facts, "reject_fact"),
    (facts, "remove_fact"),
    (facts, "add_fact"),
    (facts, "get_facts"),
    (facts, "render_facts"),
    (plan_model, "serialize_absence"),
    (normalize, "workout_name_to_canonical"),
    (normalize, "to_canonical"),
    (normalize, "_WORKOUT_QUALIFIERS"),
    (normalize, "_WORKOUT_KEYWORDS"),
    (llm, "_submit_answer_response"),
    (chat, "ASK_CAUSES"),
]


@pytest.mark.parametrize("module,name", RETIRED_NAMES,
                         ids=lambda v: getattr(v, "__name__", v))
def test_retired_name_is_absent_from_module_and_all(module, name):
    assert not hasattr(module, name)
    assert name not in getattr(module, "__all__", ())


def test_routes_module_file_is_retired():
    # Not find_spec(): when this package lacks a submodule, the editable install
    # elsewhere on the path still resolves it, so importability says nothing here.
    package_dir = Path(analysis.__file__).resolve().parent
    assert not (package_dir / "routes.py").exists()
