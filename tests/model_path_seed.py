"""The autouse seed the model-path test files share.

Every file that drives the model path gets one `fixture_metric` row in its
`conn` vault, so the day-zero check does not refuse before the test runs.
Import the fixture into a test module to opt it in:

    from tests.model_path_seed import _seed_model_path_vault  # noqa: F401

Legacy model-orchestration tests (chat, ask endpoint) stay on this path on
purpose; they must not move to the day-zero path the new tests cover.
"""
import pytest

from tests.conftest import seed_metric


@pytest.fixture(autouse=True)
def _seed_model_path_vault(conn):
    seed_metric(conn, "fixture_metric", "2026-01-01", [1])
