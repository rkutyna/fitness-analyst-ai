from __future__ import annotations

from collections import Counter
from pathlib import Path

from tests import replay_ask_captures as replay


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "ask_captures"


def test_rejected_presentation_value_is_counted_by_numeric_token_once():
    verification = {
        "unsupported": ["20"],
        "verdict": {"numbers": [{"claimed": "20 minutes", "ok": False}]},
    }

    assert replay._rejected_figures(verification) == ["20"]


def test_fixture_classifies_rejected_figures_across_all_attempts():
    rows = list(replay._captures(FIXTURE_ROOT))

    assert len(rows) == 2
    assert Counter(
        figure["classification"]
        for row in rows
        for figure in row["rejected_figures"]
    ) == Counter({"present-in-draft": 1, "never-in-draft": 1})
    present = next(row for row in rows if row["arm"] == "synthetic_present")
    never = next(row for row in rows if row["arm"] == "synthetic_never")
    assert present["after_ok"] is True
    assert never["after_ok"] is False
    assert present["rejected_figures"][0]["figure"] == 17.5
    assert never["rejected_figures"][0]["figure"] == 19.5


def test_missing_capture_directory_is_an_empty_success(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.chdir(tmp_path)

    assert replay.main([]) == 0
    output = capsys.readouterr().out
    assert "captures: 0" in output
    assert "rejected figures: 0; present-in-draft: 0; " \
           "never-in-draft: 0" in output


def test_summary_reports_fixture_counts(capsys):
    rows = list(replay._captures(FIXTURE_ROOT))

    replay._summary(rows)
    output = capsys.readouterr().out
    assert "captures: 2" in output
    assert "rejected figures: 2; present-in-draft: 1; " \
           "never-in-draft: 1" in output
