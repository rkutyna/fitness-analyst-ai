from health_advisor import deepdive_memory as M


def test_new_scratchpad_writes_schema(tmp_path):
    p = str(tmp_path / "s.json")
    data = M.new_scratchpad(p, "2026-06-29", [{"id": 1, "question": "Q1?"}])
    assert data["as_of"] == "2026-06-29"
    assert data["tasks"] == [{"id": 1, "question": "Q1?", "status": "open"}]
    assert data["findings"] == [] and data["notes"] == [] and data["log"] == []
    assert M.load(p)["tasks"][0]["question"] == "Q1?"  # persisted


def test_append_finding_persists_and_counts(tmp_path):
    p = str(tmp_path / "s.json")
    M.new_scratchpad(p, "2026-06-29", [{"id": 1, "question": "Q1?"}])
    r = M.append_finding(p, 1, "Steps rose.",
                         numbers=[{"metric": "step_count", "field": "mean", "value": 5413}],
                         tools_used=["summarize_metric"], confidence=0.8)
    assert r == {"ok": True, "n_findings": 1}
    f = M.load(p)["findings"][0]
    assert f["task_id"] == 1 and f["claim"] == "Steps rose."
    assert f["numbers"][0]["value"] == 5413 and "ts" in f


def test_append_note_and_log(tmp_path):
    p = str(tmp_path / "s.json")
    M.new_scratchpad(p, "2026-06-29", [{"id": 1, "question": "Q1?"}])
    assert M.append_note(p, 1, "check sleep next") == {"ok": True}
    M.append_log(p, 1, 3, "tool", "summarize_metric(step_count)")
    d = M.load(p)
    assert d["notes"][0]["text"] == "check sleep next"
    assert d["log"][0]["event"] == "tool" and d["log"][0]["turn"] == 3


def test_compact_state_renders_question_findings_notes(tmp_path):
    p = str(tmp_path / "s.json")
    M.new_scratchpad(p, "2026-06-29", [{"id": 1, "question": "How are steps?"}])
    M.append_finding(p, 1, "Mean steps 5413.",
                     numbers=[{"metric": "step_count", "field": "mean", "value": 5413}])
    M.append_note(p, 1, "compare to last month")
    s = M.compact_state(M.load(p), 1)
    assert "How are steps?" in s and "Mean steps 5413." in s
    assert "step_count" in s and "5413" in s and "compare to last month" in s


def test_compact_state_handles_empty(tmp_path):
    p = str(tmp_path / "s.json")
    M.new_scratchpad(p, "2026-06-29", [{"id": 2, "question": "Q?"}])
    assert "No findings recorded yet." in M.compact_state(M.load(p), 2)


def test_build_memory_tools_record_finding_mutates_file(tmp_path):
    p = str(tmp_path / "s.json")
    M.new_scratchpad(p, "2026-06-29", [{"id": 1, "question": "Q?"}])
    tools = M.build_memory_tools(p, 1)
    fn, schema = tools["record_finding"]
    out = fn(claim="Steps 5413.", numbers=[{"metric": "step_count", "value": 5413}])
    assert out["ok"] is True and out["n_findings"] == 1
    assert M.load(p)["findings"][0]["claim"] == "Steps 5413."
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "record_finding"
    assert "claim" in schema["function"]["parameters"]["properties"]


def test_build_memory_tools_note_and_read_state(tmp_path):
    p = str(tmp_path / "s.json")
    M.new_scratchpad(p, "2026-06-29", [{"id": 1, "question": "How are steps?"}])
    tools = M.build_memory_tools(p, 1)
    tools["note"][0](text="check hr")
    tools["record_finding"][0](claim="Mean 5413.",
                               numbers=[{"metric": "step_count", "value": 5413}])
    state = tools["read_state"][0]()
    assert state["task_id"] == 1
    assert "How are steps?" in state["state"] and "Mean 5413." in state["state"]
    assert "check hr" in state["state"]
    assert {"record_finding", "note", "read_state"} == set(tools)


def test_memory_tool_names_match_schemas(tmp_path):
    p = str(tmp_path / "s.json")
    M.new_scratchpad(p, "2026-06-29", [{"id": 1, "question": "Q?"}])
    for name, (_, schema) in M.build_memory_tools(p, 1).items():
        assert schema["function"]["name"] == name


def test_append_finding_dedups_same_task_and_claim(tmp_path):
    """A re-record of the same (task_id, claim) — e.g. when a finalize/compaction nudge
    re-asks a model that already recorded it — must not inflate the board (which would
    double-count it downstream in the judge/filter)."""
    p = str(tmp_path / "s.json")
    M.new_scratchpad(p, "2026-06-29", [{"id": 1, "question": "Q?"}])
    r1 = M.append_finding(p, 1, "Steps rose.", numbers=[{"metric": "step_count", "value": 5413}])
    r2 = M.append_finding(p, 1, "Steps rose.", numbers=[{"metric": "step_count", "value": 5413}])
    assert r1 == {"ok": True, "n_findings": 1}
    assert r2 == {"ok": True, "n_findings": 1}  # duplicate did not append
    assert len(M.load(p)["findings"]) == 1
    # a distinct claim still appends
    assert M.append_finding(p, 1, "HR steady.")["n_findings"] == 2
    # the same claim text under a different task id is independent (keyed on task_id+claim)
    assert M.append_finding(p, 2, "Steps rose.")["n_findings"] == 3
