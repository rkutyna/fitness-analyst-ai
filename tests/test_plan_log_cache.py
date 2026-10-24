"""plan_log.read_rules memoises its parse per vault file and log state.

Each test names the part of the key it exercises. Two tests freeze the file
stamp so that the log state, or the vault identity, must stand on its own; a
mutation of either part therefore turns a named test red. All data is synthetic
and every vault lives under pytest's tmp_path.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import sqlite3
import threading
from datetime import date, timedelta

import pytest

from health_advisor import db, plan_log, plan_model
from health_advisor.context import VaultContext, VaultOwnershipError, VaultVersionMismatch

START = date(2030, 1, 4)


@pytest.fixture(autouse=True)
def _fresh_memo():
    plan_log._rules_cache_clear()
    yield
    plan_log._rules_cache_clear()


@pytest.fixture
def parsed_rows(monkeypatch):
    """One entry per log row that read_rules turns into a Rule."""
    seen: list[int] = []
    real = plan_log._rule_from_row

    def counting(row):
        seen.append(row["sequence"])
        return real(row)

    monkeypatch.setattr(plan_log, "_rule_from_row", counting)
    return seen


def _ctx(path, user_id: str = "fixture-user") -> VaultContext:
    ctx = VaultContext.local(path, user_id=user_id, writable=True)
    conn = ctx.connect()
    db.init_db(conn)
    conn.close()
    return ctx


def _rule(label: str, day: date = START) -> plan_model.Rule:
    return plan_model.Rule(
        kind=plan_model.RuleKind.SESSION,
        scope=plan_model.Scope(days=(day,), session="easy-run", modality="running"),
        stated=plan_model.EffectiveInterval(start=day, end=day),
        statement=plan_model.Stated({"label": label}),
        provenance=plan_model.ParsedProvenance(f"fixture/cache-{label}.md", 1),
        payload={"label": label},
    )


def _labels(rules) -> list[str]:
    return [rule.payload["label"] for rule in rules]


def test_repeated_reads_at_one_state_parse_the_log_once(tmp_path, parsed_rows):
    ctx = _ctx(tmp_path / "vault.db")
    for index, label in enumerate(("a", "b", "c")):
        plan_log.append_rule(ctx, _rule(label, START + timedelta(days=index)))
    parsed_rows.clear()

    reads = [_labels(plan_log.read_rules(ctx)) for _ in range(5)]

    assert reads == [["a", "b", "c"]] * 5
    # One parse of three rows. Without the memo the five reads parse fifteen.
    assert len(parsed_rows) == 3


def test_an_append_is_seen_by_the_next_read_and_re_parses(tmp_path, parsed_rows):
    ctx = _ctx(tmp_path / "vault.db")
    plan_log.append_rule(ctx, _rule("a"))
    assert _labels(plan_log.read_rules(ctx)) == ["a"]
    parsed_rows.clear()

    plan_log.append_rule(ctx, _rule("b", START + timedelta(days=1)))

    assert _labels(plan_log.read_rules(ctx)) == ["a", "b"]
    assert len(parsed_rows) == 2
    parsed_rows.clear()
    assert _labels(plan_log.read_rules(ctx)) == ["a", "b"]
    assert parsed_rows == []


def test_the_log_state_alone_invalidates_the_memo(tmp_path, parsed_rows, monkeypatch):
    """A stamp that never moves must not hide an append.

    This is the case the log's own state exists for: a filesystem whose
    timestamps do not change on write would otherwise serve the old parse.
    """
    monkeypatch.setattr(plan_log, "_file_stamp", lambda st: (0, 0, 0))
    ctx = _ctx(tmp_path / "vault.db")
    plan_log.append_rule(ctx, _rule("a"))
    assert _labels(plan_log.read_rules(ctx)) == ["a"]

    plan_log.append_rule(ctx, _rule("b", START + timedelta(days=1)))

    assert _labels(plan_log.read_rules(ctx)) == ["a", "b"]


def test_two_vaults_in_one_process_never_share_a_parse(tmp_path, monkeypatch):
    """Identical log state in two files must not select one file's parse.

    The stamp is frozen and both logs end up with one row at sequence 1, so
    only the vault's identity can tell the two files apart.
    """
    monkeypatch.setattr(plan_log, "_file_stamp", lambda st: (0, 0, 0))
    first = _ctx(tmp_path / "first.db", user_id="fixture-first")
    second = _ctx(tmp_path / "second.db", user_id="fixture-second")
    plan_log.append_rule(first, _rule("first-only"))
    plan_log.append_rule(second, _rule("second-only"))

    assert _labels(plan_log.read_rules(first)) == ["first-only"]
    assert _labels(plan_log.read_rules(second)) == ["second-only"]
    assert _labels(plan_log.read_rules(first)) == ["first-only"]
    assert len(plan_log._RULES_CACHE) == 2


def test_a_file_rewritten_in_place_is_not_served_the_old_parse(tmp_path):
    """Same path, same inode, same log length and last sequence, new content."""
    path = tmp_path / "vault.db"
    ctx = _ctx(path)
    plan_log.append_rule(ctx, _rule("old"))
    assert _labels(plan_log.read_rules(ctx)) == ["old"]
    inode = os.stat(path).st_ino
    stamp = plan_log._file_signature(path)

    replacement = _ctx(tmp_path / "replacement.db")
    plan_log.append_rule(replacement, _rule("new"))
    shutil.copyfile(replacement.db_path, path)

    assert os.stat(path).st_ino == inode
    assert plan_log._file_signature(path) != stamp
    assert _labels(plan_log.read_rules(ctx)) == ["new"]


def test_a_file_renamed_over_the_path_mid_read_is_not_served_the_old_parse(
        tmp_path, monkeypatch):
    """The file is identified before the connection opens, so a swap in between must not be keyed as the old file.

    The swap happens inside ``read_only``: the path is then a different inode
    with the same log length and last sequence. Without the check that the
    file held still across the state read, the old file's memo entry would be
    served for the new file.
    """
    path = tmp_path / "vault.db"
    ctx = _ctx(path)
    plan_log.append_rule(ctx, _rule("old"))
    assert _labels(plan_log.read_rules(ctx)) == ["old"]
    replacement = _ctx(tmp_path / "replacement.db")
    plan_log.append_rule(replacement, _rule("new"))
    real_read_only = VaultContext.read_only

    def swap_then_read_only(self):
        os.replace(replacement.db_path, path)
        return real_read_only(self)

    monkeypatch.setattr(VaultContext, "read_only", swap_then_read_only)

    assert _labels(plan_log.read_rules(ctx)) == ["new"]


def test_a_warm_memo_does_not_bypass_ownership_or_version_checks(tmp_path):
    path = tmp_path / "owned.db"
    owner = _ctx(path, user_id="owner-a")
    owner.claim()
    plan_log.append_rule(owner, _rule("a"))
    pinned = owner.pinned()
    assert _labels(plan_log.read_rules(owner)) == ["a"]
    assert plan_log._RULES_CACHE

    intruder = VaultContext.local(path, user_id="owner-b")
    with pytest.raises(VaultOwnershipError):
        plan_log.read_rules(intruder)

    owner.set_version(pinned.vault_version + 1)
    with pytest.raises(VaultVersionMismatch):
        plan_log.read_rules(pinned)


def test_a_log_without_its_append_only_triggers_is_parsed_every_read(tmp_path, parsed_rows):
    path = tmp_path / "untriggered.db"
    ctx = _ctx(path)
    plan_log.append_rule(ctx, _rule("a"))
    plan_log.append_rule(ctx, _rule("b", START + timedelta(days=1)))
    raw = sqlite3.connect(path)
    raw.execute("DROP TRIGGER plan_statement_log_no_update")
    raw.close()
    parsed_rows.clear()

    for _ in range(3):
        assert _labels(plan_log.read_rules(ctx)) == ["a", "b"]

    assert len(parsed_rows) == 6
    assert not plan_log._RULES_CACHE


def test_a_missing_vault_raises_as_before_and_is_not_created(tmp_path):
    path = tmp_path / "absent.db"

    with pytest.raises(sqlite3.OperationalError):
        plan_log.read_rules(VaultContext.local(path))

    assert not path.exists()
    assert not plan_log._RULES_CACHE


def test_callers_cannot_corrupt_the_memo_through_the_list_or_its_rules(tmp_path):
    ctx = _ctx(tmp_path / "vault.db")
    for index, label in enumerate(("a", "b", "c")):
        plan_log.append_rule(ctx, _rule(label, START + timedelta(days=index)))
    first = plan_log.read_rules(ctx)
    hit = plan_log.read_rules(ctx)
    assert first is not hit

    hit.reverse()
    hit.clear()
    with pytest.raises(dataclasses.FrozenInstanceError):
        first[0].kind = "constraint"
    with pytest.raises(TypeError):
        first[0].payload["label"] = "changed"

    again = plan_log.read_rules(ctx)
    assert _labels(again) == ["a", "b", "c"]
    assert again is not hit


def test_the_memo_is_bounded(tmp_path):
    ctx = _ctx(tmp_path / "vault.db")
    states = plan_log._RULES_CACHE_LIMIT + 3
    for index in range(states):
        plan_log.append_rule(ctx, _rule(f"r{index}", START + timedelta(days=index)))
        plan_log.read_rules(ctx)

    assert len(plan_log._RULES_CACHE) == plan_log._RULES_CACHE_LIMIT
    assert _labels(plan_log.read_rules(ctx))[-1] == f"r{states - 1}"


def test_concurrent_readers_all_see_the_whole_log(tmp_path):
    ctx = _ctx(tmp_path / "vault.db")
    labels = ["a", "b", "c", "d", "e"]
    for index, label in enumerate(labels):
        plan_log.append_rule(ctx, _rule(label, START + timedelta(days=index)))
    results: list[list[str]] = []
    errors: list[BaseException] = []

    def reader():
        try:
            for _ in range(20):
                results.append(_labels(plan_log.read_rules(ctx)))
        except BaseException as exc:  # reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 80
    assert all(result == labels for result in results)
