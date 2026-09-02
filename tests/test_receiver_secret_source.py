"""The receiver prefers a secret FILE over the environment, and fails closed.

#101 (F-43). An environment variable is readable for the life of the process
via /proc/<pid>/environ; a file need never enter the environment at all.

The cases that matter are the refusals. `receiver` treats an EMPTY shared
secret as "no check", so any misread of the file that fell back to the
environment — or to "" — would serve an unauthenticated receiver that looks
configured. Every failure mode therefore raises at startup instead.
"""
from __future__ import annotations

import pytest

from health_advisor import receiver


GOOD = "a-perfectly-good-secret-value"


def _secret_file(tmp_path, contents: str, mode: int = 0o600):
    p = tmp_path / "receiver.secret"
    p.write_text(contents)
    p.chmod(mode)
    return p


def _load(monkeypatch, *, file=None, env=None, require=None):
    monkeypatch.delenv("HA_SECRET_FILE", raising=False)
    monkeypatch.delenv("HA_SHARED_SECRET", raising=False)
    monkeypatch.delenv("HA_REQUIRE_SECRET", raising=False)
    if file is not None:
        monkeypatch.setenv("HA_SECRET_FILE", str(file))
    if env is not None:
        monkeypatch.setenv("HA_SHARED_SECRET", env)
    if require is not None:
        monkeypatch.setenv("HA_REQUIRE_SECRET", require)
    return receiver._load_shared_secret()


# --------------------------------------------------------------- the env path

def test_no_file_configured_uses_the_environment(monkeypatch):
    assert _load(monkeypatch, env="legacy-env-secret") == ("legacy-env-secret", "env")


def test_nothing_configured_is_empty_which_means_no_check(monkeypatch):
    """Preserved deliberately: tests and unmigrated launchers rely on it."""
    assert _load(monkeypatch) == ("", "env")


# -------------------------------------------------------------- the file path

def test_a_good_file_is_used_and_reports_its_source(monkeypatch, tmp_path):
    assert _load(monkeypatch, file=_secret_file(tmp_path, GOOD)) == (GOOD, "file")


def test_the_file_wins_over_the_environment(monkeypatch, tmp_path):
    secret, source = _load(
        monkeypatch, file=_secret_file(tmp_path, GOOD), env="env-value-should-lose")
    assert (secret, source) == (GOOD, "file")


def test_surrounding_whitespace_is_stripped_like_the_shell_does(monkeypatch, tmp_path):
    """entrypoint.sh uses `tr -d '[:space:]'`; a trailing newline is normal."""
    secret, _ = _load(monkeypatch, file=_secret_file(tmp_path, f"  {GOOD}\n\n"))
    assert secret == GOOD


# ------------------------------------------------- the refusals, which matter

@pytest.mark.parametrize("contents,label", [
    ("", "an empty file"),
    ("   \n\n", "a whitespace-only file"),
    ("short", "a file under D16's 16 characters"),
])
def test_a_bad_secret_refuses_rather_than_disabling_auth(
        monkeypatch, tmp_path, contents, label):
    with pytest.raises(RuntimeError, match="refusing to start"):
        _load(monkeypatch, file=_secret_file(tmp_path, contents),
              env="fallback-secret-value")


def test_a_missing_file_refuses_even_though_the_env_would_work(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError):
        _load(monkeypatch, file=tmp_path / "does-not-exist",
              env="fallback-secret-value")


def test_a_world_readable_file_refuses(monkeypatch, tmp_path):
    """D16 requires 600 or 400. entrypoint.sh enforces it; so does this."""
    with pytest.raises(RuntimeError, match="mode"):
        _load(monkeypatch, file=_secret_file(tmp_path, GOOD, mode=0o644),
              env="fallback-secret-value")


def test_no_refusal_path_ever_falls_back_to_the_environment(monkeypatch, tmp_path):
    """The property, stated once directly rather than inferred from the cases.

    If any bad-file case returned instead of raising, it would return the
    env's value or "" — and "" means no check. That is the defect this whole
    module exists to prevent, so assert it as a property.
    """
    for contents, mode in [("", 0o600), ("short", 0o600), (GOOD, 0o644)]:
        with pytest.raises(RuntimeError):
            _load(monkeypatch, file=_secret_file(tmp_path, contents, mode=mode),
                  env="fallback-secret-value")


# ------------------------------------------- step two: the export is gone now
#
# `deploy/entrypoint.sh` no longer exports HA_SHARED_SECRET, so the environment
# fallback that used to catch a HA_SECRET_FILE misread is gone. Without a guard
# that leaves ("", "env") — and an empty secret means "no check", so the
# receiver would serve UNAUTHENTICATED on the tailnet while looking configured.
# HA_REQUIRE_SECRET turns that state into a startup failure. These are the tests
# that make removing the export safe rather than merely done.
#
# The companion assertion — that `deploy/entrypoint.sh` contains no
# `export HA_SHARED_SECRET` and does set `HA_REQUIRE_SECRET=1` — was a property
# of that deployment script, not of `receiver.py`. The script is not part of
# this repo, so the assertion has no subject here and was removed. Everything
# below still holds: whatever launches the receiver, an absent or empty secret
# under HA_REQUIRE_SECRET fails at startup instead of serving unauthenticated.


def test_require_secret_refuses_when_nothing_is_configured(monkeypatch):
    """The exact post-step-two failure: no file reached Python, no env left."""
    with pytest.raises(RuntimeError, match="HA_REQUIRE_SECRET"):
        _load(monkeypatch, require="1")


def test_require_secret_refuses_an_empty_environment_secret(monkeypatch):
    with pytest.raises(RuntimeError, match="empty secret disables"):
        _load(monkeypatch, env="", require="1")


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_require_secret_accepts_the_usual_truthy_spellings(monkeypatch, flag):
    with pytest.raises(RuntimeError):
        _load(monkeypatch, require=flag)


@pytest.mark.parametrize("flag", ["0", "false", "no", "", "off"])
def test_require_secret_off_keeps_the_historical_behaviour(monkeypatch, flag):
    """Unset or falsey must not change anything — tests and unmigrated
    launchers rely on empty-means-no-check."""
    assert _load(monkeypatch, require=flag) == ("", "env")


def test_require_secret_does_not_reject_a_working_file(monkeypatch, tmp_path):
    """The deployed configuration: file present, env absent, guard on."""
    path = _secret_file(tmp_path, GOOD)
    assert _load(monkeypatch, file=path, require="1") == (GOOD, "file")


def test_require_secret_does_not_reject_a_working_environment_secret(monkeypatch):
    """The guard is about ABSENCE, not about which source was used."""
    assert _load(monkeypatch, env=GOOD, require="1") == (GOOD, "env")


def test_a_bad_file_still_refuses_when_the_guard_is_off(monkeypatch, tmp_path):
    """The file path's own fail-closed behaviour is independent of the guard."""
    path = _secret_file(tmp_path, "short")
    with pytest.raises(RuntimeError, match="chars after trimming"):
        _load(monkeypatch, file=path, env=GOOD)
