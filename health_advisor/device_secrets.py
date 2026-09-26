"""Per-device D23 body secrets (T5 piece 9, consumer #426).

Without this module every enrolled phone shares one instance secret
(``receiver.SHARED_SECRET``): whoever wins a photographed-QR race, or has any
vantage on the wire, decrypts every phone's bodies on the instance until the
next ``rotate-secret.sh``. ``HA_DEVICE_SECRET_MODE=per_device`` gives each
QR-paired phone its own secret instead, so the race winner gets only their
own.

``HA_DEVICE_SECRET_MODE`` unset or empty means ``instance`` -- the historical
behaviour, and fully inert: nothing here is imported into a request path,
this store is never opened, and the receiver is byte-identical to one that
predates this module. ``per_device`` is the only other recognised value;
anything else refuses at startup, the same shape as ``HA_DEVICE_AUTH_MODE``.

The store is one JSON file (``HA_DEVICE_SECRETS_FILE``), never a field on
``device_auth.Device``: ``Device(**raw)`` is strict, so an older engine
reading ``devices.json`` must never see a key it does not recognise.
Locking and atomic replace reuse ``device_auth.exclusive_lock`` /
``device_auth.atomic_write_json``, the same primitives ``DeviceRegistry`` and
``enrol.TokenStore`` use -- both already share a directory with this file and
need the same reader/writer discipline.

A secret here is a live credential, not public metadata: ``get()`` and
``kids()`` never return one except through a fresh, explicit lookup, and the
CLI's ``list`` never prints one.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
import time
from dataclasses import asdict, dataclass

from . import device_auth


MODES = ("instance", "per_device")
STORE_VERSION = 1
SECRET_BYTES = 32  # secrets.token_urlsafe(32) -> 43 base64url chars, no padding


def device_secret_mode() -> str:
    """Return the configured mode, refusing an invalid value at startup."""
    mode = os.environ.get("HA_DEVICE_SECRET_MODE", "")
    if mode == "":
        return "instance"
    if mode in MODES:
        return mode
    raise RuntimeError(
        "per-device secrets refuse to start: HA_DEVICE_SECRET_MODE is set to "
        f"invalid value {mode!r}; set it to 'instance' or 'per_device', or "
        "leave it unset for 'instance'."
    )


@dataclass(frozen=True)
class DeviceSecret:
    kid: str
    secret: str
    issued_at: str


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class DeviceSecretStore:
    """The per-device D23 secrets of one instance, in one JSON file.

    Mirrors ``device_auth.DeviceRegistry`` / ``enrol.TokenStore`` in shape
    (stat-signature caching for reads, ``exclusive_lock`` + read-under-lock
    for writes) but is its own file and its own record type.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._signature: tuple | None = None
        self._secrets: dict[str, DeviceSecret] = {}
        # Test-only hook: when set, issue()/forget() sleep for this many
        # seconds after reading the file and before writing it, so a
        # concurrency test can force two callers to overlap inside what
        # should be one exclusive critical section. Production never sets
        # this.
        self._test_race_delay: float = 0.0

    # -- reading

    def _stat_signature(self) -> tuple | None:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns)

    @staticmethod
    def _parse(text: str) -> dict[str, DeviceSecret]:
        data = json.loads(text)
        if not isinstance(data, dict) or data.get("v") != STORE_VERSION:
            raise ValueError("unrecognised device secret store version")
        by_kid: dict[str, DeviceSecret] = {}
        for raw in data.get("secrets", []):
            record = DeviceSecret(**raw)
            if record.kid in by_kid:
                raise ValueError(
                    f"duplicate kid {record.kid!r} in device secret store")
            by_kid[record.kid] = record
        return by_kid

    def _read_file(self) -> dict[str, DeviceSecret]:
        try:
            with open(self.path, encoding="utf-8") as handle:
                return self._parse(handle.read())
        except FileNotFoundError:
            return {}

    def _current(self) -> dict[str, DeviceSecret]:
        """The current store, re-read only when the file has changed.

        An unreadable or corrupt file raises -- failing closed, as
        ``DeviceRegistry.devices()`` does, is the only safe reading of "I
        cannot tell which secret this kid has."
        """
        signature = self._stat_signature()
        if signature != self._signature or signature is None:
            self._secrets = self._read_file()
            self._signature = signature
        return dict(self._secrets)

    def get(self, kid: str) -> DeviceSecret | None:
        return self._current().get(kid)

    def kids(self) -> list[str]:
        """Every kid with a secret on file, in no particular order.

        Never a secret -- callers that need one call ``get(kid)`` for it.
        """
        return list(self._current().keys())

    # -- writing

    def _exclusive(self):
        return device_auth.exclusive_lock(self.path)

    def _write(self, by_kid: dict[str, DeviceSecret]) -> None:
        payload = json.dumps({
            "v": STORE_VERSION,
            "secrets": [asdict(record) for record in by_kid.values()],
        }, indent=2, sort_keys=True) + "\n"
        device_auth.atomic_write_json(self.path, payload,
                                      temp_prefix=".device-secrets.")

    def issue(self, kid: str) -> DeviceSecret:
        """Mint and store a brand-new secret for ``kid``, replacing any prior one.

        Always rotates: a re-pair of an already-known kid gets a fresh
        secret, never the one on file, so a leaked old secret does not
        survive a deliberate re-enrolment (a QR re-pair burns a fresh token
        and calls this again). The read-modify-write happens entirely inside
        ``exclusive_lock`` -- reading the file only after the lock is held --
        so two concurrent ``issue`` calls for two different kids cannot lose
        either write.
        """
        record = DeviceSecret(kid=kid, secret=secrets.token_urlsafe(SECRET_BYTES),
                              issued_at=_utc_now())
        with self._exclusive():
            current = self._read_file()
            if self._test_race_delay:
                time.sleep(self._test_race_delay)
            current[kid] = record
            self._write(current)
            self._secrets = current
            self._signature = self._stat_signature()
        return record

    def forget(self, kid: str) -> bool:
        """Remove one kid's secret; return whether it was present."""
        with self._exclusive():
            current = self._read_file()
            if self._test_race_delay:
                time.sleep(self._test_race_delay)
            if kid not in current:
                return False
            del current[kid]
            self._write(current)
            self._secrets = current
            self._signature = self._stat_signature()
        return True


def store_from_env() -> DeviceSecretStore:
    """The store named by HA_DEVICE_SECRETS_FILE, checked at startup.

    Mirrors ``device_auth.registry_from_env``: an enabled feature with no
    usable directory, an unreadable file, or a file with group/other
    permission bits refuses to start rather than silently running with a
    store that either cannot be written or should never have been readable
    by anyone but this instance.
    """
    path = os.environ.get("HA_DEVICE_SECRETS_FILE", "")
    if not path:
        raise RuntimeError(
            "per-device secrets refuse to start: HA_DEVICE_SECRET_MODE is "
            "'per_device' but HA_DEVICE_SECRETS_FILE is unset; point it at a "
            "writable, persistent path for this instance's per-device "
            "secrets.")
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory) or not os.access(directory, os.W_OK):
        raise RuntimeError(
            "per-device secrets refuse to start: the directory of "
            f"HA_DEVICE_SECRETS_FILE ({directory}) is missing or not "
            "writable.")
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = None
    if mode is not None and mode & 0o077:
        raise RuntimeError(
            "per-device secrets refuse to start: HA_DEVICE_SECRETS_FILE "
            f"({path}) is mode {mode:o}, which grants group or other access "
            "to live D23 secrets; chmod 600 it.")
    store = DeviceSecretStore(path)
    try:
        store._current()
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "per-device secrets refuse to start: HA_DEVICE_SECRETS_FILE is "
            f"unreadable ({exc}).") from exc
    return store


# --------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m health_advisor.device_secrets",
        description="List or forget an instance's per-device D23 secrets.")
    parser.add_argument("--store", default=os.environ.get("HA_DEVICE_SECRETS_FILE"),
                        help="the secret store file (default: HA_DEVICE_SECRETS_FILE)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print every kid with a per-device secret "
                                "on file, one per line -- never the secret")
    forget = sub.add_parser("forget", help="remove one kid's per-device "
                                            "secret; other devices unaffected")
    forget.add_argument("kid")
    args = parser.parse_args(argv)
    if not args.store:
        parser.error("no store: pass --store or set HA_DEVICE_SECRETS_FILE")
    store = DeviceSecretStore(args.store)

    if args.command == "list":
        by_kid = store._current()
        if not by_kid:
            print("no per-device secrets on file")
        for record in by_kid.values():
            print(f"{record.kid}  issued {record.issued_at}")
        return 0

    if not store.forget(args.kid):
        print(f"no per-device secret on file for kid {args.kid!r}; nothing "
              "removed", file=sys.stderr)
        return 1
    print(f"forgot the per-device secret for {args.kid}; other devices "
          "unaffected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
