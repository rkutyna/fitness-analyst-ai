"""Leaf helpers for the base64url wire format used by device, enrolment and push.

Imports only the standard library so any module can depend on it without an
import cycle.
"""
from __future__ import annotations

import base64
import binascii
import re


def b64url_encode(data: bytes) -> str:
    """URL-safe base64 with the trailing padding stripped."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(text: str) -> bytes:
    """Decode unpadded base64url; anything outside the alphabet is refused."""
    if not isinstance(text, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", text):
        raise ValueError("not base64url")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("not base64url") from exc
