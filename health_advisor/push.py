"""Privacy-preserving APNs delivery for completed answer notifications.

This module has no process-wide APNs configuration.  Credentials, topic and
endpoint enter through :class:`APNsSender` (or its explicit ``from_env``
factory), which keeps one user's delivery configuration from becoming an
ambient default for another vault.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature


logger = logging.getLogger(__name__)

APPROVED_APNS_HOSTS = frozenset({
    "api.push.apple.com",
    "api.sandbox.push.apple.com",
})
APNS_ENVIRONMENTS = frozenset({"sandbox", "production"})


def validate_apns_endpoint(endpoint: str) -> str:
    """Validate and return an APNs HTTPS base URL.

    Host authorization is an exact comparison on the parsed hostname.  A
    lookalike such as ``api.push.apple.com.attacker.example`` is not Apple's
    host, even though it contains the approved name as a substring.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("APNs endpoint is required")
    endpoint = endpoint.strip()
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid APNs endpoint") from exc
    host = parsed.hostname.lower() if parsed.hostname else None
    if parsed.scheme.lower() != "https":
        raise ValueError("APNs endpoint must use https")
    if host not in APPROVED_APNS_HOSTS:
        raise ValueError("APNs endpoint host is not approved")
    if parsed.username or parsed.password or port not in (None, 443):
        raise ValueError("APNs endpoint must be Apple's HTTPS host")
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("APNs endpoint must be a host URL without a path")
    return endpoint.rstrip("/")


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def build_apns_jwt(
    private_key: ec.EllipticCurvePrivateKey,
    *,
    key_id: str,
    team_id: str,
    issued_at: int | None = None,
) -> str:
    """Build an APNs provider JWT with an RFC 7515 ES256 signature."""
    if not key_id.strip() or not team_id.strip():
        raise ValueError("APNs key id and team id are required")
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise ValueError("APNs key must be an EC private key")
    if not isinstance(private_key.curve, ec.SECP256R1):
        raise ValueError("APNs key must use the P-256 curve")
    header = {"alg": "ES256", "kid": key_id}
    claims = {
        "iss": team_id,
        "iat": issued_at if issued_at is not None else int(
            datetime.now(timezone.utc).timestamp()
        ),
    }
    encoded_header = _b64url(json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode("ascii"))
    encoded_claims = _b64url(json.dumps(
        claims, separators=(",", ":"), sort_keys=True
    ).encode("ascii"))
    signing_input = f"{encoded_header}.{encoded_claims}".encode("ascii")
    der_signature = private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_signature)
    raw_signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return f"{encoded_header}.{encoded_claims}.{_b64url(raw_signature)}"


def build_answer_ready_payload(turn_id: str) -> dict[str, Any]:
    """Return the only notification content sent through APNs.

    The payload intentionally contains a generic alert and the durable turn
    identity.  It does not accept, derive or carry question, answer, figure,
    or metric data.
    """
    if not isinstance(turn_id, str) or not turn_id.strip():
        raise ValueError("turn id must be a non-empty string")
    return {
        "aps": {
            "alert": {
                "title": "Answer ready",
                "body": "Your answer is ready.",
            },
            "sound": "default",
        },
        "turn_id": turn_id.strip(),
    }


@dataclass(frozen=True)
class APNsConfig:
    key_path: Path
    key_id: str
    team_id: str
    topic: str
    endpoint: str

    def __post_init__(self) -> None:
        if not str(self.key_path).strip():
            raise ValueError("APNs key path is required")
        if not self.key_id.strip() or not self.team_id.strip():
            raise ValueError("APNs key id and team id are required")
        if not self.topic.strip():
            raise ValueError("APNs topic is required")
        object.__setattr__(self, "endpoint", validate_apns_endpoint(self.endpoint))

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "APNsConfig":
        """Load all APNs settings from explicitly named environment values."""
        import os

        values = environ if environ is not None else os.environ
        names = {
            "key_path": "HA_APNS_KEY_PATH",
            "key_id": "HA_APNS_KEY_ID",
            "team_id": "HA_APNS_TEAM_ID",
            "topic": "HA_APNS_TOPIC",
            "endpoint": "HA_APNS_ENDPOINT",
        }
        missing = [env_name for field, env_name in names.items()
                   if not (values.get(env_name) or "").strip()]
        if missing:
            raise RuntimeError(
                "APNs configuration is incomplete; unset: " + ", ".join(missing)
            )
        return cls(
            key_path=Path(values[names["key_path"]]),
            key_id=values[names["key_id"]],
            team_id=values[names["team_id"]],
            topic=values[names["topic"]],
            endpoint=values[names["endpoint"]],
        )


class APNsSender:
    """Send one generic answer-ready notification to APNs."""

    def __init__(
        self,
        *,
        key_path: str | Path,
        key_id: str,
        team_id: str,
        topic: str,
        endpoint: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not str(key_path).strip() or not key_id.strip() or not team_id.strip():
            raise ValueError("APNs key path, key id, and team id are required")
        if not topic.strip():
            raise ValueError("APNs topic is required")
        self.config = APNsConfig(
            key_path=Path(key_path), key_id=key_id.strip(),
            team_id=team_id.strip(), topic=topic.strip(),
            endpoint=validate_apns_endpoint(endpoint),
        )
        try:
            key_bytes = self.config.key_path.read_bytes()
            private_key = serialization.load_pem_private_key(key_bytes, password=None)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("could not load APNs EC private key") from exc
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            raise ValueError("APNs key must be an EC private key")
        if not isinstance(private_key.curve, ec.SECP256R1):
            raise ValueError("APNs key must use the P-256 curve")
        self._private_key = private_key
        self._http_client = http_client or httpx.Client(timeout=10.0)

    @classmethod
    def from_env(
        cls, *, http_client: httpx.Client | None = None
    ) -> "APNsSender":
        config = APNsConfig.from_env()
        return cls(
            key_path=config.key_path, key_id=config.key_id,
            team_id=config.team_id, topic=config.topic,
            endpoint=config.endpoint, http_client=http_client,
        )

    def jwt(self, *, issued_at: int | None = None) -> str:
        return build_apns_jwt(
            self._private_key, key_id=self.config.key_id,
            team_id=self.config.team_id, issued_at=issued_at,
        )

    def send_with_status(
        self,
        device_token: str,
        turn_id: str,
        *,
        environment: str | None = None,
    ) -> int | None:
        """Attempt delivery and return its HTTP status, or ``None`` on error.

        This is an additive seam for the receiver's token-retirement hook.
        ``send`` below retains its original boolean API and failure handling.
        """
        try:
            if not isinstance(device_token, str) or not device_token.strip():
                raise ValueError("device token must be a non-empty string")
            if environment is not None and environment not in APNS_ENVIRONMENTS:
                raise ValueError("APNs environment must be sandbox or production")
            payload = build_answer_ready_payload(turn_id)
            endpoint = self.config.endpoint
            if environment in APNS_ENVIRONMENTS:
                endpoint = (
                    "https://api.sandbox.push.apple.com"
                    if environment == "sandbox"
                    else "https://api.push.apple.com"
                )
            response = self._http_client.post(
                f"{endpoint}/3/device/{device_token.strip()}",
                headers={
                    "authorization": f"bearer {self.jwt()}",
                    "apns-topic": self.config.topic,
                    "apns-push-type": "alert",
                    "apns-priority": "10",
                    "apns-collapse-id": turn_id,
                },
                content=json.dumps(
                    payload, separators=(",", ":"), ensure_ascii=True
                ).encode("utf-8"),
            )
            if not 200 <= response.status_code < 300:
                logger.warning("APNs push failed with HTTP status %s",
                               response.status_code)
            return response.status_code
        except Exception as exc:  # noqa: BLE001 - push is best effort by contract
            logger.warning("APNs push failed (%s)", type(exc).__name__)
            return None

    def send(self, device_token: str, turn_id: str) -> bool:
        """Attempt delivery, announcing and swallowing every push failure."""
        status = self.send_with_status(device_token, turn_id)
        return status is not None and 200 <= status < 300
