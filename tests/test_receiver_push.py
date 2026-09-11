from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from health_advisor import chat, db, push, receiver


HEADERS = {"x-health-secret": "push-secret"}


class _RecordingSender:
    def __init__(self, status: int = 200, *, raises: bool = False):
        self.status = status
        self.raises = raises
        self.calls: list[tuple[str, str, str | None]] = []
        self.done = threading.Event()

    def send_with_status(self, token, turn_id, *, environment=None):
        if self.raises:
            raise RuntimeError("synthetic APNs failure")
        self.calls.append((token, turn_id, environment))
        self.done.set()
        return self.status


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_registration_delete_malformed_and_schema_have_no_health_columns(
        vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "push-secret")
    with TestClient(receiver.create_app(vault)) as client:
        registered = client.post(
            "/v1/push/register",
            json={"token": "synthetic-token", "environment": "sandbox"},
            headers=HEADERS,
        )
        deleted = client.request(
            "DELETE", "/v1/push/register", json={"token": "synthetic-token"},
            headers=HEADERS,
        )
        malformed = client.post(
            "/v1/push/register", json={"token": "", "environment": "sandbox"},
            headers=HEADERS,
        )

    assert registered.status_code == 200
    assert deleted.status_code == 200
    assert malformed.status_code == 422
    assert "non-empty" in malformed.json()["detail"]
    conn = vault.connect()
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(device_tokens)")
    }
    assert columns == {"token", "first_seen_at", "last_seen_at", "apns_environment"}
    assert conn.execute("SELECT COUNT(*) FROM device_tokens").fetchone()[0] == 0
    conn.close()


def test_disconnected_trigger_has_four_exactly_once_counts(vault, monkeypatch):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "push-secret")
    monkeypatch.setattr(receiver.Request, "is_disconnected",
                        lambda request: _true_async())
    sender = _RecordingSender()
    app = receiver.create_app(vault, apns_sender=sender)
    conversation = chat.create_conversation(vault, conversation_id="push-counts")
    conn = vault.connect()
    db.register_device_token(conn, "synthetic-token", "sandbox")
    conn.close()

    def answer(ctx, question, **kwargs):
        return {
            "text": f"safe answer {question}",
            "mode": "fallback" if "fallback" in question else "narration",
            "tool_trace": [], "verification": {},
        }

    monkeypatch.setattr(chat, "answer_question", answer)
    with TestClient(app) as client:
        completed = client.post(
            "/v1/ask", json={"conversation_id": conversation["id"],
                            "question": "synthetic question"},
            headers=HEADERS,
        )
        assert completed.status_code == 200
        _wait_for(lambda: len(sender.calls) == 1)
        completed_count = len(sender.calls)
        first_answer = next(
            turn for turn in chat.list_turns(vault, conversation["id"])
            if turn["role"] == "assistant"
        )
        before_retry = len(sender.calls)
        retry = app.state.apns_dispatcher.enqueue(first_answer)
        _wait_for(lambda: len(sender.calls) == before_retry)
        retry_count = len(sender.calls) - before_retry

        delivered = chat.append_turn(
            vault, conversation["id"], "assistant", "delivered answer",
            answers_turn_id=first_answer["answers_turn_id"],
            client_disconnected_at="2026-09-11T12:00:00+00:00",
        )
        chat.mark_turn_delivered(vault, delivered["id"])
        before_delivered = len(sender.calls)
        delivered_enqueue = app.state.apns_dispatcher.enqueue(delivered)
        _wait_for(lambda: len(sender.calls) == before_delivered)
        delivered_count = len(sender.calls) - before_delivered

        fallback = client.post(
            "/v1/ask", json={"conversation_id": conversation["id"],
                            "question": "fallback question"},
            headers=HEADERS,
        )
        assert fallback.status_code == 200
        before_fallback = len(sender.calls)
        _wait_for(lambda: len(sender.calls) == before_fallback)
        fallback_count = len(sender.calls) - before_fallback

    assert retry is False
    assert delivered_enqueue is False
    counts = {"completed_and_undelivered": completed_count,
              "retry_of_same_turn": retry_count,
              "delivered": delivered_count, "fallback": fallback_count}
    print(f"[exactly-once] {counts}")
    assert counts == {
        "completed_and_undelivered": 1,
        "retry_of_same_turn": 0,
        "delivered": 0,
        "fallback": 0,
    }


def test_410_removes_token_and_sender_failure_does_not_fail_request(
        vault, monkeypatch, caplog):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "push-secret")
    conn = vault.connect()
    db.init_db(conn)
    db.register_device_token(conn, "gone-token", "production")
    conn.close()
    gone = _RecordingSender(status=410)
    app = receiver.create_app(vault, apns_sender=gone)
    conversation = chat.create_conversation(vault, conversation_id="gone")
    turn = chat.append_turn(
        vault, conversation["id"], "assistant", "safe answer",
        answers_turn_id=None,
        client_disconnected_at="2026-09-11T12:00:00+00:00",
    )
    assert app.state.apns_dispatcher.enqueue(turn) is True
    _wait_for(lambda: not vault_has_token(vault, "gone-token"))

    failing = _RecordingSender(raises=True)
    app = receiver.create_app(vault, apns_sender=failing)
    second = chat.create_conversation(vault, conversation_id="failure")
    conn = vault.connect()
    db.register_device_token(conn, "failure-token", "sandbox")
    conn.close()
    monkeypatch.setattr(chat, "answer_question", lambda *args, **kwargs: {
        "text": "safe answer", "mode": "narration", "tool_trace": [],
        "verification": {},
    })
    with TestClient(app) as client:
        response = client.post(
            "/v1/ask", json={"conversation_id": second["id"], "question": "q"},
            headers=HEADERS,
        )
    assert response.status_code == 200


def test_network_failure_request_returns_200_and_failure_is_logged(
        vault, monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(receiver, "SHARED_SECRET", "push-secret")
    key_path = Path(tmp_path) / "synthetic.p8"
    key = ec.generate_private_key(ec.SECP256R1())
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))

    def failing_transport(request):
        raise httpx.ConnectError("synthetic network failure", request=request)

    sender = push.APNsSender(
        key_path=key_path, key_id="SYNTHETICKEY", team_id="SYNTHETICTEAM",
        topic="com.example.synthetic", endpoint="https://api.push.apple.com",
        http_client=httpx.Client(transport=httpx.MockTransport(failing_transport)),
    )
    app = receiver.create_app(vault, apns_sender=sender)
    conversation = chat.create_conversation(vault, conversation_id="network")
    conn = vault.connect()
    db.register_device_token(conn, "network-token", "production")
    conn.close()
    monkeypatch.setattr(receiver.Request, "is_disconnected",
                        lambda request: _true_async())
    monkeypatch.setattr(chat, "answer_question", lambda *args, **kwargs: {
        "text": "safe answer", "mode": "narration", "tool_trace": [],
        "verification": {},
    })
    with caplog.at_level("WARNING", logger=push.__name__):
        with TestClient(app) as client:
            response = client.post(
                "/v1/ask", json={"conversation_id": conversation["id"],
                                "question": "q"},
                headers=HEADERS,
            )
        _wait_for(lambda: any(
            record.name == push.__name__ and "APNs push failed" in record.message
            for record in caplog.records
        ))
    assert response.status_code == 200


def vault_has_token(vault, token):
    conn = vault.connect()
    try:
        return conn.execute(
            "SELECT 1 FROM device_tokens WHERE token = ?", (token,)
        ).fetchone() is not None
    finally:
        conn.close()


async def _true_async():
    return True


def test_receiver_main_builds_apns_config_at_entry_point(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(receiver.VaultContext, "local",
                        lambda *args, **kwargs: object())
    monkeypatch.setitem(receiver.__dict__, "uvicorn", None)
    import sys
    monkeypatch.setitem(sys.modules, "uvicorn", type(
        "Uvicorn", (), {"run": staticmethod(lambda app, **kwargs: None)}))

    def app_factory(ctx, **kwargs):
        captured.append(kwargs)
        return object()

    assert receiver.main([
        "--vault", str(tmp_path / "vault.db"),
        "--apns-key", str(tmp_path / "synthetic.p8"),
        "--apns-key-id", "SYNTHETICKEY",
        "--apns-team-id", "SYNTHETICTEAM",
        "--apns-topic", "com.example.synthetic",
        "--apns-endpoint", "https://api.push.apple.com",
    ], app_factory=app_factory) == 0
    config = captured[-1]["apns_config"]
    assert config.key_id == "SYNTHETICKEY"
    assert config.team_id == "SYNTHETICTEAM"
    assert captured[-1]["analyst_corpus_path"] is None

    captured.clear()
    monkeypatch.setenv("HA_APNS_KEY_PATH", "env-synthetic.p8")
    monkeypatch.setenv("HA_APNS_KEY_ID", "ENVKEY")
    monkeypatch.setenv("HA_APNS_TEAM_ID", "ENVTEAM")
    monkeypatch.setenv("HA_APNS_TOPIC", "com.example.env")
    monkeypatch.setenv("HA_APNS_ENDPOINT", "https://api.push.apple.com")
    assert receiver.main(["--vault", str(tmp_path / "vault.db")],
                         app_factory=app_factory) == 0
    config = captured[-1]["apns_config"]
    assert config.key_id == "ENVKEY"
    assert config.team_id == "ENVTEAM"


def test_receiver_without_apns_settings_passes_no_sender_config(
        tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(receiver.VaultContext, "local",
                        lambda *args, **kwargs: object())
    import sys
    monkeypatch.setitem(sys.modules, "uvicorn", type(
        "Uvicorn", (), {"run": staticmethod(lambda app, **kwargs: None)}))

    def app_factory(ctx, **kwargs):
        captured.append(kwargs)
        return object()

    assert receiver.main(["--vault", str(tmp_path / "vault.db")],
                         app_factory=app_factory) == 0
    assert captured[-1]["apns_config"] is None
