import asyncio

import pytest
from fastapi import Response

from app.server import health
from app.utils.config import GeminiConfig, GuestMode


class _Pool:
    def __init__(self, client_status: dict[str, bool]):
        self._client_status = client_status

    def status(self) -> dict[str, bool]:
        return self._client_status


class _Store:
    def stats(self) -> dict[str, int]:
        return {"entries": 1}


@pytest.mark.parametrize(
    ("guest_mode", "client_status", "expected_status", "expected_ok"),
    [
        (GuestMode.STRICT, {"healthy": True, "guest": False}, 503, False),
        (GuestMode.ADAPTIVE, {"healthy": True, "guest": False}, 200, True),
        (GuestMode.ADAPTIVE, {"guest-a": False, "guest-b": False}, 503, False),
        (GuestMode.PERMISSIVE, {"guest-a": False, "guest-b": False}, 200, True),
    ],
)
def test_guest_mode_controls_client_health_status(
    monkeypatch,
    guest_mode: GuestMode,
    client_status: dict[str, bool],
    expected_status: int,
    expected_ok: bool,
):
    monkeypatch.setattr(health, "GeminiClientPool", lambda: _Pool(client_status))
    monkeypatch.setattr(health, "LMDBConversationStore", _Store)
    monkeypatch.setattr(health.g_config.gemini, "guest_mode", guest_mode)
    response = Response()

    result = asyncio.run(health.health_check(response))

    assert response.status_code == expected_status
    assert result.ok is expected_ok


def test_guest_mode_defaults_to_adaptive():
    config = GeminiConfig(clients=[], auto_refresh=True, verbose=True)

    assert config.guest_mode == GuestMode.ADAPTIVE
