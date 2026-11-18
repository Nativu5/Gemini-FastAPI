"""Which stored chats may be continued under the active chat mode.

`auto_delete` promises two things at once: a chat it opened is removed from the Google account
once idle, and a chat opened in any other mode is never removed. Both hold only if a Google chat
is written to under one mode for its whole life.
"""

from types import SimpleNamespace
from typing import cast

import pytest

from app.models.core import AppMessage, ConversationInStore
from app.server.chat import _is_reusable_chat
from app.services.client import GeminiClientWrapper
from app.utils import g_config
from app.utils.config import ChatMode

CLIENT_ID = "client-id-1"


def _stored(chat_mode: ChatMode | None) -> ConversationInStore:
    return ConversationInStore(
        model="gemini-3-pro",
        client_id=CLIENT_ID,
        metadata=["c_1", "r_1", "rc_1"],
        messages=[
            AppMessage(role="user", content="hello"),
            AppMessage(role="assistant", content="hi there"),
        ],
        chat_mode=chat_mode,
    )


def _account_client() -> GeminiClientWrapper:
    """An authenticated client: no ephemeral scope, so only the chat mode decides reuse."""
    client = SimpleNamespace(
        id=CLIENT_ID, latest_chat_cid="c_1", chat_scope=lambda _temporary: None
    )
    return cast(GeminiClientWrapper, client)


@pytest.mark.parametrize(
    ("stored_mode", "active_mode", "reusable"),
    [
        (ChatMode.NORMAL, ChatMode.NORMAL, True),
        (None, ChatMode.NORMAL, True),  # stored before chat modes were recorded
        (ChatMode.AUTO_DELETE, ChatMode.AUTO_DELETE, True),
        # Continuing these would tag a chat the account was promised to keep for deletion.
        (ChatMode.NORMAL, ChatMode.AUTO_DELETE, False),
        (None, ChatMode.AUTO_DELETE, False),
        # Continuing this would keep a chat alive that was promised to be deleted.
        (ChatMode.AUTO_DELETE, ChatMode.NORMAL, False),
    ],
)
def test_a_chat_is_never_continued_across_the_auto_delete_boundary(
    stored_mode, active_mode, reusable, monkeypatch
):
    monkeypatch.setattr(g_config.gemini, "chat_mode", active_mode)

    assert _is_reusable_chat(_stored(stored_mode), _account_client(), temporary=False) is reusable
