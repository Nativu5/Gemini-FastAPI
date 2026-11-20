"""Shared fixtures: one configured account, and the real app in front of a scripted Gemini."""

import asyncio
import itertools
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from gemini_webapi.types import Candidate, GeneratedVideo, Image, ModelOutput

from app.main import create_app
from app.server import chat, gemini, middleware
from app.services.lmdb import LMDBConversationStore
from app.utils import g_config
from app.utils.config import GeminiClientSettings, GeminiGemPolicyMode

CLIENT_ID = "client-id-1"
MODEL = "gemini-test"


@pytest.fixture(autouse=True)
def _one_configured_client(monkeypatch):
    """Stored chats are looked up per configured client id, so pin the ids rather than read them
    from whatever config.yaml and the environment hold."""
    monkeypatch.setattr(g_config.gemini, "clients", [GeminiClientSettings(id=CLIENT_ID)])


def reply(*deltas: str, **last_chunk: Any) -> list[ModelOutput]:
    """One Gemini turn as the chunks it streams, each carrying the text so far and the new piece.

    `last_chunk` sets further candidate fields on the final chunk, e.g. `web_images=[...]`.
    """
    chunks, text = [], ""
    pieces = deltas or ("",)
    for i, delta in enumerate(pieces):
        text += delta
        fields: dict[str, Any] = {"rcid": "rc", "text": text, "text_delta": delta or None}
        if i == len(pieces) - 1:
            fields.update(last_chunk)
        chunks.append(ModelOutput(metadata=[], chosen=0, candidates=[Candidate(**fields)]))
    return chunks


class _Session:
    def __init__(self, upstream: SimpleNamespace, client: "_Client", **kwargs: Any) -> None:
        self._upstream = upstream
        self.geminiclient = client
        self.gem = kwargs.get("gem")
        self.metadata = list(kwargs.get("metadata") or [None, None, None])

    def _begin(self, text: str, kwargs: dict[str, Any]) -> list[ModelOutput]:
        self._upstream.sent.append(
            SimpleNamespace(
                text=text,
                files=list(kwargs.get("files") or []),
                gem=self.gem,
                continues=self.metadata[0],
            )
        )
        turn = self._upstream.replies.pop(0) if self._upstream.replies else reply("ok")
        if isinstance(turn, Exception):
            raise turn
        n = next(self._upstream.turn_numbers)
        for chunk in turn:
            chunk.metadata = [self.metadata[0] or f"c_{n}", f"r_{n}", f"rc_{n}"]
        return turn

    async def send_message(self, text: str, **kwargs: Any) -> ModelOutput:
        turn = self._begin(text, kwargs)
        self.metadata = turn[-1].metadata
        return turn[-1]

    async def send_message_stream(self, text: str, **kwargs: Any):
        turn = self._begin(text, kwargs)
        for chunk in turn:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
            await asyncio.sleep(0)  # as a network read would: lets background downloads start
        self.metadata = turn[-1].metadata


class _Client:
    id = CLIENT_ID
    latest_chat_cid = None

    def __init__(self, upstream: SimpleNamespace) -> None:
        self._upstream = upstream

    def is_guest(self) -> bool:
        return False

    def usable_model(self, model: Any) -> Any:
        return model

    def chat_scope(self, temporary: bool) -> None:
        return None

    def start_chat(self, **kwargs: Any) -> _Session:
        return _Session(self._upstream, self, **kwargs)

    async def system_prompt_gem_id_or_create(self, prompt: str) -> str | None:
        if g_config.gemini.gems.policy == GeminiGemPolicyMode.OFF:
            return None
        gems = self._upstream.gems
        return gems.setdefault(prompt, f"gem_{len(gems) + 1}")


@pytest.fixture
def api(tmp_path, monkeypatch) -> Iterator[SimpleNamespace]:
    """The real app with Gemini replaced by a script.

    `api.upstream.replies` queues what Gemini answers (a `reply(...)` or an exception to raise;
    "ok" once the queue is empty; an exception after the chunks of a reply fails its stream there), `api.upstream.sent` records what it was asked,
    `api.upstream.downloaded` the urls of the generated files that were fetched, and
    `api.post(path, json=...)` sends a request. Conversations go to a store under `tmp_path`.
    """
    upstream = SimpleNamespace(
        replies=[], sent=[], gems={}, downloaded=[], turn_numbers=itertools.count(1)
    )
    client = _Client(upstream)

    class _Pool:
        async def acquire(self, client_id: str | None = None, require_account: bool = False):
            return client

    store = LMDBConversationStore.open_isolated(db_path=str(tmp_path / "lmdb"))
    monkeypatch.setattr(chat, "LMDBConversationStore", lambda: store)
    for module in (chat, gemini):
        monkeypatch.setattr(module, "GeminiClientPool", _Pool)
        monkeypatch.setattr(module, "_resolve_model_name", lambda pool, name: MODEL)
    (tmp_path / "media").mkdir()

    def download(item: Image | GeneratedVideo, path: str, suffix: str) -> str:
        """Stands in for a download: a file's bytes are its url without the fragment, so two
        urls that differ only there are two copies of one file."""
        upstream.downloaded.append(item.url)
        target = Path(path) / f"download-{uuid.uuid4().hex}{suffix}"
        target.write_bytes(item.url.partition("#")[0].encode())
        return str(target)

    async def save_image(image: Image, path: str = "temp", **kwargs: Any) -> str:
        return download(image, path, ".png")

    async def save_video(video: GeneratedVideo, path: str = "temp", **kwargs: Any) -> dict:
        return {"video": download(video, path, ".mp4")}

    monkeypatch.setattr(Image, "save", save_image)
    monkeypatch.setattr(GeneratedVideo, "save", save_video)
    monkeypatch.setattr(middleware, "MEDIA_STORE_DIR", tmp_path / "media")
    monkeypatch.setattr(g_config.server, "api_key", None)
    # No `with`: the lifespan would log in to Google.
    http = TestClient(create_app(), raise_server_exceptions=False)
    try:
        yield SimpleNamespace(post=http.post, get=http.get, upstream=upstream, store=store)
    finally:
        store.close()
