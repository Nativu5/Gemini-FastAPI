"""Chat Completions SSE assembly.

Drives the streaming generator directly over a fake upstream so the wire contract can be
asserted without a network: what the client sees, in what order, and how the stream ends.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import orjson
import pytest
from gemini_webapi.types import Candidate, ModelOutput, WebImage

from app.models.core import AppMessage
from app.models.models import ResponseCreateRequest, StructuredOutputRequirement
from app.server.chat import (
    _create_real_streaming_response,
    _create_responses_real_streaming_response,
)
from app.services.lmdb import LMDBConversationStore

CLIENT_ID = "client-id-1"
MODEL = "gemini-3-pro"
TOOL_CALL_OUTPUT = (
    "[ToolCalls][Call:get_weather][CallParameter:city]Hanoi[/CallParameter][/Call][/ToolCalls]"
)
OBJECT_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


def _requirement() -> StructuredOutputRequirement:
    return StructuredOutputRequirement(
        schema_name="r", schema=OBJECT_SCHEMA, instruction="", raw_format={}, strict=True
    )


def _output(text: str, *, delta: str | None = None, thoughts_delta: str | None = None):
    return ModelOutput(
        metadata=["c", "r", "rc"],
        chosen=0,
        candidates=[
            Candidate(
                rcid="rc",
                text=text,
                text_delta=delta if delta is not None else text,
                thoughts_delta=thoughts_delta,
            )
        ],
    )


def _stream(*outputs: ModelOutput):
    async def generator():
        for output in outputs:
            yield output

    return generator()


@pytest.fixture
def db(tmp_path):
    opened = LMDBConversationStore.open_isolated(db_path=str(tmp_path / "lmdb"))
    try:
        yield opened
    finally:
        opened.close()


def _collect(db, stream, *, structured_requirement=None, tool_choice=None) -> list[str]:
    """Run the streaming response to completion and return its raw SSE frames."""
    # Only the few attributes the generator actually touches; a real client would need a live
    # browser session behind it.
    client = cast(
        Any,
        SimpleNamespace(id=CLIENT_ID, latest_chat_cid=None, chat_scope=lambda _temporary: None),
    )
    session = cast(Any, SimpleNamespace(metadata=["c", "r", "rc"]))
    response = _create_real_streaming_response(
        stream,
        "chatcmpl-test",
        0,
        MODEL,
        [AppMessage(role="user", content="hi")],
        db,
        MODEL,
        client,
        session,
        "http://testserver/",
        structured_requirement,
        tool_choice,
    )

    async def drain() -> list[str]:
        chunks: list[str] = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, str) else bytes(chunk).decode("utf-8"))
        return chunks

    return asyncio.run(drain())


def _payloads(frames: list[str]) -> list[dict]:
    return [
        orjson.loads(line[len("data: ") :])
        for frame in frames
        for line in frame.strip().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_a_plain_stream_opens_with_a_role_delta_and_ends_with_done(db):
    frames = _collect(db, _stream(_output("Hello"), _output("Hello world", delta=" world")))

    assert frames[-1] == "data: [DONE]\n\n"
    payloads = _payloads(frames)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload.get("choices")
    )
    assert text == "Hello world"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_reasoning_is_streamed_on_its_own_delta_field(db):
    frames = _collect(db, _stream(_output("answer", thoughts_delta="thinking")))
    reasoning = [
        payload["choices"][0]["delta"]["reasoning_content"]
        for payload in _payloads(frames)
        if payload.get("choices") and "reasoning_content" in payload["choices"][0]["delta"]
    ]
    assert reasoning == ["thinking"]


def test_a_tool_call_is_reported_and_finishes_as_tool_calls(db):
    frames = _collect(db, _stream(_output(TOOL_CALL_OUTPUT)))
    payloads = _payloads(frames)

    tool_calls = [
        payload["choices"][0]["delta"]["tool_calls"]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_calls
    assert tool_calls[0][0]["function"]["name"] == "get_weather"
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    # The protocol markers themselves must never reach the client.
    assert "[ToolCalls]" not in "".join(frames)


def test_structured_output_is_withheld_until_it_has_been_validated(db):
    """Deltas are suppressed so a schema violation cannot arrive half-rendered."""
    frames = _collect(
        db,
        _stream(_output('```json\n{"a": "x"}\n```')),
        structured_requirement=_requirement(),
    )
    contents = [
        payload["choices"][0]["delta"].get("content", "")
        for payload in _payloads(frames)
        if payload.get("choices")
    ]
    assert "".join(contents) == '{"a":"x"}'
    assert frames[-1] == "data: [DONE]\n\n"


def test_a_strict_schema_violation_ends_the_stream_with_an_error_and_a_terminator(db):
    frames = _collect(
        db,
        _stream(_output('{"wrong": true}')),
        structured_requirement=_requirement(),
    )

    assert frames[-1].endswith("data: [DONE]\n\n")
    error = _payloads(frames)[-1]["error"]
    assert error["type"] == "invalid_model_output"
    assert error["code"] == "schema_validation_failed"


def test_an_unmet_forced_tool_choice_ends_the_stream_with_an_error(db):
    frames = _collect(db, _stream(_output("just prose")), tool_choice="required")

    assert frames[-1].endswith("data: [DONE]\n\n")
    error = _payloads(frames)[-1]["error"]
    assert error["param"] == "tool_choice"
    assert error["code"] == "required_tool_missing"


def test_a_volunteered_image_does_not_satisfy_a_forced_tool_choice(db, monkeypatch):
    """Chat Completions has no image tool, so an image cannot stand in for a forced call.

    The image also covers cleanup: media downloads spawned while chunks arrive have to be
    cancelled on an early error return, not left writing files nothing will reference.
    """
    started: list[asyncio.Task] = []

    async def slow_download(_img):
        started.append(cast(asyncio.Task, asyncio.current_task()))
        await asyncio.sleep(5)

    monkeypatch.setattr("app.server.chat._process_image_item", slow_download)

    first = _output("just prose")
    first.candidates[0].web_images = [WebImage(url="http://127.0.0.1:1/x.png")]

    async def stream_with_a_scheduling_gap():
        yield first
        # Let the spawned task start, so cancelling it is observable.
        await asyncio.sleep(0)
        yield _output("just prose", delta="")

    frames = _collect(db, stream_with_a_scheduling_gap(), tool_choice="required")

    error = _payloads(frames)[-1]["error"]
    assert error["code"] == "required_tool_missing"
    assert frames[-1].endswith("data: [DONE]\n\n")
    assert started
    assert all(task.cancelled() for task in started)


def test_an_upstream_failure_mid_stream_is_reported_and_terminated(db):
    async def failing():
        yield _output("partial")
        raise RuntimeError("upstream went away")

    frames = _collect(db, failing())

    assert frames[-1].endswith("data: [DONE]\n\n")
    error = _payloads(frames)[-1]["error"]
    assert error["type"] == "server_error"
    assert "upstream went away" in error["message"]


def test_a_completed_turn_is_persisted_for_reuse(db):
    """The answer is stored with the prompt, so the next turn can resume this chat."""
    _collect(db, _stream(_output("Hello")))

    stored = db.find(
        MODEL,
        [AppMessage(role="user", content="hi"), AppMessage(role="assistant", content="Hello")],
    )
    assert stored is not None
    assert stored.client_id == CLIENT_ID
    assert stored.metadata == ["c", "r", "rc"]


def test_a_structured_turn_is_reusable_by_replaying_what_the_client_received(db):
    """What is streamed has to equal what is stored, or the next turn cannot match the prefix.

    Withholding the deltas is what makes this hold: the client is sent the validated document,
    which is exactly the form persisted, rather than the raw fenced text around it.
    """
    frames = _collect(
        db,
        _stream(_output('```json\n{"a": "x"}\n```')),
        structured_requirement=_requirement(),
    )
    received = "".join(
        payload["choices"][0]["delta"].get("content") or ""
        for payload in _payloads(frames)
        if payload.get("choices")
    )
    assert received == '{"a":"x"}'

    stored = db.find(
        MODEL,
        [AppMessage(role="user", content="hi"), AppMessage(role="assistant", content=received)],
    )
    assert stored is not None


def test_a_failed_turn_is_not_persisted(db):
    _collect(
        db,
        _stream(_output('{"wrong": true}')),
        structured_requirement=_requirement(),
    )
    assert db.keys() == []


def test_responses_schema_failure_cancels_pending_media_tasks(db, monkeypatch):
    started: list[asyncio.Task] = []

    async def slow_download(_img):
        started.append(cast(asyncio.Task, asyncio.current_task()))
        await asyncio.sleep(5)

    monkeypatch.setattr("app.server.chat._process_image_item", slow_download)
    first = _output('{"wrong": true}')
    first.candidates[0].web_images = [WebImage(url="http://127.0.0.1:1/x.png")]

    async def stream_with_a_scheduling_gap():
        yield first
        await asyncio.sleep(0)

    client = cast(
        Any,
        SimpleNamespace(id=CLIENT_ID, latest_chat_cid=None, chat_scope=lambda _temporary: None),
    )
    session = cast(Any, SimpleNamespace(metadata=["c", "r", "rc"]))
    response = _create_responses_real_streaming_response(
        stream_with_a_scheduling_gap(),
        "resp-test",
        0,
        MODEL,
        [AppMessage(role="user", content="hi")],
        db,
        MODEL,
        client,
        session,
        ResponseCreateRequest(model=MODEL, input="hi", stream=True),
        "http://testserver/",
        _requirement(),
    )

    async def drain():
        frames = []
        async for chunk in response.body_iterator:
            frames.append(chunk if isinstance(chunk, str) else bytes(chunk).decode("utf-8"))
        return frames, [task.cancelled() for task in started]

    frames, cancelled = asyncio.run(drain())
    assert any("schema_validation_failed" in frame for frame in frames)
    assert started
    assert all(cancelled)


def test_stream_disconnect_cancels_pending_media_tasks(db, monkeypatch):
    started: list[asyncio.Task] = []

    async def slow_download(_img):
        started.append(cast(asyncio.Task, asyncio.current_task()))
        await asyncio.sleep(5)

    monkeypatch.setattr("app.server.chat._process_image_item", slow_download)
    first = _output("Here is the image: ")
    first.candidates[0].web_images = [WebImage(url="http://127.0.0.1:1/x.png")]

    async def infinite_stream():
        yield first
        while True:
            await asyncio.sleep(0.1)
            yield _output("more")

    client = cast(
        Any,
        SimpleNamespace(id=CLIENT_ID, latest_chat_cid=None, chat_scope=lambda _temporary: None),
    )
    session = cast(Any, SimpleNamespace(metadata=["c", "r", "rc"]))
    response = _create_real_streaming_response(
        infinite_stream(),
        "chatcmpl-test",
        0,
        MODEL,
        [AppMessage(role="user", content="hi")],
        db,
        MODEL,
        client,
        session,
        "http://testserver/",
    )

    async def abort_early():
        it = cast(Any, response.body_iterator)
        await it.__anext__()
        await it.__anext__()
        await it.__anext__()
        await asyncio.sleep(0)
        await it.aclose()
        return [task.cancelled() for task in started]

    cancelled = asyncio.run(abort_early())
    assert started
    assert all(cancelled)


def test_responses_disconnect_cancels_pending_media_tasks(db, monkeypatch):
    started: list[asyncio.Task] = []

    async def slow_download(_img):
        started.append(cast(asyncio.Task, asyncio.current_task()))
        await asyncio.sleep(5)

    monkeypatch.setattr("app.server.chat._process_image_item", slow_download)
    first = _output("Here is the image: ")
    first.candidates[0].web_images = [WebImage(url="http://127.0.0.1:1/x.png")]

    async def infinite_stream():
        yield first
        while True:
            await asyncio.sleep(0.1)
            yield _output("more")

    client = cast(
        Any,
        SimpleNamespace(id=CLIENT_ID, latest_chat_cid=None, chat_scope=lambda _temporary: None),
    )
    session = cast(Any, SimpleNamespace(metadata=["c", "r", "rc"]))
    response = _create_responses_real_streaming_response(
        infinite_stream(),
        "resp-test",
        0,
        MODEL,
        [AppMessage(role="user", content="hi")],
        db,
        MODEL,
        client,
        session,
        ResponseCreateRequest(model=MODEL, input="hi", stream=True),
        "http://testserver/",
    )

    async def abort_early():
        it = cast(Any, response.body_iterator)
        # Consume events until media tasks are scheduled
        for _ in range(6):
            await it.__anext__()
        await asyncio.sleep(0)
        await it.aclose()
        return [task.cancelled() for task in started]

    cancelled = asyncio.run(abort_early())
    assert started
    assert all(cancelled)
