"""What a stream does when it ends before its reply is finished: the generated files still
downloading are abandoned, not left running. Driven directly, since a test client reads a stream
to its end. Also the filter that keeps protocol markers out of streamed text."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from conftest import reply
from gemini_webapi.types import WebImage

from app.models.core import AppMessage
from app.models.models import ResponseCreateRequest
from app.server import media
from app.server.chat import (
    StreamingOutputFilter,
    _create_real_streaming_response,
    _create_responses_real_streaming_response,
    _Sent,
)
from app.server.gemini import _stream_reply

MODEL = "gemini-test"
TOOL_CALL_OUTPUT = (
    "[ToolCalls][Call:get_weather][CallParameter:city]Hanoi[/CallParameter][/Call][/ToolCalls]"
)
HI = [AppMessage(role="user", content="hi")]
IMAGE = [WebImage(url="http://img/a")]
UNREAD = cast(Any, None)  # the session: a stream cut short never reaches it


@pytest.fixture
def downloads(monkeypatch) -> list[asyncio.Task]:
    """The downloads started, each of which runs until it is cancelled."""
    started: list[asyncio.Task] = []

    async def slow_download(item):
        started.append(cast(asyncio.Task, asyncio.current_task()))
        await asyncio.sleep(5)

    monkeypatch.setattr(media, "_download", slow_download)
    return started


async def _endless_reply():
    yield reply("Here is the image: ", web_images=IMAGE)[0]
    while True:
        await asyncio.sleep(0.1)
        yield reply("more")[0]


def _chat_completions(output):
    return _create_real_streaming_response(
        _Sent(output, UNREAD, MODEL, HI), "id", 0, MODEL, "http://testserver/"
    )


def _responses(output):
    request = ResponseCreateRequest(model=MODEL, input="hi", stream=True)
    return _create_responses_real_streaming_response(
        _Sent(output, UNREAD, MODEL, HI), "id", 0, request, "http://testserver/"
    )


@pytest.mark.parametrize("respond", [_chat_completions, _responses])
def test_a_client_that_goes_away_stops_the_downloads(downloads, respond):
    async def go_away_mid_stream() -> list[bool]:
        frames = cast(Any, respond(_endless_reply()).body_iterator)
        while not downloads:
            await anext(frames)
        await frames.aclose()
        return [task.cancelled() for task in downloads]  # before `asyncio.run` cancels the rest

    assert asyncio.run(go_away_mid_stream()) == [True]


def test_a_native_stream_that_ends_in_an_error_stops_the_downloads(downloads):
    async def one_chunk():
        yield reply("Just prose.", web_images=IMAGE)[0]
        await asyncio.sleep(0)  # lets the download start

    # Only what is read before the unmet tool choice ends the stream.
    output = SimpleNamespace(output=one_chunk())
    sent = SimpleNamespace(sent=output, structured_requirement=None, tool_choice="required")

    async def read_to_the_end() -> tuple[str, list[bool]]:
        frames = [frame async for frame in _stream_reply(cast(Any, sent), "http://testserver/")]
        return frames[-1], [task.cancelled() for task in downloads]

    last, cancelled = asyncio.run(read_to_the_end())
    assert '"error"' in last
    assert cancelled == [True]


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("Let me check. " + TOOL_CALL_OUTPUT + " Done.", "Let me check. Done."),
        ("<|im_start|>assistant\nHello there.<|im_end|>", "Hello there."),
        ("See [1], a <b>tag</b> and a \\ backslash.", "See [1], a <b>tag</b> and a \\ backslash."),
        ("Cost: [not a marker", "Cost: "),
    ],
    ids=["tool-call", "chatml", "ordinary-brackets", "unfinished-bracket"],
)
def test_the_stream_filter_shows_the_same_text_however_the_reply_is_chunked(raw, shown):
    """Gemini cuts its stream anywhere, including in the middle of a marker."""

    def filtered(*pieces: str) -> str:
        output_filter = StreamingOutputFilter()
        return "".join(output_filter.process(piece) for piece in pieces) + output_filter.flush()

    assert filtered(raw) == shown
    assert {filtered(raw[:cut], raw[cut:]) for cut in range(1, len(raw))} == {shown}
    assert filtered(*raw) == shown
