"""The three API surfaces end to end: real routes, conversion, storage and serialisation, with only
Gemini itself replaced by the script in `conftest.api`."""

import json
import re

import pytest
from conftest import MODEL, reply
from gemini_webapi.types import GeneratedVideo, WebImage

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}
TOOL_CALL = (
    "[ToolCalls][Call:get_weather][CallParameter:city]Hanoi[/CallParameter][/Call][/ToolCalls]"
)


def test_responses_offers_a_function_tool_declared_without_strict(api):
    """`strict` is optional in the Responses API; leaving it out must not lose the tool."""
    api.upstream.replies.append(reply(TOOL_CALL))
    response = api.post(
        "/v1/responses",
        json={
            "model": MODEL,
            "input": "Weather in Hanoi?",
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200, response.text
    assert "get_weather" in api.upstream.sent[0].text
    body = response.json()
    assert [item["name"] for item in body["output"] if item["type"] == "function_call"] == [
        "get_weather"
    ]
    assert body["tools"] == [WEATHER_TOOL]


def _continued(api) -> bool:
    """Whether the last request went into the chat the first one opened, as only its new part."""
    first, last = api.upstream.sent[0], api.upstream.sent[-1]
    return last.continues == "c_1" and "First question" not in last.text and first.continues is None


def _responses_output(response, stream: bool) -> list[dict]:
    """The output items of a Responses reply, read from the final event when it was streamed."""
    assert response.status_code == 200, response.text
    if not stream:
        return response.json()["output"]
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    return next(e for e in events if e["type"] == "response.completed")["response"]["output"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("first_reply", "follow_up"),
    [
        (reply("Hel", "lo."), [{"role": "user", "content": "Second question"}]),
        (
            reply("Answer.", thoughts="Let me think."),
            [{"role": "user", "content": "Second question"}],
        ),
        (
            reply("Checking. " + TOOL_CALL),
            [{"type": "function_call_output", "call_id": "<call_id>", "output": "Sunny"}],
        ),
    ],
    ids=["text", "reasoning", "tool-call"],
)
def test_responses_continues_the_chat_when_the_client_replays_the_output_items(
    api, stream, first_reply, follow_up
):
    """Sending the previous `output` back as input is how the OpenAI SDKs carry a conversation."""
    api.upstream.replies.append(first_reply)
    question = {"role": "user", "content": "First question"}
    request = {"model": MODEL, "tools": [WEATHER_TOOL], "stream": stream}
    first = api.post("/v1/responses", json={**request, "input": [question]})
    output = _responses_output(first, stream)
    call_ids = [item["call_id"] for item in output if item["type"] == "function_call"]
    follow_up = [
        {**item, "call_id": call_ids[0]} if "call_id" in item else item for item in follow_up
    ]

    second = api.post("/v1/responses", json={**request, "input": [question, *output, *follow_up]})

    assert second.status_code == 200, second.text
    assert _continued(api)


@pytest.mark.parametrize("stream", [False, True])
def test_responses_replays_a_generated_image_as_its_link_not_its_bytes(api, stream):
    api.upstream.replies.append(
        reply("Here.", web_images=[WebImage(url="http://img/a", title="A")])
    )
    question = {"role": "user", "content": "First question"}
    request = {"model": MODEL, "tools": [{"type": "image_generation"}], "stream": stream}
    first = api.post("/v1/responses", json={**request, "input": [question]})
    output = _responses_output(first, stream)
    image = next(item for item in output if item["type"] == "image_generation_call")
    again = {"role": "user", "content": "Second question"}

    api.post("/v1/responses", json={**request, "input": [question, *output, again]})
    assert _continued(api)

    # A history that matches no stored chat is sent in full, and still without the image bytes.
    other = {"role": "user", "content": "Another opening"}
    api.post("/v1/responses", json={**request, "input": [other, *output, again]})
    replayed = api.upstream.sent[-1]
    assert replayed.continues is None
    assert "/media/img_" in replayed.text
    assert image["result"] not in replayed.text


def test_chat_completions_continues_the_chat_when_the_reply_comes_back_as_text_parts(api):
    """Some clients send every message as a list of parts; that is the same text as one string."""
    api.upstream.replies.append(reply("Hello."))
    question = {"role": "user", "content": [{"type": "text", "text": "First question"}]}
    api.post("/v1/chat/completions", json={"model": MODEL, "messages": [question]})

    api.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello."}]},
                {"role": "user", "content": "Second question"},
            ],
        },
    )

    assert _continued(api)


def test_gemini_native_continues_the_chat_after_a_function_response(api):
    """The native API has no call ids, so none may be invented that differ between requests."""
    url = f"/v1beta/models/{MODEL}:generateContent"
    tools = [{"functionDeclarations": [{k: v for k, v in WEATHER_TOOL.items() if k != "type"}]}]
    history = [{"role": "user", "parts": [{"text": "First question"}]}]
    api.upstream.replies += [reply(TOOL_CALL), reply("Sunny.")]

    for follow_up in (
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "get_weather", "response": {"sky": "clear"}}}],
        },
        {"role": "user", "parts": [{"text": "And tomorrow?"}]},
    ):
        response = api.post(url, json={"contents": history, "tools": tools})
        assert response.status_code == 200, response.text
        history = [*history, response.json()["candidates"][0]["content"], follow_up]

    api.post(url, json={"contents": history, "tools": tools})

    assert [sent.continues for sent in api.upstream.sent] == [None, "c_1", "c_1"]
    assert "clear" not in api.upstream.sent[2].text


def _sse(response) -> list[dict]:
    """The JSON payload of every SSE frame of a response."""
    assert response.status_code == 200, response.text
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]


def _native(api, stream: bool, **body) -> list[dict]:
    """Parts of a /v1beta reply, over all chunks when streamed."""
    method = "streamGenerateContent" if stream else "generateContent"
    response = api.post(f"/v1beta/models/{MODEL}:{method}", json=body)
    chunks = _sse(response) if stream else [response.json()]
    assert "error" not in chunks[-1], chunks[-1]
    return [
        part
        for chunk in chunks
        for candidate in chunk["candidates"]
        for part in candidate.get("content", {}).get("parts", [])
    ]


def _text(parts: list[dict]) -> str:
    return "".join(part["text"] for part in parts if "text" in part and not part.get("thought"))


def _reply_text(response, stream: bool) -> str:
    """The assistant's text in a Chat Completions or Responses reply, streamed or not."""
    if stream:
        events = _sse(response)
        if "choices" in events[0]:
            return "".join(
                e["choices"][0]["delta"].get("content") or "" for e in events if e["choices"]
            )
        body = next(e for e in events if e["type"] == "response.completed")["response"]
    else:
        assert response.status_code == 200, response.text
        body = response.json()
    if "choices" in body:
        return body["choices"][0]["message"]["content"]
    return "".join(part["text"] for item in body["output"] for part in item.get("content", []))


USER_TURN = [{"role": "user", "parts": [{"text": "First question"}]}]


def test_responses_stream_delivers_a_reply_that_was_held_back_until_the_end(api):
    """A lone "S" could be the start of an echoed hint, so the filter holds it until the stream
    ends. When it arrived only with the final chunk, no message item is open by then, and the
    text still has to reach the client."""
    api.upstream.replies.append(reply(text="S"))
    events = _sse(
        api.post("/v1/responses", json={"model": MODEL, "input": "Pick a letter", "stream": True})
    )

    assert [e["delta"] for e in events if e["type"] == "response.output_text.delta"] == ["S"]
    completed = next(e for e in events if e["type"] == "response.completed")["response"]
    assert [part["text"] for item in completed["output"] for part in item.get("content", [])] == [
        "S"
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_native_normalises_tool_arguments_whether_streamed_or_not(api, stream):
    """Gemini's text is brought to NFC before tool calls are read from it; here, e + accent."""
    api.upstream.replies.append(reply(TOOL_CALL.replace("Hanoi", "Cafe\u0301")))
    tools = [{"functionDeclarations": [{k: v for k, v in WEATHER_TOOL.items() if k != "type"}]}]

    parts = _native(api, stream, contents=USER_TURN, tools=tools)

    assert [part["functionCall"]["args"] for part in parts if "functionCall" in part] == [
        {"city": "Caf\u00e9"}
    ]


def test_gemini_native_stream_sends_text_that_only_the_final_chunk_carries(api):
    api.upstream.replies.append(reply("Hel", text="Hello there"))

    assert _text(_native(api, True, contents=USER_TURN)) == "Hello there"


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_native_links_a_generated_video(api, stream):
    api.upstream.replies.append(
        reply("Watch.", generated_videos=[GeneratedVideo(url="http://vid/1", title="Clip")])
    )

    text = _text(_native(api, stream, contents=USER_TURN))

    assert text.startswith("Watch.")
    assert re.search(r"\[Clip\]\(http://testserver/media/media_[0-9a-f]{32}\.mp4", text)


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_native_does_not_take_a_replayed_thought_for_the_answer(api, stream):
    api.upstream.replies.append(
        reply("Answer.", thoughts="Let me think.", thoughts_delta="Let me think.")
    )
    parts = _native(api, stream, contents=USER_TURN)
    assert [part["text"] for part in parts if part.get("thought")] == ["Let me think."]
    again = {"role": "user", "parts": [{"text": "Second question"}]}

    _native(api, stream, contents=[*USER_TURN, {"role": "model", "parts": parts}, again])
    assert _continued(api)

    # A history that matches no stored chat is sent in full, without the thought.
    other = [{"role": "user", "parts": [{"text": "Another opening"}]}]
    _native(api, stream, contents=[*other, {"role": "model", "parts": parts}, again])
    assert "Answer." in api.upstream.sent[-1].text
    assert "Let me think." not in api.upstream.sent[-1].text


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_a_second_copy_of_a_generated_image_is_linked_once(api, path, stream):
    """Gemini can list one image under two urls; the reply then shows it once, not twice."""
    copies = [WebImage(url="http://img/a#1", title="A"), WebImage(url="http://img/a#2", title="A")]
    api.upstream.replies.append(reply("Here.", web_images=copies))
    prompt = (
        {"messages": [{"role": "user", "content": "Draw"}]} if "chat" in path else {"input": "Draw"}
    )

    response = api.post(path, json={"model": MODEL, "stream": stream, **prompt})

    assert _reply_text(response, stream).count("/media/img_") == 1


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_native_continues_the_chat_after_a_generated_image(api, stream):
    api.upstream.replies.append(
        reply("Here.", web_images=[WebImage(url="http://img/a", title="A")])
    )
    parts = _native(api, stream, contents=USER_TURN)
    assert any("inlineData" in part for part in parts)
    again = {"role": "user", "parts": [{"text": "Second question"}]}

    _native(api, stream, contents=[*USER_TURN, {"role": "model", "parts": parts}, again])

    assert _continued(api)
    assert api.upstream.sent[-1].files == []


CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "First question"}]}


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", CHAT),
        ("/v1/responses", {"model": MODEL, "input": "First question"}),
    ],
)
def test_openai_routes_report_an_error_in_the_envelope_openai_clients_read(api, path, body):
    api.upstream.replies.append(RuntimeError("upstream refused"))
    failed = api.post(path, json=body)
    assert failed.status_code == 502
    assert failed.json() == {
        "error": {
            "message": "upstream refused",
            "type": "server_error",
            "param": None,
            "code": None,
        }
    }

    invalid = api.post(path, json={"model": MODEL})
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    assert ("messages" if "chat" in path else "input") in invalid.json()["error"]["message"]


def test_the_native_route_reports_an_error_in_googles_envelope(api):
    api.upstream.replies.append(RuntimeError("upstream refused"))
    failed = api.post(f"/v1beta/models/{MODEL}:generateContent", json={"contents": USER_TURN})
    assert failed.json() == {
        "error": {"code": 502, "message": "upstream refused", "status": "INTERNAL", "details": []}
    }


def test_a_chat_completion_message_carries_what_openai_sends_and_nothing_else(api):
    api.upstream.replies += [reply("Hello."), reply(TOOL_CALL)]
    tools = [
        {"type": "function", "function": {k: v for k, v in WEATHER_TOOL.items() if k != "type"}}
    ]

    text = api.post("/v1/chat/completions", json=CHAT).json()
    call = api.post("/v1/chat/completions", json={**CHAT, "tools": tools}).json()

    assert text["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello.",
        "refusal": None,
        "annotations": [],
    }
    message = call["choices"][0]["message"]
    assert message["content"] is None  # present and null, as clients index it without checking
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["get_weather"]
    assert sorted(text["usage"]) == [
        "completion_tokens",
        "completion_tokens_details",
        "prompt_tokens",
        "total_tokens",
    ]
