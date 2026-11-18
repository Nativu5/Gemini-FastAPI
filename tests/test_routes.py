"""The three API surfaces end to end: real routes, conversion, storage and serialisation, with only
Gemini itself replaced by the script in `conftest.api`."""

import json

import pytest
from conftest import MODEL, reply
from gemini_webapi.types import WebImage

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
