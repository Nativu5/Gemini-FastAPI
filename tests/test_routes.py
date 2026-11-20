"""The three API surfaces end to end: real routes, conversion, storage and serialisation, with only
Gemini itself replaced by the script in `conftest.api`."""

import base64
import json
import re

import pytest
from conftest import MODEL, reply
from gemini_webapi.exceptions import ModelInvalidError
from gemini_webapi.types import GeneratedVideo, WebImage

from app.models.models import ModelData
from app.server import chat, gemini
from app.utils import g_config
from app.utils.config import GeminiGemPolicyMode

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


def test_gemini_native_shows_the_model_a_response_schema_it_cannot_check(api):
    """Google's API takes array and string bounds as strings, which JSON Schema does not. Such a
    schema cannot be used to check the reply, but the model still has to be told what is wanted."""
    api.upstream.replies.append(reply('```json\n["hiking"]\n```'))
    schema = {"type": "ARRAY", "minItems": "1", "maxItems": "3", "items": {"type": "STRING"}}
    config = {"responseMimeType": "application/json", "responseSchema": schema}

    parts = _native(api, False, contents=USER_TURN, generationConfig=config)

    assert '"maxItems":"3"' in api.upstream.sent[0].text
    assert _text(parts) == '["hiking"]'


CITY_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


@pytest.mark.parametrize("endpoint", ["chat", "responses", "native"])
def test_a_continued_chat_is_sent_the_output_format_again(api, endpoint):
    """Gemini is only asked for JSON in the prompt, and stops returning it a few turns after
    the last time it was asked."""
    api.upstream.replies += [reply('```json\n{"city": "Paris"}\n```')] * 2
    first, again = "First question", "Second question"
    if endpoint == "chat":
        schema = {"type": "json_schema", "json_schema": {"name": "city", "schema": CITY_SCHEMA}}
        request = {"model": MODEL, "response_format": schema}
        opening = [{"role": "user", "content": first}]
        answer = api.post("/v1/chat/completions", json={**request, "messages": opening}).json()
        history = [*opening, answer["choices"][0]["message"], {"role": "user", "content": again}]
        api.post("/v1/chat/completions", json={**request, "messages": history})
    elif endpoint == "responses":
        schema = {"type": "json_schema", "name": "city", "schema": CITY_SCHEMA}
        request = {"model": MODEL, "text": {"format": schema}}
        opening = [{"role": "user", "content": first}]
        output = api.post("/v1/responses", json={**request, "input": opening}).json()["output"]
        history = [*opening, *output, {"role": "user", "content": again}]
        api.post("/v1/responses", json={**request, "input": history})
    else:
        config = {"responseMimeType": "application/json", "responseJsonSchema": CITY_SCHEMA}
        parts = _native(api, False, contents=USER_TURN, generationConfig=config)
        history = [
            *USER_TURN,
            {"role": "model", "parts": parts},
            {"role": "user", "parts": [{"text": again}]},
        ]
        _native(api, False, contents=history, generationConfig=config)

    assert _continued(api)
    assert "STRUCTURED JSON PROTOCOL" in api.upstream.sent[-1].text
    assert '"city"' in api.upstream.sent[-1].text


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


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_a_system_prompt_travels_as_a_gem_and_a_chat_is_only_continued_under_its_own(
    api, monkeypatch, path
):
    monkeypatch.setattr(g_config.gemini.gems, "policy", GeminiGemPolicyMode.CREATE_ON_DEMAND)

    def ask(system: str | None, *history: str) -> None:
        turns = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": t}
            for i, t in enumerate(history)
        ]
        if "chat" in path:
            messages = [{"role": "system", "content": system}] if system else []
            body = {"messages": [*messages, *turns]}
        else:
            body = {"input": turns, **({"instructions": system} if system else {})}
        assert api.post(path, json={"model": MODEL, **body}).status_code == 200

    ask("Be a pirate", "First question")
    ask("Be a pirate", "First question", "ok", "Second question")
    ask("Be a lawyer", "First question", "ok", "Second question")
    ask(None, "First question", "ok", "Second question")
    first, same_prompt, other_prompt, no_prompt = api.upstream.sent

    # The prompt is carried by the gem, and so is left out of the message.
    assert (first.gem, "Be a pirate" in first.text) == ("gem_1", False)
    # The same prompt continues the chat, under the same gem, with only the new turn.
    assert (same_prompt.gem, same_prompt.continues) == ("gem_1", "c_1")
    assert "First question" not in same_prompt.text
    # Another prompt, or none, is another conversation: it starts over with the whole history.
    assert (other_prompt.gem, other_prompt.continues) == ("gem_2", None)
    assert (no_prompt.gem, no_prompt.continues) == (None, None)
    assert "First question" in other_prompt.text
    assert "First question" in no_prompt.text


def test_the_native_route_sends_its_system_instruction_inline_even_with_gems_on(api, monkeypatch):
    monkeypatch.setattr(g_config.gemini.gems, "policy", GeminiGemPolicyMode.CREATE_ON_DEMAND)

    _native(api, False, contents=USER_TURN, systemInstruction={"parts": [{"text": "Be a pirate"}]})

    assert api.upstream.sent[0].gem is None
    assert "Be a pirate" in api.upstream.sent[0].text


def test_chat_completions_continues_the_chat_when_the_client_echoes_the_reasoning(api):
    """The stream carries `reasoning_content`; a client that sends it back still continues."""
    api.upstream.replies.append(reply("Answer.", thoughts="Hm.", thoughts_delta="Hm."))
    question = {"role": "user", "content": "First question"}
    api.post("/v1/chat/completions", json={"model": MODEL, "messages": [question]})
    echoed = {"role": "assistant", "content": "Answer.", "reasoning_content": "Hm."}
    again = {"role": "user", "content": "Second question"}

    api.post("/v1/chat/completions", json={"model": MODEL, "messages": [question, echoed, again]})

    assert _continued(api)
    assert "Hm." not in api.upstream.sent[-1].text


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


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_native_continues_the_chat_after_a_function_response(api, stream):
    """The native API has no call ids, so none may be invented that differ between requests."""
    tools = [{"functionDeclarations": [{k: v for k, v in WEATHER_TOOL.items() if k != "type"}]}]
    history = [{"role": "user", "parts": [{"text": "First question"}]}]
    api.upstream.replies += [reply("Let me check. ", TOOL_CALL), reply("Sunny.")]

    for follow_up in (
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "get_weather", "response": {"sky": "clear"}}}],
        },
        {"role": "user", "parts": [{"text": "And tomorrow?"}]},
    ):
        history = [*history, *_recorded(api, stream, contents=history, tools=tools), follow_up]

    _native(api, stream, contents=history, tools=tools)

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


def _recorded(api, stream: bool, **body) -> list[dict]:
    """A /v1beta reply as Google's SDKs keep it in a chat: one content for each chunk."""
    method = "streamGenerateContent" if stream else "generateContent"
    response = api.post(f"/v1beta/models/{MODEL}:{method}", json=body)
    chunks = _sse(response) if stream else [response.json()]
    return [chunk["candidates"][0]["content"] for chunk in chunks]


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


IMAGE = {"web_images": [WebImage(url="http://img/a", title="A")]}


@pytest.mark.parametrize(
    ("deltas", "extra"),
    [
        (("\n", "\n"), IMAGE),  # all the text Google sends with a generated image
        (("Here it is.",), {"thoughts": "Thinking.", "thoughts_delta": "Thinking."}),
        # The rest of the thoughts and of the text arrive with the last chunk: a thought is
        # streamed between two pieces of the text.
        (("Here ",), {"text": "Here it is.", "thoughts": "Thinking.", **IMAGE}),
    ],
)
def test_gemini_native_continues_a_chat_replayed_the_way_googles_sdks_record_a_stream(
    api, deltas, extra
):
    api.upstream.replies.append(reply(*deltas, **extra))
    recorded = _recorded(api, True, contents=USER_TURN)
    for content in recorded:
        # The SDKs leave the whole turn out of the history they send unless every chunk has
        # parts, and none of them empty.
        assert content["parts"]
        assert all(content["parts"])
    again = {"role": "user", "parts": [{"text": "Second question"}]}

    _native(api, True, contents=[*USER_TURN, *recorded, again])

    assert _continued(api)


def test_gemini_native_joins_only_the_contents_of_one_model_turn(api):
    """What reaches the model when the history matches no stored chat and is sent whole."""
    contents = [
        {"role": "user", "parts": [{"text": "First"}]},
        {"role": "user", "parts": [{"text": "question"}]},
        {"role": "model", "parts": [{"text": "Here "}]},
        {"role": "model", "parts": [{"text": "it is."}]},
        {"role": "user", "parts": [{"text": "Second question"}]},
    ]

    _native(api, False, contents=contents)

    assert "Here it is." in api.upstream.sent[0].text
    assert "Firstquestion" not in api.upstream.sent[0].text


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


def _streamed(api, path: str) -> tuple[str, str]:
    """The text and the thoughts a stream sent, added up from its deltas as a client does."""
    if path == "native":
        parts = _native(api, True, contents=USER_TURN)
        return _text(parts), "".join(part["text"] for part in parts if part.get("thought"))
    prompt = (
        {"messages": [{"role": "user", "content": "Hi"}]} if "chat" in path else {"input": "Hi"}
    )
    events = _sse(api.post(path, json={"model": MODEL, "stream": True, **prompt}))
    if "chat" in path:
        deltas = [event["choices"][0]["delta"] for event in events if event.get("choices")]
        text, thoughts = (
            "".join(d.get(key) or "" for d in deltas) for key in ("content", "reasoning_content")
        )
        return text, thoughts
    text, thoughts = (
        "".join(e["delta"] for e in events if e["type"] == f"response.{kind}.delta")
        for kind in ("output_text", "reasoning_summary_text")
    )
    return text, thoughts


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses", "native"])
@pytest.mark.parametrize(
    ("deltas", "last_chunk", "shown"),
    [
        (("Hel",), {"text": "Hello there", "thoughts": "Hm."}, ("Hello there", "Hm.")),
        # A final text that does not continue what was streamed is not added to it.
        (("Hello",), {"text": "Goodbye"}, ("Hello", "")),
    ],
)
def test_a_stream_sends_what_only_its_final_chunk_carries(api, path, deltas, last_chunk, shown):
    api.upstream.replies.append(reply(*deltas, **last_chunk))

    assert _streamed(api, path) == shown


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
def test_an_image_that_was_not_downloaded_does_not_satisfy_a_forced_image_tool(
    api, stream, monkeypatch
):
    async def fail(*args, **kwargs):
        raise OSError("download failed")

    monkeypatch.setattr(WebImage, "save", fail)
    api.upstream.replies.append(reply("Here.", **IMAGE))
    tool = {"type": "image_generation"}
    request = {"model": MODEL, "input": "Draw", "tools": [tool], "tool_choice": tool}

    response = api.post("/v1/responses", json={**request, "stream": stream})

    if stream:
        error = next(event for event in _sse(response) if event["type"] == "error")["error"]
    else:
        assert response.status_code == 502
        error = response.json()["error"]
    assert error["message"] == "The model did not return a required image generation result"


def _as_googles_python_sdk_sends_it(part: dict) -> dict:
    """google-genai writes the fields inside `inlineData` under their proto names, and the data
    in the URL-safe Base64 alphabet."""
    if inline := part.get("inlineData"):
        data = base64.urlsafe_b64encode(base64.b64decode(inline["data"])).decode()
        return {"inlineData": {"data": data, "mime_type": inline["mimeType"]}}
    return part


@pytest.mark.parametrize("python_sdk", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_gemini_native_continues_the_chat_after_a_generated_image(api, stream, python_sdk):
    # The image's Base64 has a `+`, which is `-` in the URL-safe alphabet.
    api.upstream.replies.append(
        reply("Here.", web_images=[WebImage(url="http://img/~a", title="A")])
    )
    parts = _native(api, stream, contents=USER_TURN)
    assert any("inlineData" in part for part in parts)
    if python_sdk:
        parts = [_as_googles_python_sdk_sends_it(part) for part in parts]
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
        "error": {"code": 502, "message": "upstream refused", "status": "INTERNAL"}
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


AUTH = {"Authorization": "Bearer secret"}


@pytest.fixture
def locked(api, monkeypatch):
    """The scripted app with an API key configured."""
    monkeypatch.setattr(g_config.server, "api_key", "secret")
    return api


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 403),
        ({"Authorization": "Basic secret"}, 401),
        (AUTH, 200),
    ],
)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_openai_routes_serve_only_a_request_that_carries_the_api_key(locked, path, headers, status):
    body = CHAT if "chat" in path else {"model": MODEL, "input": "First question"}

    assert locked.post(path, json=body, headers=headers).status_code == status
    assert len(locked.upstream.sent) == (status == 200)  # nothing reaches Gemini without it


@pytest.mark.parametrize(
    ("headers", "query", "status"),
    [
        ({}, "", 401),
        ({"x-goog-api-key": "secret"}, "", 200),
        ({}, "?key=secret", 200),
        (AUTH, "", 200),
        (
            {"x-goog-api-key": "wrong"},
            "?key=secret",
            403,
        ),  # a wrong key is not rescued by a right one
        (AUTH, "?key=wrong", 403),
    ],
)
def test_the_native_route_takes_the_key_each_way_google_clients_send_it(
    locked, headers, query, status
):
    url = f"/v1beta/models/{MODEL}:generateContent{query}"

    assert locked.post(url, json={"contents": USER_TURN}, headers=headers).status_code == status
    assert len(locked.upstream.sent) == (status == 200)


@pytest.mark.parametrize(
    ("headers", "query"), [({"x-goog-api-key": "secret"}, ""), ({}, "?key=secret")]
)
def test_google_sdks_reach_the_native_api_on_v1_where_openai_clients_keep_theirs(
    locked, monkeypatch, headers, query
):
    """Google's SDKs call /v1 unless told otherwise; they send the key the way Google takes it,
    and OpenAI clients as a bearer token."""
    monkeypatch.setattr(
        chat, "_AVAILABLE_MODELS_CACHE", [ModelData(id=MODEL, created=0, owned_by="google")]
    )
    models = locked.get(f"/v1/models{query}", headers=headers).json()
    assert [model["name"] for model in models["models"]] == [f"models/{MODEL}"]
    assert [model["id"] for model in locked.get("/v1/models", headers=AUTH).json()["data"]] == [
        MODEL
    ]

    url = f"/v1/models/{MODEL}:generateContent{query}"
    answered = locked.post(url, json={"contents": USER_TURN}, headers=headers)
    assert answered.json()["candidates"][0]["content"]["parts"] == [{"text": "ok"}]

    # A Google key sent along to an OpenAI route changes nothing.
    chat_headers = {**AUTH, "x-goog-api-key": "secret"}
    assert locked.post("/v1/chat/completions", json=CHAT, headers=chat_headers).status_code == 200


def test_google_sdks_get_errors_on_v1_in_googles_envelope(locked, monkeypatch):
    url = f"/v1/models/{MODEL}:generateContent"
    refused = locked.post(url, json={"contents": USER_TURN}, headers={"x-goog-api-key": "wrong"})
    assert refused.json()["error"]["status"] == "PERMISSION_DENIED"

    def broken(request):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(gemini, "_validate_gemini_request", broken)
    failed = locked.post(url, json={"contents": USER_TURN}, headers={"x-goog-api-key": "secret"})
    assert failed.json()["error"] == {"code": 500, "message": "unexpected", "status": "INTERNAL"}


def test_a_generated_file_is_served_only_through_the_signed_link_in_the_reply(locked):
    locked.upstream.replies.append(
        reply("Here.", web_images=[WebImage(url="http://img/a", title="A")])
    )
    text = locked.post("/v1/chat/completions", json=CHAT, headers=AUTH).json()["choices"][0][
        "message"
    ]["content"]
    link = re.search(r"\(http://testserver(/media/img_[0-9a-f]{32}\.png\?token=[0-9a-f]+)\)", text)
    assert link, text
    url = link.group(1)

    assert locked.get(url).content == b"http://img/a"  # no API key needed: the token is the grant
    assert locked.get(url.partition("?")[0]).status_code == 403
    assert locked.get(url[:-1] + ("0" if url[-1] != "0" else "1")).status_code == 403
    other_file = re.sub(r"img_[0-9a-f]{32}", "img_" + "0" * 32, url)
    assert locked.get(other_file).status_code == 403  # a token does not open another file


def test_a_chat_google_no_longer_has_is_replayed_in_full_in_a_new_one(api):
    api.post("/v1/chat/completions", json=CHAT)
    history = [
        *CHAT["messages"],
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Second question"},
    ]
    api.upstream.replies += [ModelInvalidError("chat is gone"), reply("Again.")]

    recovered = api.post("/v1/chat/completions", json={**CHAT, "messages": history})

    assert recovered.json()["choices"][0]["message"]["content"] == "Again."
    assert [sent.continues for sent in api.upstream.sent] == [None, "c_1", None]
    assert "First question" in api.upstream.sent[2].text

    # The new chat is remembered: the next turn continues it.
    next_turn = [
        *history,
        {"role": "assistant", "content": "Again."},
        {"role": "user", "content": "Third question"},
    ]
    api.post("/v1/chat/completions", json={**CHAT, "messages": next_turn})
    assert api.upstream.sent[3].continues not in (None, "c_1")
    assert "Second question" not in api.upstream.sent[3].text

    # The dead chat is forgotten: repeating the request that hit it does not try it again.
    api.post("/v1/chat/completions", json={**CHAT, "messages": history})
    assert [sent.continues for sent in api.upstream.sent[4:]] == [None]


def test_a_history_that_matches_no_stored_chat_is_sent_whole_to_a_new_one(api):
    api.post("/v1/chat/completions", json=CHAT)
    edited = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "something the server never said"},
        {"role": "user", "content": "Second question"},
    ]

    api.post("/v1/chat/completions", json={**CHAT, "messages": edited})

    assert api.upstream.sent[1].continues is None
    assert "something the server never said" in api.upstream.sent[1].text


def test_a_request_with_nothing_new_to_send_is_refused(api):
    api.post("/v1/chat/completions", json=CHAT)
    replayed = [*CHAT["messages"], {"role": "assistant", "content": "ok"}]

    refused = api.post("/v1/chat/completions", json={**CHAT, "messages": replayed})

    assert refused.status_code == 400
    assert len(api.upstream.sent) == 1


@pytest.mark.parametrize(
    ("field", "value"), [("previous_response_id", "resp_1"), ("conversation", {"id": "conv_1"})]
)
def test_responses_refuses_a_request_that_leaves_its_history_to_the_server(api, field, value):
    request = {"model": MODEL, "input": "And then?"}

    refused = api.post("/v1/responses", json={**request, field: value})

    assert refused.status_code == 400
    assert field in refused.json()["error"]["message"]
    assert api.upstream.sent == []
    # null is how a client says there is none
    assert api.post("/v1/responses", json={**request, field: None}).status_code == 200


def _events(response) -> list[dict]:
    """The events of a Responses stream, checked for framing as they are read."""
    assert response.status_code == 200, response.text
    *frames, done = response.text.strip().split("\n\n")
    assert done == "data: [DONE]"
    events = []
    for frame in frames:
        name, data = frame.split("\n")
        events.append(json.loads(data.removeprefix("data: ")))
        assert name == f"event: {events[-1]['type']}"
    return events


@pytest.mark.parametrize(
    ("answer", "item_types"),
    [
        (reply("Hel", "lo."), ["message"]),
        (reply("Answer.", thoughts="Hm.", thoughts_delta="Hm."), ["reasoning", "message"]),
        (reply("Checking. " + TOOL_CALL), ["message", "function_call"]),
        (reply(TOOL_CALL), ["function_call"]),
        (reply("\n", TOOL_CALL), ["function_call"]),
        (reply(), ["message"]),
        (reply(**IMAGE), ["message", "image_generation_call"]),
        # Thoughts that arrive only with the final chunk, after the text has started.
        (reply("Answer.", thoughts="Hm."), ["message", "reasoning"]),
        (reply("Answer.", thoughts="Hm, more.", thoughts_delta="Hm"), ["reasoning", "message"]),
    ],
    ids=[
        "text",
        "reasoning",
        "tool-call",
        "only-a-tool-call",
        "a-tool-call-after-a-newline",
        "empty",
        "image",
        "late-thought",
        "more-thought-at-the-end",
    ],
)
def test_a_responses_stream_can_be_reassembled_into_its_final_response(api, answer, item_types):
    """What an SDK does with the stream: follow the items as they open and close, add up the
    text deltas, and end up with exactly the response the last event carries. Without
    streaming, the response has the same items."""
    api.upstream.replies += [answer, answer]
    request = {"model": MODEL, "input": "Hi", "tools": [WEATHER_TOOL]}
    events = _events(api.post("/v1/responses", json={**request, "stream": True}))
    unstreamed = api.post("/v1/responses", json=request).json()["output"]

    assert [e["type"] for e in events[:2]] == ["response.created", "response.in_progress"]
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    final = events[-1]
    assert final["type"] == "response.completed"

    opened = [e for e in events if e["type"] == "response.output_item.added"]
    closed = [e for e in events if e["type"] == "response.output_item.done"]
    assert [e["output_index"] for e in opened] == list(range(len(item_types)))
    assert [e["item"]["type"] for e in opened] == item_types
    for index in range(len(item_types)):
        about_item = [events.index(e) for e in events if e.get("output_index") == index]
        assert events[about_item[0]] is opened[index]  # nothing refers to an item before it opens
        assert events[about_item[-1]]["type"] == "response.output_item.done"
    closed.sort(key=lambda e: e["output_index"])
    assert [e["item"] for e in closed] == final["response"]["output"]

    streamed_text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    texts = [
        item["content"][0]["text"] for item in final["response"]["output"] if "content" in item
    ]
    # Text before a tool call keeps its trailing space in the deltas and loses it in the final.
    assert texts == ([streamed_text.rstrip()] if "message" in item_types else [])
    assert sorted(item["type"] for item in unstreamed) == sorted(item_types)


@pytest.mark.parametrize("stream", [False, True])
def test_gemini_native_ends_a_reply_with_a_finish_reason_and_its_usage(api, stream):
    api.upstream.replies.append(reply("Hello ", "there.", thoughts="Hm.", thoughts_delta="Hm."))
    method = "streamGenerateContent" if stream else "generateContent"
    response = api.post(f"/v1beta/models/{MODEL}:{method}", json={"contents": USER_TURN})
    chunks = _sse(response) if stream else [response.json()]

    assert [c["candidates"][0].get("finishReason") for c in chunks] == [None] * (
        len(chunks) - 1
    ) + ["STOP"]
    usage = chunks[-1]["usageMetadata"]
    assert usage["candidatesTokenCount"] > 0
    assert usage["thoughtsTokenCount"] > 0
    assert usage["totalTokenCount"] == (
        usage["promptTokenCount"] + usage["candidatesTokenCount"] + usage["thoughtsTokenCount"]
    )
    assert chunks[-1]["modelVersion"] == MODEL


def test_a_chat_completions_stream_ends_with_its_usage_beside_the_choices(api):
    """Where OpenAI's SDKs read it: `ChatCompletionChunk.usage`, not inside a choice."""
    api.upstream.replies.append(reply("Hello there, how can I help?"))
    chunks = _sse(
        api.post(
            "/v1/chat/completions",
            json={"model": MODEL, "stream": True, "messages": [{"role": "user", "content": "Hi"}]},
        )
    )

    assert chunks[-1]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert chunks[-1]["usage"]["completion_tokens"] > 0


@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions", "/v1/responses", f"/v1beta/models/{MODEL}:streamGenerateContent"],
)
def test_a_file_that_every_chunk_of_a_stream_lists_is_downloaded_once(api, path):
    """Gemini repeats the files of a reply in each chunk that follows their first mention."""
    chunks = reply("Here ", "it ", "is.")
    for chunk in chunks:
        chunk.candidates[0].web_images = [WebImage(url="http://img/a", title="A")]
    api.upstream.replies.append(chunks)
    prompt = {"messages": CHAT["messages"]} if "chat" in path else {"input": "Draw"}
    body = (
        {"contents": USER_TURN} if "v1beta" in path else {"model": MODEL, "stream": True, **prompt}
    )

    assert api.post(path, json=body).status_code == 200
    assert api.upstream.downloaded == ["http://img/a"]


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_a_developer_message_is_a_system_message(api, path):
    """OpenAI renamed the role; Gemini is given the instructions the same way under either."""
    for role in ("system", "developer"):
        turns = [
            {"role": role, "content": "Be brief"},
            {"role": "user", "content": "First question"},
        ]
        key = "messages" if "chat" in path else "input"
        assert api.post(path, json={"model": MODEL, key: turns}).status_code == 200

    as_system, as_developer = api.upstream.sent
    assert "Be brief" in as_system.text
    assert as_developer.text == as_system.text
