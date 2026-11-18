"""The three API surfaces end to end: real routes, conversion, storage and serialisation, with only
Gemini itself replaced by the script in `conftest.api`."""

from conftest import MODEL, reply

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
