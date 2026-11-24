"""Wire-format compatibility tests for the OpenAI- and Gemini-shaped surfaces.

The contract these lock down: a request naming only standard attributes must never be rejected
just because Gemini Web cannot honour one of them. Options that cannot be forwarded are accepted
and dropped; only malformed or unrepresentable *content* is refused.
"""

import asyncio
import base64
import time

import orjson
import pytest
from pydantic import ValidationError

from app.models.core import AppContentItem, AppMessage, AppToolCall, AppToolCallFunction
from app.models.gemini_models import GeminiGenerateContentRequest
from app.models.models import (
    ChatCompletionNamedToolChoice,
    FunctionCallOutput,
    FunctionTool,
    ResponseCreateRequest,
    ResponseFunctionToolCall,
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputMessage,
    StructuredOutputRequirement,
    ToolChoiceFunction,
    ToolChoiceTypes,
)
from app.server.chat import (
    _build_structured_requirement,
    _convert_responses_to_app_messages,
    _responses_response_format,
    _tool_choice_declaration_error,
    _tool_choice_failure,
)
from app.server.gemini import (
    _gemini_structured_requirement,
    _gemini_tools_to_internal,
    _validate_gemini_request,
)
from app.server.media import _media_markdown
from app.services.client import process_conversation
from app.utils.helper import (
    SCHEMA_ADHERENCE_PROMPT,
    STRICT_SCHEMA_ADHERENCE_PROMPT,
    StructuredOutputValidationError,
    canonicalize_structured_output,
    decode_base64_data,
    extract_tool_calls,
    guess_extension_for_mime,
    normalize_openapi_schema,
    process_llm_output,
    strip_markdown_fence,
)

TOOL_CALL_OUTPUT = (
    "[ToolCalls][Call:get_weather][CallParameter:city]Hanoi[/CallParameter][/Call][/ToolCalls]"
)
WEATHER_TOOLS = [FunctionTool(type="function", name="get_weather")]
OBJECT_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


def _requirement(schema: dict, *, strict: bool = True) -> StructuredOutputRequirement:
    return StructuredOutputRequirement(
        schema_name="r", schema=schema, instruction="", strict=strict
    )


def _gemini_request(**overrides) -> GeminiGenerateContentRequest:
    payload = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], **overrides}
    return GeminiGenerateContentRequest.model_validate(payload)


@pytest.mark.parametrize("format_type", ["text", "json_object"])
def test_non_json_schema_response_formats_are_accepted(format_type):
    """`text` is the API default and `json_object` is JSON mode; neither may 400."""
    requirement = _build_structured_requirement({"type": format_type})
    if format_type == "text":
        assert requirement is None
    else:
        assert requirement is not None
        # JSON mode promises valid JSON only, so it must not be enforced as strict.
        assert requirement.strict is False


@pytest.mark.parametrize(("requested", "applied"), [(True, True), (False, False), (None, False)])
def test_json_schema_sets_strict_from_the_request(requested, applied):
    """A null `strict` is the default, as it is for OpenAI and on the Responses route."""
    requirement = _build_structured_requirement(
        {"type": "json_schema", "json_schema": {"schema": OBJECT_SCHEMA, "strict": requested}}
    )
    assert requirement is not None
    assert requirement.strict is applied


@pytest.mark.parametrize("strict", ["true", 1])
def test_chat_json_schema_rejects_non_boolean_strict_values(strict):
    with pytest.raises(ValueError, match="strict must be a boolean"):
        _build_structured_requirement(
            {"type": "json_schema", "json_schema": {"schema": OBJECT_SCHEMA, "strict": strict}}
        )


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {}},
    ],
)
def test_malformed_json_schema_is_still_rejected(response_format):
    with pytest.raises(ValueError, match="schema"):
        _build_structured_requirement(response_format)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "not-a-type"},
        {"type": "integer", "exclusiveMinimum": True},  # draft-4 spelling
    ],
)
def test_an_unrepresentable_schema_is_asked_for_but_not_enforced(schema):
    """A schema we cannot evaluate must not 400: that loses an answer over a gap on our side."""
    requirement = _build_structured_requirement(
        {"type": "json_schema", "json_schema": {"schema": schema, "strict": True}}
    )
    assert requirement is not None
    # Still shown to the model, but it cannot be used to judge the reply.
    assert orjson.dumps(schema, option=orjson.OPT_SORT_KEYS).decode() in requirement.instruction
    assert requirement.schema == {}
    assert requirement.strict is False


@pytest.mark.parametrize("response_format", [None, {}])
def test_an_absent_response_format_asks_for_nothing(response_format):
    assert _build_structured_requirement(response_format) is None


def test_a_json_schema_without_a_name_is_named_and_asked_for():
    requirement = _build_structured_requirement(
        {"type": "json_schema", "json_schema": {"schema": OBJECT_SCHEMA, "name": ""}}
    )
    assert requirement is not None
    assert requirement.schema_name == "response"
    assert SCHEMA_ADHERENCE_PROMPT in requirement.instruction


@pytest.mark.parametrize("strict", [True, False])
def test_only_a_strict_schema_is_prompted_for_exact_conformance(strict):
    requirement = _build_structured_requirement(
        {"type": "json_schema", "json_schema": {"schema": OBJECT_SCHEMA, "strict": strict}}
    )
    assert requirement is not None
    assert (STRICT_SCHEMA_ADHERENCE_PROMPT in requirement.instruction) is strict


def test_tool_call_turn_is_not_failed_by_a_response_format():
    """The schema constrains the final answer, not a turn that asks for a tool."""
    raw = "Let me check. " + TOOL_CALL_OUTPUT
    _, visible, tool_calls = process_llm_output(
        None, raw, _requirement(OBJECT_SCHEMA), WEATHER_TOOLS
    )
    assert [call.function.name for call in tool_calls] == ["get_weather"]
    assert visible == "Let me check."


def test_strict_violation_raises_and_best_effort_violation_degrades():
    with pytest.raises(StructuredOutputValidationError):
        process_llm_output(None, '{"b": 1}', _requirement(OBJECT_SCHEMA))

    _, visible, _ = process_llm_output(None, '{"b": 1}', _requirement(OBJECT_SCHEMA, strict=False))
    assert visible == '{"b": 1}'


@pytest.mark.parametrize("raw_text", ["", "   \n  "])
def test_a_reply_with_no_text_is_not_a_schema_violation(raw_text):
    """An image-only or empty turn has nothing to validate; failing it would invent an error."""
    _, visible, _ = process_llm_output(None, raw_text, _requirement(OBJECT_SCHEMA))
    assert visible == ""


def test_text_alongside_a_tool_call_is_still_canonicalized():
    raw = f'```json\n{{"a": "x"}}\n```\n{TOOL_CALL_OUTPUT}'
    _, visible, tool_calls = process_llm_output(
        None, raw, _requirement(OBJECT_SCHEMA), WEATHER_TOOLS
    )
    assert [call.function.name for call in tool_calls] == ["get_weather"]
    assert visible == '{"a":"x"}'


def test_conforming_output_is_canonicalized():
    _, visible, _ = process_llm_output(
        None, '```json\n{"a": "x"}\n```', _requirement(OBJECT_SCHEMA)
    )
    assert visible == '{"a":"x"}'


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#/definitions/missing"},  # dangling local reference
        {"type": "OBJECT"},  # foreign dialect that slipped through
    ],
)
def test_unusable_schemas_leave_the_reply_unverified_rather_than_failing_it(schema):
    """Failing to run the check is our problem, not the model's, so it cannot be a violation."""
    assert canonicalize_structured_output('{"a": 1}', _requirement(schema)) == '{"a":1}'
    # And it must not reach the caller as an error, even under strict.
    _, visible, _ = process_llm_output(None, '{"a": 1}', _requirement(schema))
    assert visible == '{"a":1}'


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "http://169.254.169.254/latest/meta-data"},
        {"type": "object", "properties": {"a": {"$ref": "http://internal.invalid/schema.json"}}},
        {"$schema": "http://internal.invalid/meta.json", "$ref": "http://internal.invalid/s.json"},
    ],
)
@pytest.mark.filterwarnings("ignore:The metaschema specified by:DeprecationWarning")
def test_a_client_schema_never_makes_the_server_fetch_a_url(schema, monkeypatch):
    """The schema is client input, so resolving a remote reference in it would be SSRF."""
    fetched: list[str] = []

    def record_fetch(request, *args, **kwargs):
        fetched.append(getattr(request, "full_url", str(request)))
        raise OSError("network access is not expected here")

    monkeypatch.setattr("urllib.request.urlopen", record_fetch)

    assert canonicalize_structured_output('{"a": 1}', _requirement(schema)) == '{"a":1}'
    assert fetched == []


def test_local_references_are_still_resolved_and_enforced():
    schema = {
        "$defs": {"count": {"type": "integer"}},
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/count"}},
    }
    assert canonicalize_structured_output('{"a": 1}', _requirement(schema)) == '{"a":1}'
    assert canonicalize_structured_output('{"a": "x"}', _requirement(schema)) is None


def test_json_mode_requires_only_that_the_payload_parses():
    requirement = _requirement({}, strict=False)
    assert canonicalize_structured_output('{"anything": [1]}', requirement) == '{"anything":[1]}'
    assert canonicalize_structured_output("not json", requirement) is None


def test_pathological_schema_regex_is_bounded(monkeypatch):
    """A client-controlled pattern must not monopolize the async server thread.

    This one takes exponential time on the input below: unbounded, it would not return.
    """
    monkeypatch.setattr("app.utils.helper.SCHEMA_REGEX_BUDGET_SECONDS", 0.005)
    requirement = _requirement(
        {
            "type": "object",
            "properties": {"value": {"type": "string", "pattern": "^(a|aa)+$"}},
            "required": ["value"],
        }
    )
    reply = '{"value":"' + "a" * 100 + 'b"}'
    started = time.perf_counter()
    result = canonicalize_structured_output(reply, requirement)
    assert time.perf_counter() - started < 0.5
    assert result == reply  # passed on unverified, as any reply we ran out of time for


def test_an_exhausted_budget_leaves_the_reply_unverified_rather_than_failing_it(monkeypatch):
    """Running out of time is our limit, not a schema violation, so strict must not 502."""
    monkeypatch.setattr("app.utils.helper.SCHEMA_REGEX_BUDGET_SECONDS", 1e-9)
    requirement = _requirement({"type": "object", "properties": {"a": {"pattern": "^x$"}}})
    assert canonicalize_structured_output('{"a": "x"}', requirement) == '{"a":"x"}'


def test_an_ordinary_large_payload_fits_the_regex_budget():
    """The budget is cumulative, so it has to cover realistic volume, not just one pattern."""
    requirement = _requirement(
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "pattern": r"^[^@]+@[^@]+\.[A-Za-z]{2,}$"},
                    "sku": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{6}$"},
                    "slug": {"type": "string", "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$"},
                },
                "required": ["email", "sku", "slug"],
            },
        }
    )
    rows = [
        {"email": f"u{i}@example.com", "sku": f"ABC-{i:06d}", "slug": f"row-{i}"}
        for i in range(2000)
    ]
    assert canonicalize_structured_output(orjson.dumps(rows).decode(), requirement) is not None
    # A violation in the last row is still caught, so the budget lasted the whole payload: a
    # reply it ran out on would have been passed on unverified instead.
    rows[-1]["sku"] = "not a sku"
    assert canonicalize_structured_output(orjson.dumps(rows).decode(), requirement) is None


def test_bounded_validator_preserves_pattern_properties_and_additional_properties():
    requirement = _requirement(
        {
            "type": "object",
            "patternProperties": {"^item_[0-9]+$": {"type": "integer"}},
            "additionalProperties": False,
        }
    )
    assert canonicalize_structured_output('{"item_1": 1}', requirement) == '{"item_1":1}'
    assert canonicalize_structured_output('{"other": 1}', requirement) is None


@pytest.mark.parametrize(
    ("format_payload", "expected_type"),
    [
        ({"type": "text"}, None),
        ({"type": "json_object"}, "json_object"),
        ({"type": "json_schema", "name": "r", "schema": OBJECT_SCHEMA}, "json_schema"),
    ],
)
def test_text_format_accepts_every_standard_variant(format_payload, expected_type):
    request = ResponseCreateRequest.model_validate(
        {"model": "m", "input": "hi", "text": {"format": format_payload}}
    )
    resolved = _responses_response_format(request)
    assert (resolved or {}).get("type") == expected_type


def test_default_text_block_does_not_conflict_with_the_legacy_extension():
    request = ResponseCreateRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "text": {"format": {"type": "text"}},
            "response_format": {"type": "json_object"},
        }
    )
    assert _responses_response_format(request) == {"type": "json_object"}


def test_neither_format_block_yields_no_requirement():
    assert (
        _responses_response_format(
            ResponseCreateRequest.model_validate({"model": "m", "input": "hi"})
        )
        is None
    )


def test_legacy_response_format_alone_is_honored():
    request = ResponseCreateRequest.model_validate(
        {"model": "m", "input": "hi", "response_format": {"type": "json_object"}}
    )
    assert _responses_response_format(request) == {"type": "json_object"}


def test_text_format_json_schema_defaults_name_and_strict():
    request = ResponseCreateRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "text": {"format": {"type": "json_schema", "schema": OBJECT_SCHEMA}},
        }
    )
    resolved = _responses_response_format(request) or {}
    assert resolved["json_schema"]["name"] == "response"
    # Same default as Chat Completions and as OpenAI: an omitted `strict` is best-effort.
    assert resolved["json_schema"]["strict"] is False


@pytest.mark.parametrize("strict", ["true", 1])
def test_responses_json_schema_rejects_non_boolean_strict_values(strict):
    with pytest.raises(ValidationError):
        ResponseCreateRequest.model_validate(
            {
                "model": "m",
                "input": "hi",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "schema": OBJECT_SCHEMA,
                        "strict": strict,
                    }
                },
            }
        )


def test_text_format_json_schema_without_a_schema_is_rejected():
    request = ResponseCreateRequest.model_validate(
        {"model": "m", "input": "hi", "text": {"format": {"type": "json_schema", "name": "r"}}}
    )
    with pytest.raises(ValueError, match="schema is required"):
        _responses_response_format(request)


def test_two_conflicting_formats_are_rejected():
    request = ResponseCreateRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "text": {"format": {"type": "json_schema", "name": "r", "schema": OBJECT_SCHEMA}},
            "response_format": {"type": "json_object"},
        }
    )
    with pytest.raises(ValueError, match="not both"):
        _responses_response_format(request)


def test_openapi_response_schema_is_translated_not_rejected():
    """`responseSchema` is the OpenAPI subset: uppercase types and `nullable`."""
    request = _gemini_request(
        generationConfig={
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "age": {"type": "INTEGER", "nullable": True},
                },
                "required": ["name"],
                "propertyOrdering": ["name", "age"],
            },
        }
    )
    assert _validate_gemini_request(request) is None
    requirement = _gemini_structured_requirement(request.generationConfig)
    assert requirement is not None
    assert requirement.schema == {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": ["integer", "null"]}},
        "required": ["name"],
    }


def test_invalid_response_json_schema_is_rejected():
    """`responseJsonSchema` really is JSON Schema, so it can be judged as such."""
    request = _gemini_request(
        generationConfig={
            "responseMimeType": "application/json",
            "responseJsonSchema": {"type": "not-a-type"},
        }
    )
    assert "Invalid JSON Schema" in (_validate_gemini_request(request) or "")


def test_the_gemini_surface_asks_for_the_schema_without_enforcing_it():
    """Google guarantees conformance by constrained decoding; this wrapper can only ask.

    The native surface has no `strict` flag for a caller to turn off, so failing the request on
    a near-miss would throw away an answer with no way to opt out of that.
    """
    request = _gemini_request(
        generationConfig={"responseMimeType": "application/json", "responseSchema": OBJECT_SCHEMA}
    )
    requirement = _gemini_structured_requirement(request.generationConfig)

    assert requirement is not None
    assert requirement.schema == OBJECT_SCHEMA
    assert requirement.strict is False
    assert STRICT_SCHEMA_ADHERENCE_PROMPT not in requirement.instruction

    # A violation therefore comes back as text rather than costing the caller the reply.
    _, visible, _ = process_llm_output(None, '{"b": 1}', requirement)
    assert visible == '{"b": 1}'


@pytest.mark.parametrize(
    "generation_config",
    [
        {"responseMimeType": "application/json"},
        {"responseMimeType": "application/json", "responseJsonSchema": {}},
        {"responseMimeType": "application/json", "responseSchema": {}},
    ],
)
def test_gemini_json_mode_survives_absent_or_empty_schemas(generation_config):
    requirement = _gemini_structured_requirement(
        _gemini_request(generationConfig=generation_config).generationConfig
    )
    assert requirement is not None
    assert requirement.schema == {}
    assert requirement.strict is False
    assert canonicalize_structured_output('{"valid": true}', requirement) == '{"valid":true}'


@pytest.mark.parametrize(
    "generation_config",
    [
        None,
        {"responseSchema": OBJECT_SCHEMA},
        {"responseMimeType": "text/plain", "responseJsonSchema": OBJECT_SCHEMA},
    ],
)
def test_a_native_schema_is_asked_for_only_with_the_json_mime_type(generation_config):
    request = _gemini_request(generationConfig=generation_config)
    assert _gemini_structured_requirement(request.generationConfig) is None


def test_normalize_openapi_schema_leaves_json_schema_alone():
    assert normalize_openapi_schema(OBJECT_SCHEMA) == OBJECT_SCHEMA


def test_normalize_openapi_schema_recurses_through_containers():
    assert normalize_openapi_schema(
        {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"x": {"anyOf": [{"type": "STRING"}, {"type": "INTEGER"}]}},
            },
        }
    ) == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        },
    }


def test_normalize_openapi_schema_treats_property_names_as_names():
    """A property called `type` or `nullable` must not be mistaken for the keyword."""
    assert normalize_openapi_schema(
        {
            "type": "OBJECT",
            "properties": {"type": {"type": "STRING"}, "nullable": {"type": "BOOLEAN"}},
        }
    ) == {
        "type": "object",
        "properties": {"type": {"type": "string"}, "nullable": {"type": "boolean"}},
    }


def test_normalize_openapi_schema_reads_keywords_under_their_proto_names():
    """Google's Python SDK writes some keywords that way, such as `max_items`."""
    assert normalize_openapi_schema(
        {
            "type": "OBJECT",
            "properties": {
                "tags": {"type": "ARRAY", "items": {"type": "STRING"}, "max_items": 3},
                "first_name": {"any_of": [{"$ref": "#/$defs/name"}]},
            },
            "property_ordering": ["tags", "first_name"],
            "$defs": {"name": {"type": "STRING"}},
        }
    ) == {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "first_name": {"anyOf": [{"$ref": "#/$defs/name"}]},
        },
        "$defs": {"name": {"type": "string"}},
    }


def test_normalize_openapi_schema_drops_nullable_false_without_widening():
    assert normalize_openapi_schema({"type": "STRING", "nullable": False}) == {"type": "string"}


def test_normalize_openapi_schema_passes_a_boolean_schema_through():
    """`additionalProperties: true` puts a boolean where a schema would be."""
    assert normalize_openapi_schema(True) is True


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"contents": []}, "contents is required"),
        (
            {"contents": [{"role": "user", "parts": []}]},
            "must contain at least one part",
        ),
        (
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"fileData": {"mimeType": "text/plain", "fileUri": "gs://x"}}],
                    }
                ]
            },
            "fileData is not supported",
        ),
        ({"cachedContent": "cachedContents/abc"}, "cachedContent is not supported"),
    ],
)
def test_unrepresentable_content_is_refused(overrides, expected):
    """Content the wrapper cannot resolve is refused; dropping it would change the question."""
    payload = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], **overrides}
    request = GeminiGenerateContentRequest.model_validate(payload)
    assert expected in (_validate_gemini_request(request) or "")


def test_file_data_in_system_instruction_is_refused():
    request = _gemini_request(
        systemInstruction={"parts": [{"fileData": {"mimeType": "text/plain", "fileUri": "gs://x"}}]}
    )
    assert "fileData is not supported" in (_validate_gemini_request(request) or "")


_TOOLS = [
    {"functionDeclarations": [{"name": "a", "description": "d"}, {"name": "b", "description": "d"}]}
]


@pytest.mark.parametrize(
    ("mode", "expected_tools", "expected_choice"),
    [
        ("AUTO", ["a", "b"], "auto"),
        ("NONE", ["a", "b"], "none"),
        ("VALIDATED", ["a"], "auto"),
    ],
)
def test_allowed_function_names_only_narrows_the_modes_that_use_it(
    mode, expected_tools, expected_choice
):
    request = _gemini_request(
        tools=_TOOLS,
        toolConfig={"functionCallingConfig": {"mode": mode, "allowedFunctionNames": ["a"]}},
    )
    tools, choice = _gemini_tools_to_internal(request.tools, request.toolConfig)
    assert [tool.name for tool in tools or []] == expected_tools
    assert choice == expected_choice
    # Names it does not act on cannot be a validation error either.
    assert _validate_gemini_request(request) is None


def test_any_mode_with_one_allowed_name_forces_that_function():
    request = _gemini_request(
        tools=_TOOLS,
        toolConfig={"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["a"]}},
    )
    tools, choice = _gemini_tools_to_internal(request.tools, request.toolConfig)
    assert [tool.name for tool in tools or []] == ["a"]
    assert getattr(choice, "name", None) == "a"


def test_any_mode_rejects_undeclared_allowed_names():
    request = _gemini_request(
        tools=_TOOLS,
        toolConfig={"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["zzz"]}},
    )
    assert "undeclared functions" in (_validate_gemini_request(request) or "")


@pytest.mark.parametrize(
    "tool_config",
    [
        None,
        {"functionCallingConfig": {"mode": "ANY"}},
        {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["a", "b"]}},
    ],
    ids=["no-config", "any-unrestricted", "any-multiple-names"],
)
def test_any_mode_without_a_single_target_forces_only_that_a_tool_is_called(tool_config):
    request = _gemini_request(tools=_TOOLS, toolConfig=tool_config)
    tools, choice = _gemini_tools_to_internal(request.tools, request.toolConfig)
    assert [tool.name for tool in tools or []] == ["a", "b"]
    assert choice == (None if tool_config is None else "required")


# `parameters_json_schema` is what Google's Python SDK sends for a tool given as a Python function.
@pytest.mark.parametrize("field", ["parameters", "parameters_json_schema"])
def test_a_function_keeps_its_parameters_whichever_field_declares_them(field):
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    request = _gemini_request(tools=[{"functionDeclarations": [{"name": "a", field: schema}]}])
    tools, _ = _gemini_tools_to_internal(request.tools, None)
    assert [tool.parameters for tool in tools or []] == [schema]


@pytest.mark.parametrize(
    ("mode", "strict"), [("AUTO", False), ("NONE", False), ("ANY", True), ("VALIDATED", True)]
)
def test_the_modes_that_promise_schema_adherence_make_every_function_strict(mode, strict):
    request = _gemini_request(tools=_TOOLS, toolConfig={"functionCallingConfig": {"mode": mode}})
    tools, _ = _gemini_tools_to_internal(request.tools, request.toolConfig)
    assert [tool.strict for tool in tools or []] == [strict, strict]


def test_no_tools_yields_no_tool_choice():
    assert _gemini_tools_to_internal(None, None) == (None, None)


_CALL = AppToolCall(id="1", type="function", function=AppToolCallFunction(name="a", arguments="{}"))
_NAMED = ChatCompletionNamedToolChoice.model_validate(
    {"type": "function", "function": {"name": "a"}}
)
_FUNCTION = ToolChoiceFunction(type="function", name="a")
_IMAGE = ToolChoiceTypes(type="image_generation")


@pytest.mark.parametrize(
    ("tool_choice", "tool_calls", "has_images", "has_image_tool", "expected"),
    [
        (None, [], False, False, None),
        ("auto", [], False, False, None),
        ("none", [], False, False, None),
        ("required", [_CALL], False, False, None),
        ("required", [], False, False, "required tool result"),
        # An image satisfies `required` only when an image tool was declared; one Gemini
        # volunteers on its own cannot stand in for the function call that was forced.
        ("required", [], True, True, None),
        ("required", [], True, False, "required tool result"),
        (_NAMED, [_CALL], False, False, None),
        (_NAMED, [], False, False, "required function 'a'"),
        (_FUNCTION, [_CALL], False, False, None),
        (_FUNCTION, [], False, False, "required function 'a'"),
        (_IMAGE, [], True, True, None),
        (_IMAGE, [], False, True, "image generation result"),
    ],
)
def test_forced_tool_choice_failure_detection(
    tool_choice, tool_calls, has_images, has_image_tool, expected
):
    result = _tool_choice_failure(
        tool_choice, tool_calls, has_images=has_images, has_image_tool=has_image_tool
    )
    if expected is None:
        assert result is None
    else:
        assert expected in (result or "")


@pytest.mark.parametrize(
    ("names", "has_image_tool", "tool_choice", "expected"),
    [
        (set(), False, "auto", None),
        ({"a"}, False, "required", None),
        (set(), True, "required", None),
        (set(), False, "required", "requires at least one tool"),
        ({"a"}, False, _NAMED, None),
        ({"b"}, False, _NAMED, "undeclared function 'a'"),
        ({"b"}, False, _FUNCTION, "undeclared function 'a'"),
        (set(), True, _IMAGE, None),
        (set(), False, _IMAGE, "requires an image_generation tool"),
    ],
)
def test_forced_tool_choice_must_name_a_declared_tool(names, has_image_tool, tool_choice, expected):
    result = _tool_choice_declaration_error(names, has_image_tool, tool_choice)
    if expected is None:
        assert result is None
    else:
        assert expected in (result or "")


def _input_message(*parts) -> ResponseInputMessage:
    return ResponseInputMessage.model_validate({"role": "user", "content": list(parts)})


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ("a plain string prompt", None),
        ([_input_message({"type": "input_text", "text": "hi"})], None),
        ([_input_message({"type": "input_image", "image_url": "https://x/y.png"})], None),
        ([_input_message({"type": "input_file", "file_url": "https://x/a.pdf"})], None),
        ([_input_message({"type": "input_file", "file_data": "aGk="})], None),
        ([_input_message({"type": "input_file", "file_id": "file-1"})], "file_id inputs"),
        ([_input_message({"type": "input_image"})], "input_image must contain image_url"),
        (
            [
                _input_message(
                    {"type": "input_file", "file_url": "https://x/a.pdf", "file_data": "aGk="}
                )
            ],
            "exactly one of file_url or file_data",
        ),
        (
            [_input_message({"type": "input_file", "filename": "a.pdf"})],
            "exactly one of file_url or file_data",
        ),
    ],
)
def test_responses_input_refuses_only_unusable_content(items, expected):
    if expected is None:
        assert _convert_responses_to_app_messages(items)
    else:
        with pytest.raises(ValueError, match=expected):
            _convert_responses_to_app_messages(items)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("done", None),
        ([{"type": "input_text", "text": "done"}], None),
        ([{"type": "input_file", "file_id": "file-1"}], "file_id inputs"),
    ],
)
def test_tool_result_parts_are_validated_too(output, expected):
    items: list = [FunctionCallOutput.model_validate({"call_id": "c", "output": output})]
    if expected is None:
        assert _convert_responses_to_app_messages(items)
    else:
        with pytest.raises(ValueError, match=expected):
            _convert_responses_to_app_messages(items)


def test_responses_input_sent_without_types_is_read_by_its_fields():
    """Only an item with a `role` has to be a message; the rest are read as they always were."""
    untyped_parts = [{"image_url": "data:image/png;base64,AA=="}, {"file_url": "https://x/a"}]
    items = [
        {"call_id": "c", "name": "f", "arguments": "{}"},
        {"call_id": "c", "output": "Sunny."},
        {"role": "user", "content": untyped_parts},
    ]
    request = ResponseCreateRequest.model_validate({"model": "m", "input": items})

    assert isinstance(request.input, list)
    call, output, message = request.input
    assert isinstance(call, ResponseFunctionToolCall)
    assert isinstance(output, FunctionCallOutput)
    assert isinstance(message, ResponseInputMessage)
    assert [type(part) for part in message.content] == [ResponseInputImage, ResponseInputFile]


def test_non_ascii_data_url_does_not_abort_model_construction():
    item = AppContentItem(type="image_url", url="data:text/plain;charset=utf-8,Hé")
    assert item.content_digest


def test_digest_distinguishes_inline_payloads():
    first = base64.b64encode(b"one").decode()
    second = base64.b64encode(b"two").decode()
    assert (
        AppContentItem(type="file", file_data=first, filename="a.bin").content_digest
        != AppContentItem(type="file", file_data=second, filename="a.bin").content_digest
    )


@pytest.mark.parametrize(
    ("mime_type", "expected"),
    [
        ("image/jpeg", ".jpg"),
        ("application/pdf", ".pdf"),
        # Unregistered but widely sent: the subtype is the fallback, not ".bin".
        ("audio/mp3", ".mp3"),
        ("application/x-foo", ".x-foo"),
        ("image/png; charset=binary", ".png"),
        (None, ".bin"),
        ("nosubtype", ".bin"),
        # A separator here would place the temp file outside its directory.
        ("image/../../etc/passwd", ".etcpasswd"),
    ],
)
def test_mime_extensions_fall_back_to_a_scrubbed_subtype(mime_type, expected):
    suffix = guess_extension_for_mime(mime_type)
    assert suffix == expected
    assert not set(suffix) & set("/\\")


def test_both_base64_alphabets_decode():
    """Clients built on `urlsafe_b64encode` send `-` and `_`."""
    payload = bytes(range(256))
    assert decode_base64_data(base64.b64encode(payload).decode()) == payload
    assert decode_base64_data(base64.urlsafe_b64encode(payload).decode()) == payload
    with pytest.raises(ValueError, match="Base64"):
        decode_base64_data("not base64 at all!!")


def test_base64_accepts_bytes_line_wrapping_and_data_urls():
    payload = b"\x00\x01binary payload\xff"
    encoded = base64.b64encode(payload).decode()
    assert decode_base64_data(encoded.encode()) == payload
    # MIME-style encoders wrap long payloads across lines.
    assert decode_base64_data(f"{encoded[:4]}\n{encoded[4:]}") == payload
    assert decode_base64_data(f"data:application/octet-stream;base64,{encoded}") == payload


def test_data_url_without_a_base64_payload_is_rejected():
    with pytest.raises(ValueError, match="Data URL"):
        decode_base64_data("data:text/plain,hello")


def test_non_ascii_is_an_error_rather_than_something_to_strip():
    """Dropping a stray character could make a corrupt payload decode to the wrong bytes."""
    payload = base64.b64encode(b"body").decode()
    with pytest.raises(ValueError, match="non-ASCII"):
        decode_base64_data(f"{payload[:2]}é{payload[2:]}")


def test_digest_is_identical_across_equivalent_representations():
    """The same bytes must hash alike whether sent as a data URL or as inline file data."""
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nbody").decode()
    as_data_url = AppContentItem(type="image_url", url=f"data:image/png;base64,{encoded}")
    as_file = AppContentItem(type="file", file_data=encoded, filename="a.png")
    assert as_data_url.content_digest == as_file.content_digest


def test_only_payloads_excluded_from_serialization_get_a_digest():
    """Everything else survives the round trip and is compared directly."""
    assert AppContentItem(type="text", text="hi").content_digest is None
    assert AppContentItem(type="image_url", url="https://example.com/a.png").content_digest is None
    assert AppContentItem(type="x", raw_data={"a": 1}).content_digest is None


def test_digest_survives_a_round_trip_without_the_excluded_bytes():
    original = AppContentItem(
        type="file", file_data=base64.b64encode(b"payload").decode(), filename="a.bin"
    )
    restored = AppContentItem.model_validate(original.model_dump())
    assert restored.file_data is None
    assert restored.content_digest == original.content_digest


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ('```json\n{"city": "Hanoi"}\n```', '{"city": "Hanoi"}'),
        ("```python\nprint('hello')\n```", "print('hello')"),
        ('```json\n{"city": "Hanoi"}```', '{"city": "Hanoi"}'),
        ('```json {"city": "Hanoi"}```', '{"city": "Hanoi"}'),
        ("```yaml key: value```", "key: value"),
        ("```\nHanoi\n```", "Hanoi"),
        ("````typescript\nconst x = `inner`;\n````", "const x = `inner`;"),
        ("plain text without fence", "plain text without fence"),
        ("```", "```"),
        ("", ""),
    ],
)
def test_strip_markdown_fence_with_arbitrary_language_tags(payload, expected):
    assert strip_markdown_fence(payload) == expected


_OPTIONAL_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}  # pydantic's `str | None`


@pytest.mark.parametrize(
    ("param", "value", "expected"),
    [
        (_OPTIONAL_STRING, "10001", "10001"),
        (_OPTIONAL_STRING, "null", None),
        ({"$ref": "#/$defs/zip"}, "10001", "10001"),
    ],
    ids=["optional-string", "optional-string-null", "ref-to-string"],
)
def test_an_argument_is_kept_as_text_only_when_only_the_text_fits(param, value, expected):
    schema = {"type": "object", "properties": {"p": param}, "$defs": {"zip": {"type": "string"}}}
    text = f"[ToolCalls][Call:f][CallParameter:p]{value}[/CallParameter][/Call][/ToolCalls]"

    _, calls = extract_tool_calls(
        text, [FunctionTool(type="function", name="f", parameters=schema)]
    )

    assert orjson.loads(calls[0].function.arguments) == {"p": expected}


def test_a_strict_tool_whose_schema_is_not_valid_is_not_enforced():
    """Not being able to check is this server's failure, not the model's."""
    schema = {"type": "object", "required": "city"}  # `required` must be a list
    tool = FunctionTool(type="function", name="get_weather", parameters=schema, strict=True)

    _, calls = extract_tool_calls(TOOL_CALL_OUTPUT, [tool])

    assert [call.function.name for call in calls] == ["get_weather"]


def test_tool_call_with_language_tagged_parameter_fences():
    raw_llm_text = (
        "[ToolCalls]\n"
        "[Call:execute_code]\n"
        "[CallParameter:code]\n"
        "```python\n"
        "print('hello world')\n"
        "```\n"
        "[/CallParameter]\n"
        "[CallParameter:config]\n"
        "```json\n"
        '{"timeout": 30, "debug": true}\n'
        "```\n"
        "[/CallParameter]\n"
        "[/Call]\n"
        "[/ToolCalls]"
    )
    visible, tool_calls = extract_tool_calls(
        raw_llm_text, [FunctionTool(type="function", name="execute_code")]
    )
    assert visible == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "execute_code"
    args = orjson.loads(tool_calls[0].function.arguments)
    assert args["code"] == "print('hello world')"
    assert args["config"] == {"timeout": 30, "debug": True}


def test_generated_media_is_linked_once_and_a_second_copy_is_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr("app.server.media.MEDIA_STORE_DIR", tmp_path)
    monkeypatch.setattr("app.server.media.get_media_token", lambda name: f"t-{name}")
    for name in ("v1.mp4", "v1.jpg", "copy.mp4", "copy.jpg"):
        (tmp_path / name).write_bytes(b"x")
    seen: dict[str, str] = {}
    first = {"video": ("v1.mp4", "hash-v"), "video_thumbnail": ("v1.jpg", "hash-t")}
    copy = {"video": ("copy.mp4", "hash-v"), "video_thumbnail": ("copy.jpg", "hash-t")}

    link = "[![Clip](http://h/media/v1.jpg?token=t-v1.jpg)](http://h/media/v1.mp4?token=t-v1.mp4)"
    assert _media_markdown("Clip", first, seen, "http://h/") == link
    assert _media_markdown("Clip", copy, seen, "http://h/") == ""
    assert sorted(f.name for f in tmp_path.iterdir()) == ["v1.jpg", "v1.mp4"]
    # Audio without a thumbnail is a plain link; an item that produced no file yields nothing.
    audio = {"audio": ("a.mp3", "hash-a")}
    assert _media_markdown("Clip", audio, seen, "http://h/") == (
        "[Clip - Audio](http://h/media/a.mp3?token=t-a.mp3)"
    )
    assert _media_markdown("Clip", {}, seen, "http://h/") == ""


def test_a_tool_round_trip_is_sent_in_the_protocol_the_model_is_taught():
    def call(name: str, arguments: str) -> AppToolCall:
        return AppToolCall(id=name, function=AppToolCallFunction(name=name, arguments=arguments))

    def text(value: str) -> AppContentItem:
        return AppContentItem(type="text", text=value)

    messages = [
        AppMessage(role="user", content=[text("Weather?"), text("")]),
        AppMessage(
            role="assistant",
            content="Checking.",
            tool_calls=[call("get_weather", '{"city": "Hanoi", "days": 2}'), call("ping", "[1]")],
        ),
        AppMessage(role="tool", name="get_weather", content="Sunny"),
        AppMessage(role="tool", name="ping", content=[text("a"), text(""), text("b")]),
        AppMessage(role="user", content="Thanks"),
    ]

    prompt, files = asyncio.run(process_conversation(messages))

    assert files == []
    assert prompt == (
        "<|im_start|>user\nWeather?\n<|im_end|>\n"
        "<|im_start|>assistant\nChecking.\n[ToolCalls]\n[Call:get_weather]\n"
        "[CallParameter:city]\n```\nHanoi\n```\n[/CallParameter]\n"
        "[CallParameter:days]\n```\n2\n```\n[/CallParameter]\n[/Call]\n"
        "[Call:ping]\n```\n[1]\n```\n[/Call]\n[/ToolCalls]\n<|im_end|>\n"
        "<|im_start|>tool\n[ToolResults]\n"
        "[Result:get_weather]\n[ToolResult]\nSunny\n[/ToolResult]\n[/Result]\n"
        "[Result:ping]\n[ToolResult]\na\n\nb\n[/ToolResult]\n[/Result]\n"
        "[/ToolResults]\n<|im_end|>\n"
        "<|im_start|>user\nThanks\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
