"""Wire-format compatibility tests for the OpenAI- and Gemini-shaped surfaces.

The contract these lock down: a request naming only standard attributes must never be rejected
just because Gemini Web cannot honour one of them. Options that cannot be forwarded are accepted
and dropped; only malformed or unrepresentable *content* is refused.
"""

import base64
import time

import orjson
import pytest
from pydantic import ValidationError

from app.models.core import AppContentItem, AppToolCall, AppToolCallFunction
from app.models.gemini_models import GeminiGenerateContentRequest, GeminiGenerationConfig
from app.models.models import (
    ChatCompletionNamedToolChoice,
    FunctionCallOutput,
    ResponseCreateRequest,
    ResponseFormatTextJSONSchemaConfig,
    ResponseInputMessage,
    ResponseUsage,
    StructuredOutputRequirement,
    ToolChoiceFunction,
    ToolChoiceTypes,
)
from app.server.chat import (
    _build_structured_requirement,
    _create_responses_standard_payload,
    _responses_response_format,
    _sse_error,
    _tool_choice_declaration_error,
    _tool_choice_failure,
    _validate_responses_input,
)
from app.server.gemini import (
    _gemini_response_schema,
    _gemini_structured_requirement,
    _gemini_tools_to_internal,
    _validate_gemini_request,
)
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
OBJECT_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
_EMPTY_USAGE = ResponseUsage(input_tokens=0, output_tokens=0, total_tokens=0)


def _requirement(schema: dict, *, strict: bool = True) -> StructuredOutputRequirement:
    return StructuredOutputRequirement(
        schema_name="r", schema=schema, instruction="", raw_format={}, strict=strict
    )


def _gemini_request(**overrides) -> GeminiGenerateContentRequest:
    payload = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], **overrides}
    return GeminiGenerateContentRequest.model_validate(payload)


def _generation_config(request: GeminiGenerateContentRequest) -> GeminiGenerationConfig:
    """Narrow the optional generationConfig the caller just supplied."""
    config = request.generationConfig
    assert config is not None
    return config


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


def test_json_schema_sets_strict_from_the_request():
    for strict in (True, False):
        requirement = _build_structured_requirement(
            {"type": "json_schema", "json_schema": {"schema": OBJECT_SCHEMA, "strict": strict}}
        )
        assert requirement is not None
        assert requirement.strict is strict


@pytest.mark.parametrize("strict", ["false", "true", 0, 1, None])
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


@pytest.mark.parametrize("response_format", [None, {}, "json_schema", ["json_schema"]])
def test_absent_or_non_object_response_format_is_ignored(response_format):
    assert _build_structured_requirement(response_format) is None


def test_json_schema_defaults_to_best_effort_and_a_generated_name():
    """OpenAI defaults `strict` to false, and so must we.

    The flag is not decorative here: a strict miss costs the caller the whole answer, and this
    wrapper prompts for schema adherence rather than constraining decoding. A caller who never
    asked for strict enforcement must not be opted into losing replies.
    """
    requirement = _build_structured_requirement(
        {"type": "json_schema", "json_schema": {"schema": OBJECT_SCHEMA, "name": ""}}
    )
    assert requirement is not None
    assert requirement.schema_name == "response"
    assert requirement.strict is False
    assert STRICT_SCHEMA_ADHERENCE_PROMPT not in requirement.instruction
    # The schema is still asked for; only the failure mode softens.
    assert SCHEMA_ADHERENCE_PROMPT in requirement.instruction


def test_non_strict_schema_omits_the_exact_conformance_line():
    requirement = _build_structured_requirement(
        {"type": "json_schema", "json_schema": {"schema": OBJECT_SCHEMA, "strict": False}}
    )
    assert requirement is not None
    assert STRICT_SCHEMA_ADHERENCE_PROMPT not in requirement.instruction


def test_tool_call_turn_is_not_failed_by_a_response_format():
    """The schema constrains the final answer, not a turn that asks for a tool."""
    _, visible, _, tool_calls = process_llm_output(
        None, TOOL_CALL_OUTPUT, _requirement(OBJECT_SCHEMA)
    )
    assert [call.function.name for call in tool_calls] == ["get_weather"]
    assert visible == ""


def test_strict_violation_raises_and_best_effort_violation_degrades():
    with pytest.raises(StructuredOutputValidationError):
        process_llm_output(None, '{"b": 1}', _requirement(OBJECT_SCHEMA))

    _, visible, _, _ = process_llm_output(
        None, '{"b": 1}', _requirement(OBJECT_SCHEMA, strict=False)
    )
    assert visible == '{"b": 1}'


@pytest.mark.parametrize("raw_text", ["", "   \n  "])
def test_a_reply_with_no_text_is_not_a_schema_violation(raw_text):
    """An image-only or empty turn has nothing to validate; failing it would invent an error."""
    _, visible, storage, _ = process_llm_output(None, raw_text, _requirement(OBJECT_SCHEMA))
    assert visible == storage == ""


def test_text_alongside_a_tool_call_is_still_canonicalized():
    raw = f'```json\n{{"a": "x"}}\n```\n{TOOL_CALL_OUTPUT}'
    _, visible, _, tool_calls = process_llm_output(None, raw, _requirement(OBJECT_SCHEMA))
    assert [call.function.name for call in tool_calls] == ["get_weather"]
    assert visible == '{"a":"x"}'


def test_conforming_output_is_canonicalized():
    _, visible, storage, _ = process_llm_output(
        None, '```json\n{"a": "x"}\n```', _requirement(OBJECT_SCHEMA)
    )
    assert visible == storage == '{"a":"x"}'


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "http://169.254.169.254/latest/meta-data"},  # unresolvable remote reference
        {"$ref": "#/definitions/missing"},  # dangling local reference
        {"type": "OBJECT"},  # foreign dialect that slipped through
    ],
)
def test_unusable_schemas_leave_the_reply_unverified_rather_than_failing_it(schema):
    """Failing to run the check is our problem, not the model's, so it cannot be a violation."""
    assert canonicalize_structured_output('{"a": 1}', _requirement(schema)) == '{"a":1}'
    # And it must not reach the caller as an error, even under strict.
    _, visible, _, _ = process_llm_output(None, '{"a": 1}', _requirement(schema))
    assert visible == '{"a":1}'


def test_json_mode_requires_only_that_the_payload_parses():
    requirement = _requirement({}, strict=False)
    assert canonicalize_structured_output('{"anything": [1]}', requirement) == '{"anything":[1]}'
    assert canonicalize_structured_output("not json", requirement) is None


def test_pathological_schema_regex_is_bounded(monkeypatch):
    """A client-controlled pattern must not monopolize the async server thread."""
    monkeypatch.setattr("app.utils.helper.SCHEMA_REGEX_BUDGET_SECONDS", 0.005)
    requirement = _requirement(
        {
            "type": "object",
            "properties": {"value": {"type": "string", "pattern": "^(a+)+$"}},
            "required": ["value"],
        }
    )
    started = time.perf_counter()
    result = canonicalize_structured_output('{"value":"' + "a" * 100 + 'b"}', requirement)
    assert time.perf_counter() - started < 0.5
    assert result is None


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


@pytest.mark.parametrize("strict", ["false", "true", 0, 1])
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


def test_both_surfaces_resolve_an_omitted_strict_the_same_way():
    """One schema must not hard-fail on Responses and degrade on Chat Completions."""
    responses_request = ResponseCreateRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "text": {"format": {"type": "json_schema", "name": "r", "schema": OBJECT_SCHEMA}},
        }
    )
    chat = _build_structured_requirement(
        {"type": "json_schema", "json_schema": {"name": "r", "schema": OBJECT_SCHEMA}}
    )
    responses = _build_structured_requirement(_responses_response_format(responses_request))
    assert chat is not None
    assert responses is not None
    assert chat.strict is responses.strict is False


@pytest.mark.parametrize(("requested", "applied"), [(True, True), (False, False), (None, False)])
def test_the_echoed_strict_reports_what_was_enforced(requested, applied):
    json_schema: dict = {"name": "r", "schema": OBJECT_SCHEMA}
    if requested is not None:
        json_schema["strict"] = requested
    request = ResponseCreateRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "response_format": {"type": "json_schema", "json_schema": json_schema},
        }
    )
    requirement = _build_structured_requirement(_responses_response_format(request))
    assert requirement is not None
    assert requirement.strict is applied

    payload = _create_responses_standard_payload(
        "resp_1", 0, "m", None, [], [], _EMPTY_USAGE, request, requirement
    )
    text_format = payload.text.format if payload.text else None
    assert isinstance(text_format, ResponseFormatTextJSONSchemaConfig)
    assert text_format.strict is applied


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
    assert _gemini_response_schema(_generation_config(request)) == {
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


def test_untranslatable_response_schema_drops_enforcement_rather_than_failing():
    request = _gemini_request(
        generationConfig={
            "responseMimeType": "application/json",
            "responseSchema": {"type": "WHAT"},
        }
    )
    assert _validate_gemini_request(request) is None
    assert _gemini_response_schema(_generation_config(request)) is None


def test_the_gemini_surface_asks_for_the_schema_without_enforcing_it():
    """Google guarantees conformance by constrained decoding; this wrapper can only ask.

    The native surface has no `strict` flag for a caller to turn off, so failing the request on
    a near-miss would throw away an answer with no way to opt out of that.
    """
    request = _gemini_request(
        generationConfig={"responseMimeType": "application/json", "responseSchema": OBJECT_SCHEMA}
    )
    schema, requirement = _gemini_structured_requirement(request)

    assert schema == OBJECT_SCHEMA
    assert requirement is not None
    assert requirement.strict is False
    assert STRICT_SCHEMA_ADHERENCE_PROMPT not in requirement.instruction

    # A violation therefore comes back as text rather than costing the caller the reply.
    _, visible, _, _ = process_llm_output(None, '{"b": 1}', requirement)
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
    schema, requirement = _gemini_structured_requirement(
        _gemini_request(generationConfig=generation_config)
    )
    assert schema in (None, {})
    assert requirement is not None
    assert requirement.schema == {}
    assert requirement.strict is False
    assert canonicalize_structured_output('{"valid": true}', requirement) == '{"valid":true}'


def test_no_response_schema_yields_no_requirement():
    schema, requirement = _gemini_structured_requirement(_gemini_request())
    assert schema is None
    assert requirement is None


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


def test_normalize_openapi_schema_drops_nullable_false_without_widening():
    assert normalize_openapi_schema({"type": "STRING", "nullable": False}) == {"type": "string"}


@pytest.mark.parametrize("value", ["text", None, 7, True])
def test_normalize_openapi_schema_passes_non_schemas_through(value):
    assert normalize_openapi_schema(value) == value


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
    result = _validate_responses_input(items)
    if expected is None:
        assert result is None
    else:
        assert expected in (result or "")


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("done", None),
        ([{"type": "input_text", "text": "done"}], None),
        ([{"type": "input_file", "file_id": "file-1"}], "file_id inputs"),
    ],
)
def test_tool_result_parts_are_validated_too(output, expected):
    items = [FunctionCallOutput.model_validate({"call_id": "c", "output": output})]
    result = _validate_responses_input(items)
    if expected is None:
        assert result is None
    else:
        assert expected in (result or "")


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


def test_sse_errors_terminate_the_stream():
    frame = _sse_error("boom", "server_error")
    assert frame.endswith("data: [DONE]\n\n")
    assert '"message":"boom"' in frame


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
        ("```sql\nSELECT 1;\n```", "SELECT 1;"),
        ("```bash\necho hi\n```", "echo hi"),
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
    visible, tool_calls = extract_tool_calls(raw_llm_text)
    assert visible == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "execute_code"
    args = orjson.loads(tool_calls[0].function.arguments)
    assert args["code"] == "print('hello world')"
    assert args["config"] == {"timeout": 30, "debug": True}
