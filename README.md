# Gemini-FastAPI

[![Python 3.13](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ English | [中文](README.zh.md) ]

Web-based Gemini models wrapped into an OpenAI-compatible API. Powered by [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API).

**Call Gemini's web-based models via API without an API Key, completely free!**

## Features

- **No Google API key**: authenticates with the cookies of a signed-in Gemini web session.
- **Three API surfaces**: OpenAI Chat Completions, OpenAI Responses, and Gemini-native `generateContent`.
- **Streaming, tool calling and structured output** on every surface.
- **Multi-modal**: image and file input, and image generation.
- **Google Search included**: answers can draw on Gemini's built-in web search.
- **Conversation reuse**: a follow-up request continues the same Google chat instead of resending the history.
- **Multiple accounts**: requests are spread across accounts, each with its own proxy.

## Quick Start

You need Python 3.13+, [uv](https://docs.astral.sh/uv/), and a Google account that can use [Gemini](https://gemini.google.com/). To run in a container instead, see [Docker](#docker).

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

Put your cookies ([how to get them](#getting-your-cookies)) in `config/config.yaml`:

```yaml
gemini:
  clients:
    - id: "client-a"
      cookies: "__Secure-1PSID=...; __Secure-1PSIDTS=...; ..."
```

Start the server, which listens on `http://localhost:8000`:

```bash
uv run python run.py
```

Model names come from Google and differ between accounts, so list them first and use one in a request:

```bash
curl http://localhost:8000/v1/models
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MODEL_FROM_THE_LIST", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Getting your cookies

1. Open [Gemini](https://gemini.google.com/) in a private window and sign in to the one account the server should use. With several accounts signed in, the server uses the first.
2. Open Developer Tools (F12), go to **Network**, and reload the page.
3. Select a request to `gemini.google.com`, such as `app` or `batchexecute`.
4. Under **Request Headers**, copy the whole value of `cookie`.
5. Close the private window without signing out, so the browser stops renewing the cookies you copied.

> [!WARNING]
> These cookies give access to your Google account. Keep them out of version control.

- **Use Firefox if cookies expire within hours.** Chromium-based browsers bind sessions to the device, so cookies copied from them cannot be renewed.
- **Turn on [Gemini Apps activity](https://myactivity.google.com/product/gemini).** Without it Google can drop a chat at any time, and a multi-turn conversation loses its context.

## Docker

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  -e CONFIG_SERVER__API_KEY="your-api-key-here" \
  -e CONFIG_GEMINI__CLIENTS__0__ID="client-a" \
  -e CONFIG_GEMINI__CLIENTS__0__COOKIES='your-cookie-header' \
  -e GEMINI_COOKIE_PATH="/app/cache" \
  ghcr.io/nativu5/gemini-fastapi
```

Or with Docker Compose:

```yaml
services:
  gemini-fastapi:
    image: ghcr.io/nativu5/gemini-fastapi:latest
    ports:
      - "8000:8000"
    volumes:
      # - ./config:/app/config      # Uncomment to use a custom config file
      # - ./certs:/app/certs        # Uncomment to enable HTTPS with your certs
      - ./data:/app/data
      - ./cache:/app/cache
    environment:
      - CONFIG_SERVER__API_KEY=${API_KEY}
      - CONFIG_GEMINI__CLIENTS__0__ID=client-a
      - CONFIG_GEMINI__CLIENTS__0__COOKIES=${GEMINI_COOKIES}
      - GEMINI_COOKIE_PATH=/app/cache # Must match the cache volume above
    restart: on-failure:3
```

> [!IMPORTANT]
> Mount both volumes. `/app/data` holds stored conversations and generated media. `/app/cache` holds the cookies the server refreshes, so you do not have to sign in again after recreating the container.

## API

| Endpoint | Description |
| --- | --- |
| `GET /v1/models` | Models the configured accounts can serve. |
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming, tools, image and file input, `response_format`. |
| `POST /v1/responses` | OpenAI Responses: streaming, tools, image and file input, image generation, `text.format`. |
| `GET /v1beta/models`, `GET /v1beta/models/{model}` | Gemini-native model listing. |
| `POST /v1beta/models/{model}:generateContent` | Gemini-native generation. |
| `POST /v1beta/models/{model}:streamGenerateContent` | Gemini-native streaming generation. |
| `GET /health` | Readiness: storage and per-account status. |
| `GET /media/{filename}` | Generated images and media. Responses link here with a signed token. |

If `server.api_key` is set, send it as `Authorization: Bearer <key>`. The Gemini-native endpoints also accept the `x-goog-api-key` header and the `key` query parameter. A request to `/v1/models…` that sends its key either of those two ways is served by them as if it were `/v1beta/models…`, because that is where Google's SDKs send requests unless told otherwise. With no key set, the API and the media links are open to anyone who can reach the server.

### Compatibility

Gemini Web offers fewer controls than the APIs this server imitates. A valid request is not rejected because of an option that cannot be forwarded.

- **Ignored options.** `temperature`, `top_p`, output-token limits, `parallel_tool_calls`, Gemini `generationConfig` fields and safety settings are accepted and have no effect.
- **Rejected input.** Malformed requests, and anything the server cannot resolve at all: a Files API `file_id`, a `cachedContent` handle, or earlier turns referenced by `previous_response_id` or `conversation` on `/v1/responses`. Send those turns in `input` instead.
- **Errors use the envelope of the API that was called**: `{"error": {"message", "type", "param", "code"}}` on the OpenAI endpoints, `{"error": {"code", "message", "status"}}` on the Gemini-native endpoints. A malformed request is a `400` on both.
- **Generated videos and audio are returned as links** in the reply text on every endpoint. Images are links too on the OpenAI endpoints, where `/v1/responses` adds an `image_generation_call` item for each; the Gemini-native endpoints return them as `inlineData` parts instead.
- **Structured output is requested in the prompt** and checked by the server afterwards, because Gemini Web has no constrained decoding.
  - With `strict: true`, a reply that does not match the schema is an error: `502`, or an error event when streaming.
  - With `strict: false` (the default on both OpenAI surfaces) or JSON mode, the reply is returned as it is.
  - Gemini-native `responseSchema` and `responseJsonSchema` are always best effort.
  - A turn that returns a tool call is not checked against the response schema.
- **Tool calls are requested in the prompt as well**, and checked once the reply is complete.
  - A call to a tool the request does not offer is an error: `502`, or an error event when streaming. Any call under `tool_choice: "none"` or Gemini-native mode `NONE` counts as one.
  - So is a call whose arguments fail the schema of a `strict: true` function, or of any function under Gemini-native modes `ANY` and `VALIDATED`. Other calls are returned even when their arguments miss the schema, with a warning in the log. An omitted `strict` is not enforced, and `/v1/responses` reports it back as `false`.
  - Arguments keep their JSON types, except where only the text fits the parameter's schema: `10001` for a string parameter stays `"10001"`.
- **Schemas the server cannot evaluate are still sent to the model, but not enforced**, and do not fail the request. That covers invalid JSON Schema, a `responseSchema` that does not translate to JSON Schema, a `$ref` that does not resolve (remote references are never fetched), and regex keywords that exceed `server.schema_validation_budget_seconds`. The one exception is a `responseJsonSchema` that is not valid JSON Schema, which is a `400`.
- **Streaming with a schema** delivers the answer as one chunk, since JSON can only be validated once it is complete.
- **Google's SDKs work with their default settings** for chats with text and images, streamed or not, and for `responseSchema` and function tools. Field names are accepted in camelCase or snake_case, as Google accepts them.

## Configuration

Settings live in [`config/config.yaml`](config/config.yaml), and environment variables can override any of them. [CONFIG.md](CONFIG.md) covers both, along with accounts, chat modes, guests and rejected cookies, models, gems and input size.

## Maintenance

[`scripts/USAGE.md`](scripts/USAGE.md) covers the scripts for dumping and pruning the conversation store.

## Acknowledgments

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - The underlying Gemini web API client
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - This project originated from this repository. After extensive refactoring and engineering improvements, it has evolved into an independent project, featuring multi-turn conversation reuse among other enhancements. Special thanks for the inspiration and foundational work provided.

## Disclaimer

This project is not affiliated with Google or OpenAI and is intended solely for educational and research purposes. It uses reverse-engineered APIs and may not comply with Google's Terms of Service. Use at your own risk.
