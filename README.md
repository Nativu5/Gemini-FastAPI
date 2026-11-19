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
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
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

1. Open [Gemini](https://gemini.google.com/) in a private window and sign in.
2. Open Developer Tools (F12) and go to **Application** → **Storage** → **Cookies**.
3. Copy the values of `__Secure-1PSID` and `__Secure-1PSIDTS`.

The [Gemini-API authentication guide](https://github.com/HanaokaYuzu/Gemini-API?tab=readme-ov-file#authentication) has more detail.

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
  -e CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid" \
  -e CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts" \
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
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID=${SECURE_1PSID}
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS=${SECURE_1PSIDTS}
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
  - A turn that returns a tool call is not checked.
- **Schemas the server cannot evaluate are still sent to the model, but not enforced**, and do not fail the request. The one exception is a `responseJsonSchema` that is not valid JSON Schema, which is a `400`. That covers invalid JSON Schema, a `responseSchema` that does not translate to JSON Schema, a `$ref` that does not resolve (remote references are never fetched), and regex keywords that exceed `server.schema_validation_budget_seconds`.
- **Streaming with a schema** delivers the answer as one chunk, since JSON can only be validated once it is complete.
- **Google's SDKs work with their default settings** for chats with text and images, streamed or not, and for `responseSchema` and function tools. Field names are accepted in camelCase or snake_case, as Google accepts them.

## Configuration

Every option is described in [`config/config.yaml`](config/config.yaml). Any of them can be overridden by an environment variable named `CONFIG_` plus the path to the key, with `__` between levels:

```bash
export CONFIG_SERVER__API_KEY="your-secure-api-key"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"
export CONFIG_GEMINI__CHAT_MODE="auto_delete"
```

Two more variables are read directly: `CONFIG_PATH` selects a different config file, and `GEMINI_COOKIE_PATH` sets the directory where refreshed cookies are cached.

### Accounts

Each entry under `gemini.clients` is one Google account.

- `id`: stored with every conversation the account opens. Keep it stable when you update cookies, or stored conversations can no longer be continued.
- `proxy`: an optional proxy URL for this account only.
- `impersonate`: an optional browser fingerprint for [`curl_cffi`](https://github.com/lexiforest/curl_cffi), such as `chrome`. An unsupported value stops the server at startup.

### Chat modes

`gemini.chat_mode` decides what happens to the chats the server opens in your Google account.

| Mode | Kept in the Google account | Context across turns |
| --- | --- | --- |
| `normal` (default) | Yes, until you delete it | Reliable |
| `temporary` | Never saved | Best effort |
| `auto_delete` | Yes, then deleted once idle | Reliable |

**`temporary`** uses Google's temporary chats. Google keeps one temporary chat open per account and may close it at any time without an error, so the reply can come back without the earlier context. The server continues only the most recent temporary chat and replays the full history into a new one otherwise. The input limit is also lower. Prefer another mode for long conversations.

**`auto_delete`** uses normal chats and removes them from both the Google account and the local store after `gemini.auto_delete_ttl` seconds without a request (default `86400`, minimum `60`).

- Only chats the server opened in this mode are deleted. Chats from other modes, and everything else in the account, are left alone.
- Deletion runs on a periodic pass, every half TTL (between one minute and six hours), so a chat can outlive its TTL by up to one interval.
- A deletion that fails, for example while the account is signed out, is retried on later passes and abandoned with an error in the log after five attempts.
- Generated media expires on the same TTL.

After `chat_mode` changes, stored chats from the other side of the `auto_delete` boundary are not continued. Their history is replayed into a new chat, and chats opened under `auto_delete` are still deleted on schedule.

### Guest sessions

When an account's cookies are rejected, that client keeps running as a guest instead of failing. A guest can answer text prompts only: it cannot upload files, cannot choose a model, and keeps no history.

- Authenticated clients are preferred. A guest receives requests only when no authenticated client is left.
- A request that needs an upload fails with a clear message if only guests remain.
- `/health` reports a guest as unhealthy. `gemini.guest_mode` sets when that turns into HTTP 503:

| `guest_mode` | `/health` returns 503 when |
| --- | --- |
| `strict` | any client is unhealthy |
| `adaptive` (default) | every client is unhealthy |
| `permissive` | never because of clients |

A storage failure always returns 503.

### Models

Models are discovered from each account at startup, so there is nothing to configure. Which models you get, and what they are called, depends on the account's plan.

- A request can name a model by its listed name or its display name.
- Matching is loose. Version numbers are ignored and one word of a listed name is enough, so `gemini-2.5-pro` and `pro` both resolve to `gemini-pro`.
- A name that matches nothing is rejected with `400`. The server never falls back to another model.

### Gems

> [!WARNING]
> Google has announced the [retirement of Gems](https://support.google.com/gemini/answer/18560919): for personal accounts they become read-only from 2026-11-17 and are removed on 2026-12-17; Workspace accounts follow in March 2027 and education accounts in June 2027. This feature only works while the account still has Gems. Once it does not, set the policy back to `off`: a gem request that Google rejects is retried for up to about 15 seconds, restarting the client each time, before the prompt is sent inline. That happens on every request.

Off by default; set `gemini.gems.policy` to turn it on. The system prompt at the start of a `/v1/chat/completions` or `/v1/responses` request is then stored once as a gem (Gemini's saved custom instructions) in the Google account and referenced from then on, instead of being sent inside the message. Conversations with the same system prompt share one gem.

| `policy` | System prompt |
| --- | --- |
| `off` (default) | Sent inline |
| `fetch_only` | Uses a gem that already exists for it; never creates one |
| `create_on_demand` | Creates a gem the first time it is seen |

> [!WARNING]
> `create_on_demand` creates gems named `fastapi_policy_...` in the Google account.

- There is one gem per distinct prompt, so use this for stable prompts. A prompt that changes with every request, for example one that contains the time, creates a new gem each time.
- Creation is limited by `create_rate_limit_per_minute` (default `4`) and `managed_gems_max_total` (default `200`). Past a limit, or whenever a gem cannot be used, the prompt is sent inline.
- Gems stay in the account until you delete them. With `cleanup.enabled: true`, under either policy, gems unused for `cleanup.unused_days` (default `7`) are deleted by a cleanup that runs at startup and every 6 hours after that, and each gem's description is updated to record when it was last used.
- Guest sessions and Gemini-native requests always send the prompt inline.

### Input size

Input longer than 90% of `gemini.max_chars_per_request` (81% in `temporary` mode) is sent to Gemini as a `message.txt` attachment. `server.max_request_body_bytes` is a separate local guard on the HTTP body and says nothing about what Gemini accepts.

### Removed settings

`gemini.models`, `gemini.model_strategy`, `gemini.oversized_context_strategy`, `storage.images_path` and, under `gemini.gems`, `enabled`, `fetch_on_init`, `include_hidden_on_fetch`, `policies.default_policy` and `cleanup.require_managed_marker` no longer exist and are ignored if present. Generated media is stored under `storage.media_path`.

## Maintenance

[`scripts/USAGE.md`](scripts/USAGE.md) covers the scripts for dumping and pruning the conversation store.

## Acknowledgments

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - The underlying Gemini web API client
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - This project originated from this repository. After extensive refactoring and engineering improvements, it has evolved into an independent project, featuring multi-turn conversation reuse among other enhancements. Special thanks for the inspiration and foundational work provided.

## Disclaimer

This project is not affiliated with Google or OpenAI and is intended solely for educational and research purposes. It uses reverse-engineered APIs and may not comply with Google's Terms of Service. Use at your own risk.
