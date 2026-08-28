# Gemini-FastAPI

[![Python 3.13](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ English | [中文](README.zh.md) ]

Web-based Gemini models wrapped into an OpenAI-compatible API. Powered by [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API).

**Call Gemini's web-based models via API without an API Key, completely free!**

## Features

- **No Google API Key Required**: Use web cookies to freely access Gemini's models via API.
- **Google Search Included**: Get up-to-date answers using web-based Gemini's search capabilities.
- **Conversation Persistence**: LMDB-based storage supporting multi-turn conversations.
- **Multi-modal Support**: Support for handling text, images, and file uploads.
- **Multi-account Load Balancing**: Distribute requests across multiple accounts with per-account proxy settings.

## Quick Start

**For Docker deployment, see the [Docker Deployment](#docker-deployment) section below.**

### Prerequisites

- Python >= 3.13
- Google account with Gemini access on web (Enable **[Gemini Apps activity](https://myactivity.google.com/product/gemini)** for best conversation persistence)
- `secure_1psid` and `secure_1psidts` cookies from Gemini web interface

### Installation

#### Using uv (Recommended)

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

#### Using pip

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
pip install -e .
```

### Basic Configuration

Edit `config/config.yaml` and provide at least one credential pair:

```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
      proxy: null # Optional proxy URL (null/empty keeps direct connection)
      impersonate: null # Optional browser impersonation target (null uses library default)
```

> [!NOTE]
> For details, refer to the [Configuration](#configuration) section below.

### Running the Server

```bash
# Using uv
uv run python run.py

# Using Python directly
python run.py
```

The server will start on `http://localhost:8000` by default.

## API Endpoints

The server provides several endpoints, including OpenAI-compatible ones.

### OpenAI-Compatible Endpoints

These endpoints use OpenAI-compatible wire formats while translating requests to Gemini Web.
Compatibility is intentionally broader than the controls exposed by the Gemini Web client. Valid
but unforwardable options are accepted for client compatibility, ignored, and recorded at debug
level so they do not prevent an otherwise representable request from running.

- **`GET /v1/models`**: Lists all supported Gemini models.
- **`POST /v1/chat/completions`**: Unified chat interface.
  - **Streaming**: Set `stream: true` to receive real-time delta chunks.
  - **Multi-modal**: Supports text, images, and file uploads.
  - **Tool Calling**: Supports function calling via the `tools` parameter.
  - **Structured Output**: Supports every `response_format` mode. `json_schema` is validated
    server-side against the supplied schema; `json_object` (JSON mode) only requires that the
    reply parses as JSON; `text` is the default and imposes nothing.

### Advanced Endpoints

- **`POST /v1/responses`**: Supports current `text.format` structured output (`text`,
  `json_object` and `json_schema`), external/inline file inputs, generated images, and tool
  calls. Files API `file_id` references are rejected because this wrapper does not expose an
  OpenAI Files API.

Schema enforcement follows OpenAI's own guarantee. With `strict: true`, a reply that does not
match the schema is an error. With `strict: false` or JSON mode, only a best effort is promised,
so a non-conforming reply is returned as text instead of failing the request. A turn that returns
a tool call is never judged against the schema, which constrains the final answer only.

`strict` defaults to `false` on both OpenAI surfaces, matching OpenAI itself, so one schema
behaves the same whichever endpoint it is sent to. The flag matters more here than upstream,
because Gemini Web has no constrained decoding — the schema is asked for in the prompt, so a
strict requirement the model narrowly misses costs the caller the whole reply. The Gemini-native
`generationConfig.responseSchema` / `responseJsonSchema` is always best-effort for the same
reason: it has no `strict` flag to turn off.

Only the model's own failures are enforced. A schema this wrapper cannot evaluate — one that is
not valid JSON Schema, or whose `$ref`s do not resolve — is still shown to the model but is not
used to judge the reply, and a schema whose regex keywords exhaust
`server.schema_validation_budget_seconds` leaves the reply unverified. None of these fail the
request, even under `strict`: they are gaps on this side, not violations by the model.

Because a JSON document can only be validated once it is complete, a streamed response carrying
a structured requirement is delivered as a single chunk after validation rather than
incrementally. This applies to Gemini-native `responseMimeType: application/json` as well.

Generation controls that Gemini Web does not expose—such as `temperature`, `top_p`, maximum
output-token limits, `parallel_tool_calls`, Gemini `generationConfig` fields, and safety settings—are
accepted but cannot affect upstream generation. They are ignored with a debug log. Only malformed
input, or content this wrapper cannot resolve at all (an unresolved Files API ID, a `cachedContent`
handle), is rejected—dropping those silently would change what the model is answering.

On the Gemini surface, `generationConfig.responseSchema` is the OpenAPI 3.0 subset (uppercase
type names, `nullable`) and is translated to JSON Schema before use; `responseJsonSchema` is
already JSON Schema and is validated as such. `toolConfig.functionCallingConfig.allowedFunctionNames`
narrows the tool list only in the `ANY` and `VALIDATED` modes that act on it.

### Utility Endpoints

- **`GET /health`**: Readiness endpoint. Conversation storage failures always return HTTP 503.
  Client failures follow the configured `gemini.guest_mode` health policy; the default
  `adaptive` policy returns 503 only when every Gemini client is unhealthy.
- **`GET /media/{filename}`**: Internal endpoint to serve generated media. Requires a valid token (automatically included in image URLs returned by the API).

## Docker Deployment

### Run with Options

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

> [!TIP]
> Add `CONFIG_GEMINI__CLIENTS__N__PROXY` only if you need a proxy; omit the variable to keep direct connections.
>
> `GEMINI_COOKIE_PATH` points to the directory inside the container where refreshed cookies are stored. Bind-mounting it (e.g. `-v $(pwd)/cache:/app/cache`) preserves those cookies across container rebuilds/recreations so you rarely need to re-authenticate.

### Run with Docker Compose

Create a `docker-compose.yml` file:

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
      - CONFIG_SERVER__HOST=0.0.0.0
      - CONFIG_SERVER__PORT=8000
      - CONFIG_SERVER__API_KEY=${API_KEY}
      - CONFIG_GEMINI__CLIENTS__0__ID=client-a
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID=${SECURE_1PSID}
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS=${SECURE_1PSIDTS}
      - GEMINI_COOKIE_PATH=/app/cache # must match the cache volume mount above
    restart: on-failure:3 # Avoid retrying too many times
```

Then run:

```bash
docker compose up -d
```

> [!IMPORTANT]
> Make sure to mount the `/app/data` volume to persist conversation data between container restarts.
> Also mount `/app/cache` so refreshed cookies (including rotated 1PSIDTS values) survive container rebuilds/recreates without re-auth.

## Configuration

The server reads a YAML configuration file located at `config/config.yaml`.

For details on each configuration option, refer to the comments in the [`config/config.yaml`](https://github.com/Nativu5/Gemini-FastAPI/blob/main/config/config.yaml) file.

### Environment Variable Overrides

> [!TIP]
> This feature is particularly useful for Docker deployments and production environments where you want to keep sensitive credentials separate from configuration files.

You can override any configuration option using environment variables with the `CONFIG_` prefix. Use double underscores (`__`) to represent nested keys, for example:

```bash
# Override server settings
export CONFIG_SERVER__API_KEY="your-secure-api-key"

# Override Gemini credentials for client 0
export CONFIG_GEMINI__CLIENTS__0__ID="client-a"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts"

# Override optional proxy settings for client 0
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"

# Override browser impersonation for client 0
export CONFIG_GEMINI__CLIENTS__0__IMPERSONATE="chrome"


# Override conversation storage size limit
export CONFIG_STORAGE__MAX_SIZE=268435456  # 256 MB

# Override the local HTTP-body resource guard (0 disables it)
export CONFIG_SERVER__MAX_REQUEST_BODY_BYTES=268435456

# Override the JSON Schema regex evaluation budget, in seconds
export CONFIG_SERVER__SCHEMA_VALIDATION_BUDGET_SECONDS=1.0
```

`max_request_body_bytes` is only a wrapper-side memory/resource safety ceiling. It is not an
OpenAI or Gemini API compatibility limit and does not claim to describe Gemini Web capacity.
Gemini Web remains authoritative for whether a request that passes this local guard is accepted.

### Client IDs and Conversation Reuse

Conversations are stored with the ID of the client that generated them.
Keep these identifiers stable in your configuration so that sessions remain valid
when you update the cookie list.

### Gemini Credentials

> [!WARNING]
> Keep these credentials secure and never commit them to version control. These cookies provide access to your Google account.

<!-- Keeps the credential warnings as separate blockquotes (markdownlint MD028). -->

> [!WARNING]
> **Session Stability**: If cookies expire frequently, use Firefox to extract cookies. Recent versions of Chromium-based browsers use "Device Bound Session Credentials", which improves security but causes cookies to remain valid for only a few hours and prevents them from being renewed.

To use Gemini-FastAPI, you need to extract your Gemini session cookies:

1. Open [Gemini](https://gemini.google.com/) in a private/incognito browser window and sign in
2. Open Developer Tools (F12)
3. Navigate to **Application** → **Storage** → **Cookies**
4. Find and copy the values for:
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

> [!IMPORTANT]
> **Enable [Gemini Apps activity](https://myactivity.google.com/product/gemini)** to ensure stable conversation persistence.
>
> While active chat turns may work temporarily without it, any transient error, TLS session restart, or server reboot can cause Google to expire the conversation metadata. If this setting is disabled, the model will **completely lose the context of your multi-turn conversation**, making old threads unreachable even if they are stored in your local LMDB.

<!-- Keeps the two callouts as separate blockquotes (markdownlint MD028). -->

> [!TIP]
> For detailed instructions, refer to the [HanaokaYuzu/Gemini-API authentication guide](https://github.com/HanaokaYuzu/Gemini-API?tab=readme-ov-file#authentication).

### Proxy Settings

Each client entry can be configured with a different proxy to work around rate limits. Omit the `proxy` field or set it to `null` or an empty string to keep a direct connection.

### Browser Impersonation

Each client can optionally set an `impersonate` value to control the TLS/HTTP fingerprint used by `curl_cffi`.

- Set to `null` (default) to use the library's default.
- Set to any value supported by [`curl_cffi`'s `BrowserTypeLiteral`](https://github.com/lexiforest/curl_cffi).
- The value is validated at startup; an invalid value will prevent the server from starting.

```yaml
gemini:
  clients:
    - id: "client-a"
      impersonate: "chrome" # Use Chrome fingerprint
    - id: "client-b"
      impersonate: null # Use library default
```

### Chat Session Mode

You can control whether requests use normal Google chats or Google's temporary chat mode:

```yaml
gemini:
  chat_mode: "normal" # "normal" or "temporary"
  guest_mode: "adaptive" # "strict", "adaptive", or "permissive"
  max_chars_per_request: 1000000
```

With `temporary`, conversations are not saved to the Google account. A temporary chat is still
continuable for as long as Google keeps the window open, so session reuse and conversation
storage work exactly as they do in normal mode.

When a stored chat can no longer be continued - after changing `chat_mode`, or once Google has
closed a temporary window - the server falls back to replaying the full conversation history
into a fresh chat, so the context is rebuilt rather than lost.

Google keeps at most one temporary window open per account and closes the previous one as soon
as a new conversation is created, so only the most recently opened temporary chat is still
continuable. The server tracks that chat per client and reuses **only** it; any older temporary
conversation is replayed in full into a fresh chat instead. There is no timeout to tune - the
rule follows Google's actual behaviour rather than guessing at an expiry.

That tracking is deliberately in-memory, so it is also cleared whenever the client session
restarts - an `auto_close` after inactivity, a server restart, or a redeploy. After any of
those, no window can be vouched for and every stored temporary conversation is replayed rather
than reused.

The same rule applies to a client running as a guest, regardless of `chat_mode`. When every
cookie group fails, or an authenticated session is rejected mid-flight because its cookies
expired, the client keeps serving text prompts without an account - and a guest chat is never
written to any history, so it behaves exactly like a temporary one. Each stored conversation
records the session window it belongs to, so chats opened while authenticated are never replayed
into a guest session and chats opened as a guest are never replayed once cookies are restored;
either crossing falls back to a full history replay in a fresh chat.

A guest session keeps the service up rather than taking it down, and requests degrade instead of
failing:

- The pool prefers authenticated clients, so a downgraded one only receives traffic when no
  authenticated client is left.
- Requests that need a file upload - attachments, or input long enough to be sent as
  `message.txt` - are routed to an authenticated client, including when a stored session would
  otherwise pin them to a guest one. If none exists, the request fails with an explicit message
  instead of Google's `Permission denied`.
- Google gives a guest no model choice, so the requested model is replaced by the default one it
  is allowed to use, logged as a warning. `/v1/models` advertises only models a client can
  actually serve.

`/health` reports a guest client as unhealthy - refresh its cookies to restore full capability.
The `guest_mode` setting controls how those unhealthy clients affect the readiness response:

- `strict`: return HTTP 503 when any client is unhealthy.
- `adaptive` (default): return HTTP 503 only when all clients are unhealthy; otherwise log a
  warning and remain ready.
- `permissive`: log a warning but do not change readiness, even when all clients are unhealthy.

All three modes log unhealthy clients. Conversation storage failures still return HTTP 503
regardless of `guest_mode`.

Otherwise this applies **only** in temporary mode. A normal chat opened by an authenticated
client is kept by Google until you delete it, so its metadata stays reusable indefinitely and
across restarts.

> [!WARNING]
> Google can close a temporary chat window at any time, without notice and mid-conversation.
> When that happens the reply may come back without the earlier context instead of raising an
> error, so the loss can be silent. The server replays the full history into a fresh chat when
> it can detect the chat is gone, but detection is not guaranteed. Prefer `normal` for long or
> context-sensitive conversations, and treat `temporary` as best-effort continuity.

Because temporary chats accept a smaller payload, the server applies an additional 10% reduction
on top of the standard safety margin, so the effective input limit becomes 81% of
`max_chars_per_request` instead of 90%. Input exceeding the effective limit is still sent as a
`message.txt` attachment in both modes.

Environment variable equivalent:

```bash
export CONFIG_GEMINI__CHAT_MODE="temporary"
export CONFIG_GEMINI__GUEST_MODE="adaptive"
```

### Models

Models are discovered from Google at startup - there is nothing to configure. Each client reads
the models its own account may use and builds the request headers for them at runtime, so a newly
launched model is served as soon as Google offers it. `GET /v1/models` lists what the running
clients can actually serve.

Requests may name a model by its canonical name (currently `gemini-pro`, `gemini-flash` and
`gemini-flash-lite`), by an alias or display name (`pro`, `Flash Lite`) or by its internal id;
all forms resolve to the same conversation history. Call `GET /v1/models` for the list your own
accounts see - the names come from Google, not from this project. Only discovered models are
served: a name no client offers is rejected with `400` rather than quietly answered by a
different model.

> [!NOTE]
> The `models` and `model_strategy` settings are gone, as is any use of the library's removed
> static `Model` name lookups. They existed to hand-write model headers while the library lagged behind
> new releases, which dynamic discovery has made unnecessary - and hardcoded headers now risk
> pinning requests to a stale model. Both keys are simply ignored if left in a config file or
> environment.

## Acknowledgments

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - The underlying Gemini web API client
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - This project originated from this repository. After extensive refactoring and engineering improvements, it has evolved into an independent project, featuring multi-turn conversation reuse among other enhancements. Special thanks for the inspiration and foundational work provided.

## Disclaimer

This project is not affiliated with Google or OpenAI and is intended solely for educational and research purposes. It uses reverse-engineered APIs and may not comply with Google's Terms of Service. Use at your own risk.
