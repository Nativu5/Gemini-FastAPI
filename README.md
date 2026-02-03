# Gemini-FastAPI

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ English | [中文](README.zh.md) ]

Web-based Gemini models wrapped into an OpenAI-compatible API. Powered by [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API).

**✅ Call Gemini's web-based models via API without an API Key, completely free!**

## Features

- **🔐 No Google API Key Required**: Use web cookies to freely access Gemini's models via API.
- **🔍 Google Search Included**: Get up-to-date answers using web-based Gemini's search capabilities.
- **💾 Conversation Persistence**: LMDB-based storage supporting multi-turn conversations.
- **🖼️ Multi-modal Support**: Support for handling text, images, and file uploads.
- **⚖️ Multi-account Load Balancing & Fallback**: Distribute requests across multiple accounts. Automatically switches to a healthy account if the current one fails (e.g. due to rate limits or expired cookies), ensuring high availability.

## Quick Start

**For Docker deployment, see the [Docker Deployment](#docker-deployment) section below.**

### Prerequisites

- Python 3.12
- Google account with Gemini access on web
- `secure_1psid` and `secure_1psidts` cookies from Gemini web interface

### Installation

#### Using uv (Recommended)

```bash
git clone https://github.com/Sagit-chu/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

#### Using pip

```bash
git clone https://github.com/Sagit-chu/Gemini-FastAPI.git
cd Gemini-FastAPI
pip install -e .
```

### Configuration

Edit `config/config.yaml` and provide at least one credential pair:

```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
      proxy: null # Optional proxy URL (null/empty keeps direct connection)
```

> [!NOTE]
> For details, refer to the [Configuration](#configuration-1) section below.

### Running the Server

```bash
# Using uv
uv run python run.py

# Using Python directly
python run.py
```

The server will start on `http://localhost:8000` by default.

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
  ghcr.io/Sagit-chu/gemini-fastapi
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
    image: ghcr.io/Sagit-chu/gemini-fastapi:latest
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

For details on each configuration option, refer to the comments in the [`config/config.yaml`](https://github.com/Sagit-chu/Gemini-FastAPI/blob/main/config/config.yaml) file.

### Gems (`model=gem:<id>`)

You can define reusable presets ("gems") in the config file and select them per request by setting `model` to `gem:<id>`.

When a gem is selected:

- The server uses the gem definition's `model` as the actual Gemini model name for the upstream call.
- The OpenAI-compatible response tries to echo back the client-provided `model` (i.e. `gem:<id>`) to keep client-side consistency.
- **Auto-Sync**: On startup, the server checks if these gems exist on your Google account. If missing, it automatically creates them using the provided `id` (as title) and `system_prompt`.

Example config:

```yaml
gemini:
  gems:
    - id: "coding-helper"
      model: "gemini-2.0-flash"
      system_prompt: "You are an expert software engineer."
      tool_policy: "allow" # allow | disallow | auto
      default_temperature: 0.2
      top_p: 0.8
      max_output_tokens: 8192
```

Example request (Chat Completions):

```json
{
  "model": "gem:default-gem",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

Example request (Responses API):

```json
{
  "model": "gem:default-gem",
  "input": "Hello"
}
```

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

# Override conversation storage size limit
export CONFIG_STORAGE__MAX_SIZE=268435456  # 256 MB
```

### Client IDs and Conversation Reuse

Conversations are stored with the ID of the client that generated them.
Keep these identifiers stable in your configuration so that sessions remain valid
when you update the cookie list.

### Gemini Credentials

> [!WARNING]
> Keep these credentials secure and never commit them to version control. These cookies provide access to your Google account.

To use Gemini-FastAPI, you need to extract your Gemini session cookies:

1. Open [Gemini](https://gemini.google.com/) in a private/incognito browser window and sign in
2. Open Developer Tools (F12)
3. Navigate to **Application** → **Storage** → **Cookies**
4. Find and copy the values for:
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

> [!TIP]
> For detailed instructions, refer to the [HanaokaYuzu/Gemini-API authentication guide](https://github.com/HanaokaYuzu/Gemini-API?tab=readme-ov-file#authentication).

### Proxy Settings

Each client entry can be configured with a different proxy to work around rate limits. Omit the `proxy` field or set it to `null` or an empty string to keep a direct connection.

### Custom Models

You can define custom models in `config/config.yaml` or via environment variables.

#### YAML Configuration

```yaml
gemini:
  model_strategy: "append" # "append" (default + custom) or "overwrite" (custom only)
  models:
    - model_name: "gemini-3.0-pro"
      model_header:
        x-goog-ext-525001261-jspb: '[1,null,null,null,"9d8ca3786ebdfbea",null,null,0,[4],null,null,1]'
```

#### Environment Variables

You can supply models as a JSON string or list structure via `CONFIG_GEMINI__MODELS`. This provides a flexible way to override settings via the shell or in automated environments (e.g. Docker) without modifying the configuration file.

```bash
export CONFIG_GEMINI__MODEL_STRATEGY="overwrite"
export CONFIG_GEMINI__MODELS='[{"model_name": "gemini-3.0-pro", "model_header": {"x-goog-ext-525001261-jspb": "[1,null,null,null,\"9d8ca3786ebdfbea\",null,null,0,[4],null,null,1]"}}]'
```

## Acknowledgments

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - The underlying Gemini web API client
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - This project originated from this repository. After extensive refactoring and engineering improvements, it has evolved into an independent project, featuring multi-turn conversation reuse among other enhancements. Special thanks for the inspiration and foundational work provided.

## Disclaimer

This project is not affiliated with Google or OpenAI and is intended solely for educational and research purposes. It uses reverse-engineered APIs and may not comply with Google's Terms of Service. Use at your own risk.
