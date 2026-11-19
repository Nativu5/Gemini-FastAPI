# Gemini-FastAPI

[![Python 3.13](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ [English](README.md) | 中文 ]

将 Gemini 网页端模型封装为兼容 OpenAI API 的 API Server。基于 [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 实现。

**无需 API Key，免费通过 API 调用 Gemini 网页端模型！**

## 功能特性

- **无需 Google API Key**：使用已登录的 Gemini 网页会话的 Cookie 进行认证。
- **三套 API 接口**：OpenAI Chat Completions、OpenAI Responses，以及 Gemini 原生的 `generateContent`。
- **流式输出、工具调用与结构化输出**：三套接口均支持。
- **多模态**：支持图片与文件输入，以及图片生成。
- **内置 Google 搜索**：回答可以使用 Gemini 网页端自带的搜索能力。
- **会话复用**：后续请求会接着同一个 Google 会话继续，而不是重新发送全部历史。
- **多账号**：请求在多个账号之间分发，每个账号可单独配置代理。

## 快速开始

需要 Python 3.13+、[uv](https://docs.astral.sh/uv/)，以及一个可以使用 [Gemini](https://gemini.google.com/) 的 Google 账号。如需在容器中运行，请参见 [Docker](#docker)。

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

把 Cookie（[获取方法](#获取-cookie)）填入 `config/config.yaml`：

```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
```

启动服务，默认监听 `http://localhost:8000`：

```bash
uv run python run.py
```

模型名称由 Google 返回，不同账号看到的并不相同，所以先列出模型，再选一个发起请求：

```bash
curl http://localhost:8000/v1/models
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "MODEL_FROM_THE_LIST", "messages": [{"role": "user", "content": "Hello"}]}'
```

### 获取 Cookie

1. 在无痕窗口中打开 [Gemini](https://gemini.google.com/) 并登录。
2. 打开开发者工具（F12），进入 **Application** → **Storage** → **Cookies**。
3. 复制 `__Secure-1PSID` 和 `__Secure-1PSIDTS` 的值。

更多细节见 [Gemini-API 认证指南](https://github.com/HanaokaYuzu/Gemini-API?tab=readme-ov-file#authentication)。

> [!WARNING]
> 这些 Cookie 可以访问你的 Google 账号，请勿提交到版本控制。

- **如果 Cookie 几小时内就失效，请改用 Firefox。** 基于 Chromium 的浏览器会把会话绑定到设备，从中复制出来的 Cookie 无法续期。
- **请开启 [Gemini Apps 应用活动](https://myactivity.google.com/product/gemini)。** 未开启时 Google 随时可能丢弃会话，多轮对话会因此丢失上下文。

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

或者使用 Docker Compose：

```yaml
services:
  gemini-fastapi:
    image: ghcr.io/nativu5/gemini-fastapi:latest
    ports:
      - "8000:8000"
    volumes:
      # - ./config:/app/config      # 取消注释以使用自定义配置文件
      # - ./certs:/app/certs        # 取消注释以使用自己的证书启用 HTTPS
      - ./data:/app/data
      - ./cache:/app/cache
    environment:
      - CONFIG_SERVER__API_KEY=${API_KEY}
      - CONFIG_GEMINI__CLIENTS__0__ID=client-a
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID=${SECURE_1PSID}
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS=${SECURE_1PSIDTS}
      - GEMINI_COOKIE_PATH=/app/cache # 必须与上面的 cache 卷一致
    restart: on-failure:3
```

> [!IMPORTANT]
> 两个卷都要挂载。`/app/data` 保存会话记录和生成的媒体文件。`/app/cache` 保存服务刷新后的 Cookie，这样重建容器后无需重新登录。

## API 接口

| 接口 | 说明 |
| --- | --- |
| `GET /v1/models` | 已配置账号可以使用的模型。 |
| `POST /v1/chat/completions` | OpenAI Chat Completions：流式输出、工具调用、图片与文件输入、`response_format`。 |
| `POST /v1/responses` | OpenAI Responses：流式输出、工具调用、图片与文件输入、图片生成、`text.format`。 |
| `GET /v1beta/models`、`GET /v1beta/models/{model}` | Gemini 原生的模型列表。 |
| `POST /v1beta/models/{model}:generateContent` | Gemini 原生的生成接口。 |
| `POST /v1beta/models/{model}:streamGenerateContent` | Gemini 原生的流式生成接口。 |
| `GET /health` | 就绪检查：存储状态和每个账号的状态。 |
| `GET /media/{filename}` | 生成的图片和媒体文件。响应中的链接带有签名令牌。 |

设置了 `server.api_key` 时，请通过 `Authorization: Bearer <key>` 传入。`/v1beta` 接口还接受 `x-goog-api-key` 请求头和 `key` 查询参数。未设置 Key 时，任何能访问到服务的人都可以调用 API 和打开媒体链接。

### 兼容性

Gemini 网页端提供的控制项比本服务所模拟的 API 少。合法的请求不会因为某个无法转发的选项而被拒绝。

- **会被忽略的选项。** `temperature`、`top_p`、输出 token 上限、`parallel_tool_calls`、Gemini 的 `generationConfig` 字段和安全设置都会被接受，但不产生任何效果。
- **会被拒绝的输入。** 格式错误的请求，以及服务完全无法解析的内容：Files API 的 `file_id`、`cachedContent` 句柄，以及 `/v1/responses` 上用 `previous_response_id` 或 `conversation` 引用的前几轮对话。请把这几轮放在 `input` 中发送。
- **错误使用被调用 API 自己的格式**：OpenAI 接口为 `{"error": {"message", "type", "param", "code"}}`，`/v1beta` 为 `{"error": {"code", "message", "status"}}`。格式错误的请求在两者上都返回 `400`。
- **生成的视频和音频以链接形式返回**，在所有接口上都放在回复文本中。图片在 OpenAI 接口上同样是链接，其中 `/v1/responses` 还会为每张图片附带一个 `image_generation_call` 项；`/v1beta` 则以 `inlineData` 分段返回图片。
- **结构化输出是在提示词中提出要求**，再由服务在事后校验，因为 Gemini 网页端没有约束解码。
  - `strict: true` 时，回复不符合 Schema 即为错误：返回 `502`，流式响应中则是一个错误事件。
  - `strict: false`（两套 OpenAI 接口的默认值）或 JSON 模式下，回复按原样返回。
  - Gemini 原生的 `responseSchema` 和 `responseJsonSchema` 始终是尽力而为。
  - 返回工具调用的那一轮不做校验。
- **服务无法求值的 Schema 仍会发给模型，但不会被强制执行**，也不会导致请求失败。唯一的例外是不合法的 `responseJsonSchema`，会返回 `400`。包括非法的 JSON Schema、无法翻译成 JSON Schema 的 `responseSchema`、无法解析的 `$ref`（远程引用一律不会去抓取），以及正则关键字超出 `server.schema_validation_budget_seconds` 的情况。
- **带 Schema 的流式响应**会在校验后作为单个分块返回，因为 JSON 只有完整后才能校验。
- **`/v1beta` 对 Google 的 SDK 是部分支持。** 纯文字的对话可以正常使用，流式和非流式都可以。已知的限制：
  - 需要把 API 版本设为 `v1beta`。SDK 默认请求 `/v1`，会得到 `404`。
  - 字段名必须是驼峰形式。Python SDK 会把部分嵌套字段写成下划线形式，因此它发出的带图片或文件的请求会被拒绝并返回 `400`。在聊天中，这包括回复里出现生成图片之后的每一轮，因为 SDK 会把图片带回。JavaScript SDK 不受影响。

## 配置说明

每个配置项的说明都在 [`config/config.yaml`](config/config.yaml) 中。任何配置项都可以用环境变量覆盖：变量名为 `CONFIG_` 加上配置项的路径，层级之间用 `__` 分隔：

```bash
export CONFIG_SERVER__API_KEY="your-secure-api-key"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"
export CONFIG_GEMINI__CHAT_MODE="auto_delete"
```

另有两个变量是直接读取的：`CONFIG_PATH` 指定另一个配置文件，`GEMINI_COOKIE_PATH` 指定刷新后的 Cookie 的缓存目录。

### 账号

`gemini.clients` 下的每一项对应一个 Google 账号。

- `id`：该账号开启的每个会话都会记录这个值。更新 Cookie 时请保持不变，否则已保存的会话将无法继续。
- `proxy`：仅用于该账号的代理 URL，可选。
- `impersonate`：供 [`curl_cffi`](https://github.com/lexiforest/curl_cffi) 使用的浏览器指纹，例如 `chrome`，可选。填入不支持的值会导致服务无法启动。

### 会话模式

`gemini.chat_mode` 决定服务在你的 Google 账号中开启的会话如何处置。

| 模式 | 是否保留在 Google 账号中 | 多轮上下文 |
| --- | --- | --- |
| `normal`（默认） | 保留，直到你手动删除 | 可靠 |
| `temporary` | 从不保存 | 尽力而为 |
| `auto_delete` | 保留，闲置后删除 | 可靠 |

**`temporary`** 使用 Google 的临时会话。Google 为每个账号只保留一个打开的临时会话，并且可能随时将其关闭而不报错，此时回复可能不带之前的上下文。服务只会继续最近的那个临时会话，其他情况下会把完整历史重放到一个新会话中。输入长度上限也更低。长对话请选用其他模式。

**`auto_delete`** 使用普通会话，并在连续 `gemini.auto_delete_ttl` 秒没有请求后（默认 `86400`，最小 `60`），把它从 Google 账号和本地存储中一并删除。

- 只会删除服务在该模式下开启的会话。其他模式下的会话，以及账号里的其他任何内容，都不会被触碰。
- 删除由周期性的清理任务执行，间隔为 TTL 的一半（最短一分钟，最长六小时），因此会话的实际存活时间可能比 TTL 多出一个间隔。
- 删除失败时（例如账号处于未登录状态），会在之后的清理中重试，连续五次失败后放弃并在日志中记录错误。
- 生成的媒体文件按相同的 TTL 过期。

修改 `chat_mode` 之后，位于 `auto_delete` 边界另一侧的已保存会话不会被继续使用，其历史会重放到一个新会话中；在 `auto_delete` 下开启的会话仍会按期删除。

### 访客会话

当某个账号的 Cookie 被拒绝时，该客户端会以访客身份继续运行，而不是直接失败。访客只能回答文本请求：不能上传文件，不能选择模型，也不保留历史。

- 优先使用已登录的客户端。只有在没有已登录的客户端时，访客才会收到请求。
- 如果只剩访客，需要上传的请求会失败并给出明确的提示。
- `/health` 会把访客报告为不健康。`gemini.guest_mode` 决定何时因此返回 HTTP 503：

| `guest_mode` | `/health` 返回 503 的条件 |
| --- | --- |
| `strict` | 任一客户端不健康 |
| `adaptive`（默认） | 所有客户端都不健康 |
| `permissive` | 不会因客户端状态而返回 503 |

存储故障始终返回 503。

### 模型

模型在启动时从各个账号自动发现，无需配置。能用哪些模型、它们叫什么名字，取决于账号的订阅档位。

- 请求可以使用列表中的名称，也可以使用显示名称。
- 匹配是宽松的。版本号会被忽略，并且只要匹配到列表名称中的一个词即可，因此 `gemini-2.5-pro` 和 `pro` 都会解析为 `gemini-pro`。
- 匹配不到任何模型的名称会以 `400` 拒绝。服务不会改用其他模型。

### Gem

> [!WARNING]
> Google 已宣布[下线 Gem](https://support.google.com/gemini/answer/18560919)：个人账号自 2026-11-17 起只读，2026-12-17 移除；Workspace 账号为 2027 年 3 月，教育账号为 2027 年 6 月。此功能仅在账号仍可使用 Gem 时有效。账号失去 Gem 后请把策略改回 `off`：被 Google 拒绝的 Gem 请求最多会重试约 15 秒，每次重试都会重启客户端，之后提示词才放在消息正文中发送，每个请求都是如此。

默认关闭，通过 `gemini.gems.policy` 启用。启用后，`/v1/chat/completions` 或 `/v1/responses` 请求开头的系统提示词会作为一个 Gem（Gemini 中保存的自定义指令）在 Google 账号里保存一次，之后的请求只引用它，而不再把提示词放进消息正文。系统提示词相同的会话共用一个 Gem。

| `policy` | 系统提示词 |
| --- | --- |
| `off`（默认） | 放在消息正文中发送 |
| `fetch_only` | 使用账号里已有的对应 Gem；不会创建 |
| `create_on_demand` | 首次出现时创建一个 Gem |

> [!WARNING]
> `create_on_demand` 会在 Google 账号中创建名为 `fastapi_policy_...` 的 Gem。

- 每个不同的提示词对应一个 Gem，因此只适用于稳定的提示词。如果提示词每次请求都变化（例如包含当前时间），每次都会创建一个新的 Gem。
- 创建受 `create_rate_limit_per_minute`（默认 `4`）和 `managed_gems_max_total`（默认 `200`）限制。超出限制或 Gem 无法使用时，提示词会放在消息正文中发送。
- Gem 会一直保留在账号中，直到你删除。设置 `cleanup.enabled: true` 后，无论使用哪种策略，超过 `cleanup.unused_days`（默认 `7`）未使用的 Gem 会被清理任务删除（启动时运行一次，之后每 6 小时一次），并且每个 Gem 的描述会被更新以记录最近使用时间。
- 访客会话和 `/v1beta` 请求始终把提示词放在消息正文中发送。

### 输入长度

输入超过 `gemini.max_chars_per_request` 的 90%（`temporary` 模式下为 81%）时，会作为 `message.txt` 附件发送给 Gemini。`server.max_request_body_bytes` 是另一项针对 HTTP 请求体的本地保护，与 Gemini 实际接受的长度无关。

### 已移除的配置项

`gemini.models`、`gemini.model_strategy`、`gemini.oversized_context_strategy`、`storage.images_path`，以及 `gemini.gems` 下的 `enabled`、`fetch_on_init`、`include_hidden_on_fetch`、`policies.default_policy` 和 `cleanup.require_managed_marker` 已不存在，即使出现在配置中也会被忽略。生成的媒体文件保存在 `storage.media_path` 下。

## 维护

导出和清理会话存储的脚本见 [`scripts/USAGE.md`](scripts/USAGE.md)。

## 鸣谢

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - 底层 Gemini Web API 客户端
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - 本项目最初基于此仓库，经过深度重构与工程化改进，现已成为独立项目，并增加了多轮会话复用等新特性。在此表示特别感谢。

## 免责声明

本项目与 Google 或 OpenAI 无关，仅供学习和研究使用。本项目使用了逆向工程 API，可能不符合 Google 服务条款。使用风险自负。
