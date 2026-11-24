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
      cookies: "__Secure-1PSID=...; __Secure-1PSIDTS=...; ..."
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

1. 在无痕窗口中打开 [Gemini](https://gemini.google.com/)，只登录要给服务使用的那一个账号。同时登录多个账号时，服务使用第一个。
2. 打开开发者工具（F12），进入 **Network**（网络）面板，刷新页面。
3. 选中一个发往 `gemini.google.com` 的请求，例如 `app` 或 `batchexecute`。
4. 在 **Request Headers**（请求标头）中复制 `cookie` 的完整值。
5. 关闭无痕窗口，但不要退出登录，这样浏览器就不会再更新你复制的 Cookie。

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
  -e CONFIG_GEMINI__CLIENTS__0__COOKIES='your-cookie-header' \
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
      - CONFIG_GEMINI__CLIENTS__0__COOKIES=${GEMINI_COOKIES}
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

设置了 `server.api_key` 时，请通过 `Authorization: Bearer <key>` 传入。Gemini 原生接口还接受 `x-goog-api-key` 请求头和 `key` 查询参数。发往 `/v1/models…` 的请求如果用这两种方式之一传 Key，会按 `/v1beta/models…` 由原生接口处理，因为 Google 的 SDK 在没有另行设置时就把请求发到这里。未设置 Key 时，任何能访问到服务的人都可以调用 API 和打开媒体链接。

### 兼容性

Gemini 网页端提供的控制项比本服务所模拟的 API 少。合法的请求不会因为某个无法转发的选项而被拒绝。

- **会被忽略的选项。** `temperature`、`top_p`、输出 token 上限、`parallel_tool_calls`、Gemini 的 `generationConfig` 字段和安全设置都会被接受，但不产生任何效果。
- **会被拒绝的输入。** 格式错误的请求，以及服务完全无法解析的内容：Files API 的 `file_id`、`cachedContent` 句柄，以及 `/v1/responses` 上用 `previous_response_id` 或 `conversation` 引用的前几轮对话。请把这几轮放在 `input` 中发送。
- **错误使用被调用 API 自己的格式**：OpenAI 接口为 `{"error": {"message", "type", "param", "code"}}`，Gemini 原生接口为 `{"error": {"code", "message", "status"}}`。格式错误的请求在两者上都返回 `400`。
- **生成的视频和音频以链接形式返回**，在所有接口上都放在回复文本中。图片在 OpenAI 接口上同样是链接，其中 `/v1/responses` 还会为每张图片附带一个 `image_generation_call` 项；Gemini 原生接口则以 `inlineData` 分段返回图片。
- **结构化输出是在提示词中提出要求**，再由服务在事后校验，因为 Gemini 网页端没有约束解码。
  - `strict: true` 时，回复不符合 Schema 即为错误：返回 `502`，流式响应中则是一个错误事件。
  - `strict: false`（两套 OpenAI 接口的默认值）或 JSON 模式下，回复按原样返回。
  - Gemini 原生的 `responseSchema` 和 `responseJsonSchema` 始终是尽力而为。
  - 返回工具调用的那一轮不按响应 Schema 校验。
- **工具调用同样是在提示词中提出要求**，在回复完整后校验。
  - 调用了请求没有提供的工具即为错误：返回 `502`，流式响应中则是一个错误事件。`tool_choice: "none"` 或 Gemini 原生 `NONE` 模式下的任何调用都属于这种情况。
  - 函数声明了 `strict: true`，或在 Gemini 原生 `ANY`、`VALIDATED` 模式下，参数不符合 Schema 的调用也是错误。其他调用即使参数不符合 Schema 也照样返回，并在日志中记录警告。省略 `strict` 时不做强制校验，`/v1/responses` 会把它回报为 `false`。
  - 参数保留其 JSON 类型，但只有原文符合参数 Schema 时保留原文：字符串参数的 `10001` 仍是 `"10001"`。
- **服务无法求值的 Schema 仍会发给模型，但不会被强制执行**，也不会导致请求失败。包括非法的 JSON Schema、无法翻译成 JSON Schema 的 `responseSchema`、无法解析的 `$ref`（远程引用一律不会去抓取），以及正则关键字超出 `server.schema_validation_budget_seconds` 的情况。唯一的例外是不合法的 `responseJsonSchema`，会返回 `400`。
- **带 Schema 的流式响应**会在校验后作为单个分块返回，因为 JSON 只有完整后才能校验。
- **Google 的 SDK 使用默认设置即可**，文字和图片的对话（流式和非流式）以及 `responseSchema` 和函数工具都可以正常使用。字段名接受驼峰和下划线两种形式，与 Google 一致。

## 配置说明

配置项都在 [`config/config.yaml`](config/config.yaml) 中，任何一项都可以用环境变量覆盖。[CONFIG.zh.md](CONFIG.zh.md) 说明了这两种方式，以及账号、会话模式、访客与失效的 Cookie、模型、Gem 和输入长度。

## 维护

导出和清理会话存储的脚本见 [`scripts/USAGE.md`](scripts/USAGE.md)。

## 鸣谢

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - 底层 Gemini Web API 客户端
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - 本项目最初基于此仓库，经过深度重构与工程化改进，现已成为独立项目，并增加了多轮会话复用等新特性。在此表示特别感谢。

## 免责声明

本项目与 Google 或 OpenAI 无关，仅供学习和研究使用。本项目使用了逆向工程 API，可能不符合 Google 服务条款。使用风险自负。
