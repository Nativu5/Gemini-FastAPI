# Gemini-FastAPI

[![Python 3.13](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ [English](README.md) | 中文 ]

将 Gemini 网页端模型封装为兼容 OpenAI API 的 API Server。基于 [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 实现。

**无需 API Key，免费通过 API 调用 Gemini 网页端模型！**

## 功能特性

- **无需 Google API Key**：只需网页 Cookie，即可免费通过 API 调用 Gemini 模型。
- **内置 Google 搜索**：API 已内置 Gemini 网页端的搜索能力，模型响应更加准确。
- **会话持久化**：基于 LMDB 存储，支持多轮对话历史记录。
- **多模态支持**：可处理文本、图片及文件上传。
- **多账户负载均衡**：支持多账户分发请求，可为每个账户单独配置代理。

## 快速开始

**如需 Docker 部署，请参见下方 [Docker 部署](#docker-部署) 部分。**

### 前置条件

- Python >= 3.13
- 拥有网页版 Gemini 访问权限的 Google 账号 (开启 **[Gemini Apps 应用活动](https://myactivity.google.com/product/gemini)** 以获得最佳会话持久化体验)
- 从 Gemini 网页获取的 `secure_1psid` 和 `secure_1psidts` Cookie

### 安装

#### 使用 uv (推荐)

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

#### 使用 pip

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
pip install -e .
```

### 配置

编辑 `config/config.yaml` 并提供至少一组凭证：

```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
      proxy: null # 可选代理 URL (null/空值则保持直连)
      impersonate: null # 可选浏览器指纹模拟 (null 则使用库的默认值)
```

> [!NOTE]
> 详细说明请参见下方 [配置](#配置说明) 部分。

### 启动服务

```bash
# 使用 uv
uv run python run.py

# 直接用 Python
python run.py
```

服务默认启动在 `http://localhost:8000`。

## API 接口

本服务器提供了一系列接口，重点支持 OpenAI 兼容协议。

### OpenAI 兼容接口

这些接口使用 OpenAI 兼容的传输格式，并将请求转换后发送给 Gemini 网页端。兼容范围有意
覆盖 Gemini 网页端客户端实际暴露的控制项：对于客户端无法转发但请求模型已识别的有效选项，
服务会正常接受并忽略，同时在调试日志中记录选项名称，避免其阻止其他可表示的请求内容执行。

- **`GET /v1/models`**: 列出所有可用的 Gemini 模型。
- **`POST /v1/chat/completions`**: 统一聊天对话接口。
  - **流式传输**: 设置 `stream: true` 即可实时接收增量响应 (Stream Delta)。
  - **多模态支持**: 支持在消息中包含文本、图片以及文件上传。
  - **工具调用**: 支持通过 `tools` 参数进行函数调用 (Function Calling)。
  - **结构化输出**: 支持 `response_format` 的全部模式。`json_schema` 会在服务器端按所给
    Schema 验证；`json_object`（JSON 模式）只要求回复能解析为 JSON；`text` 为默认值，不作限制。

### 高级接口

- **`POST /v1/responses`**: 支持当前的 `text.format` 结构化输出（`text`、`json_object`
  与 `json_schema`）、外部或内联文件输入、图片生成及工具调用。由于本项目没有实现
  OpenAI Files API，因此会拒绝 `file_id` 引用。

Schema 的强制程度与 OpenAI 自身的承诺保持一致：`strict: true` 时，回复不符合 Schema 即视为
错误；`strict: false` 或 JSON 模式仅承诺尽力而为，因此不符合的回复会以文本形式返回而不会让
请求失败。返回工具调用的轮次不受 Schema 约束——Schema 只约束最终答案。

`strict` 在两个 OpenAI 接口上均默认为 `false`，与 OpenAI 自身一致，因此同一个 Schema 在任一
接口上的行为相同。该开关在本项目中比上游更关键：Gemini 网页端没有受约束解码，Schema 只能
通过提示词表达，因此一旦启用严格模式，模型的细微偏差就会让调用方彻底失去这次回复。出于同样
的原因，Gemini 原生接口的 `generationConfig.responseSchema` / `responseJsonSchema` 始终按尽力
而为处理——它没有可供关闭的 `strict` 开关。

只有模型自身的失败才会被追究。本服务无法求值的 Schema（不是合法的 JSON Schema，或 `$ref`
无法解析）仍会展示给模型，但不会用于判定回复；正则关键字耗尽
`server.schema_validation_budget_seconds` 时，回复将保持未校验状态。即使在 `strict` 下，这些
情况都不会让请求失败——它们是本服务的能力缺口，而非模型的违规。

由于 JSON 文档只有在完整后才能校验，带结构化要求的流式响应会在校验完成后作为单个分块返回，
而非逐步下发。Gemini 原生接口的 `responseMimeType: application/json` 同样适用。

Gemini 网页端未暴露的生成控制项，例如 `temperature`、`top_p`、最大输出 Token 数、
`parallel_tool_calls`、Gemini `generationConfig` 字段及安全设置，仍会被接受，但无法影响上游生成。
服务会忽略这些已识别选项并写入调试日志。只有格式错误的输入，或本服务完全无法解析的内容
（例如无法解析的 Files API ID、`cachedContent` 句柄）才会被拒绝——静默丢弃这类内容会改变
模型实际回答的问题。未知字段遵循 Pydantic 的默认忽略行为。

在 Gemini 接口上，`generationConfig.responseSchema` 属于 OpenAPI 3.0 子集（大写类型名、
`nullable`），使用前会转换为 JSON Schema；`responseJsonSchema` 本身就是 JSON Schema，按其
标准验证。`toolConfig.functionCallingConfig.allowedFunctionNames` 仅在真正生效的 `ANY` 与
`VALIDATED` 模式下才会收窄工具列表。

### 实用工具接口

- **`GET /health`**: 就绪状态接口。对话存储不可用时始终返回 HTTP 503。客户端故障则遵循
  `gemini.guest_mode` 健康策略；默认的 `adaptive` 策略仅在所有 Gemini 客户端均不健康时返回 503。
- **`GET /media/{filename}`**: 用于分发生成的媒体内容的内部接口。需要有效的 Token（API 返回的图片 URL 中已自动包含该 Token）。

## Docker 部署

### 直接运行

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
> 需要代理时可添加 `CONFIG_GEMINI__CLIENTS__0__PROXY`；省略该变量将保持直连。
>
> `GEMINI_COOKIE_PATH` 指定容器内保存刷新后 Cookie 的目录。将其挂载（例如 `-v $(pwd)/cache:/app/cache`）可以在容器重建或重启后保留这些 Cookie，避免频繁重新认证。

### 使用 Docker Compose

创建 `docker-compose.yml` 文件：

```yaml
services:
  gemini-fastapi:
    image: ghcr.io/nativu5/gemini-fastapi:latest
    ports:
      - "8000:8000"
    volumes:
      # - ./config:/app/config  # Uncomment to use a custom config file
      # - ./certs:/app/certs    # Uncomment to enable HTTPS with your certs
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

然后运行：

```bash
docker compose up -d
```

> [!IMPORTANT]
> 请务必挂载 `/app/data` 卷以保证对话数据在容器重启后持久化。
> 同时挂载 `/app/cache`（或与 `GEMINI_COOKIE_PATH` 对应的目录）以保存刷新后的 Cookie，这样在容器重建/重启后无需频繁重新认证。

## 配置说明

服务器读取 `config/config.yaml` 配置文件。

各项配置说明请参见 [`config/config.yaml`](https://github.com/Nativu5/Gemini-FastAPI/blob/main/config/config.yaml) 文件中的注释。

### 环境变量覆盖

> [!TIP]
> 该功能适用于 Docker 部署和生产环境，可将敏感信息与配置文件分离。

你可以通过带有 `CONFIG_` 前缀的环境变量覆盖任意配置项，嵌套键用双下划线（`__`）分隔，例如：

```bash
# 覆盖服务器设置
export CONFIG_SERVER__API_KEY="your-secure-api-key"

# 覆盖 Client 0 的用户凭据
export CONFIG_GEMINI__CLIENTS__0__ID="client-a"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts"

# 覆盖 Client 0 的代理设置
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"

# 覆盖 Client 0 的浏览器指纹模拟
export CONFIG_GEMINI__CLIENTS__0__IMPERSONATE="chrome"


# 覆盖对话存储大小限制
export CONFIG_STORAGE__MAX_SIZE=268435456  # 256 MB

# 覆盖本地 HTTP 请求体资源保护上限（设为 0 可禁用）
export CONFIG_SERVER__MAX_REQUEST_BODY_BYTES=268435456

# 覆盖 JSON Schema 正则求值预算（单位：秒）
export CONFIG_SERVER__SCHEMA_VALIDATION_BUDGET_SECONDS=1.0
```

`max_request_body_bytes` 仅是封装层用于保护内存和本地资源的可配置上限，并非 OpenAI 或
Gemini API 的兼容性限制，也不代表 Gemini 网页端的容量。通过本地检查后，请求最终是否可被
接受仍由 Gemini 网页端决定。

### 客户端 ID 与会话重用

会话在保存时会绑定创建它的客户端 ID。请在配置中保持这些 `id` 值稳定，
这样在更新 Cookie 列表时依然可以复用旧会话。

### Gemini 凭据

> [!WARNING]
> 请妥善保管这些凭据，切勿提交到版本控制。这些 Cookie 可访问你的 Google 账号。

<!-- 使用注释将凭据警告分隔为独立的引用块（markdownlint MD028）。 -->

> [!WARNING]
> **会话稳定性**：如果 Cookie 频繁过期，请使用 Firefox 提取 Cookie。较新的 Chromium 内核浏览器版本使用“设备绑定会话凭据”（Device Bound Session Credentials），虽然提高了安全性，但会使 Cookie 仅在几小时内有效且无法续期。

使用 Gemini-FastAPI 需提取 Gemini 会话 Cookie：

1. 在无痕/隐私窗口打开 [Gemini](https://gemini.google.com/) 并登录
2. 打开开发者工具（F12）
3. 进入 **Application** → **Storage** → **Cookies**
4. 查找并复制以下值：
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

> [!IMPORTANT]
> **请开启 [Gemini Apps 应用活动](https://myactivity.google.com/product/gemini)** 以确保稳定的会话持久化。
>
> 虽然在没有开启该设置的情况下，连续的聊天过程可能暂时正常，但任何瞬时错误、TLS 会话重启或服务器重启都可能导致 Google 端过期的会话元数据。如果该设置被禁用，模型将 **完全丢失多轮对话的上下文**，导致即使本地 LMDB 中存有历史记录，旧对话也将无法继续。

### 代理设置

每个客户端条目可以配置不同的代理，从而规避速率限制。省略 `proxy` 字段或将其设置为 `null` 或空字符串以保持直连。

### 浏览器指纹模拟

每个客户端可以通过 `impersonate` 参数设置 `curl_cffi` 使用的 TLS/HTTP 指纹。

- 设置为 `null`（默认）则使用库的默认值。
- 可设为 [`curl_cffi` 的 `BrowserTypeLiteral`](https://github.com/lexiforest/curl_cffi) 支持的任意值。
- 启动时会校验该值；无效值会阻止服务启动。

```yaml
gemini:
  clients:
    - id: "client-a"
      impersonate: "chrome" # 使用 Chrome 指纹
    - id: "client-b"
      impersonate: null # 使用库默认值
```

### 会话模式

你可以控制请求使用普通的 Google 会话，还是 Google 的临时会话模式：

```yaml
gemini:
  chat_mode: "normal" # "normal"（普通）或 "temporary"（临时）
  guest_mode: "adaptive" # "strict"（严格）、"adaptive"（自适应）或 "permissive"（宽松）
  max_chars_per_request: 1000000
```

设置为 `temporary` 时，对话不会保存到 Google 账号中。只要 Google 尚未关闭该临时窗口，
临时会话仍然可以继续对话，因此会话重用与会话存储的行为与普通模式完全一致。

当已存储的会话无法再被延续时——例如切换了 `chat_mode`，或 Google 已关闭该临时窗口——
服务会回退到将完整对话历史重放到一个全新的会话中，从而重建上下文，而不是丢失上下文。

每个账号在 Google 侧最多只保留一个处于开启状态的临时窗口：一旦创建新的会话，上一个临时
会话就会被关闭。因此只有最近一次开启的临时会话仍可继续对话。服务会按客户端记录该会话，
并且**只**重用它；任何更早的临时会话都会以完整历史重放到全新会话中。这里没有需要调节的
超时时间——该规则直接依据 Google 的实际行为，而不是靠猜测过期时长。

该记录刻意只保存在内存中，因此只要客户端会话被重新初始化——例如因闲置触发 `auto_close`、
服务重启或重新部署——它同样会被清空。发生上述情况后，服务无法再确认任何窗口仍然有效，
所有已存储的临时会话都会改为重放，而不是重用。

同一规则也适用于以访客身份运行的客户端，且与 `chat_mode` 无关。当所有 Cookie 分组都失败，
或已认证的会话因 Cookie 过期而在使用过程中被拒绝时，客户端仍会以无账号状态继续处理纯文本
请求——而访客会话的对话不会写入任何历史记录，因此其行为与临时会话完全一致。每条已存储的
对话都会记录其所属的会话窗口，因此已认证状态下开启的会话不会被重放到访客会话中，访客状态
下开启的会话也不会在 Cookie 恢复后被重放；无论哪个方向的跨越，都会回退为在全新会话中重放
完整历史。

访客会话会维持服务不中断，而不是让服务失效；相关请求会降级处理，而不是直接失败：

- 客户端池优先选择已认证的客户端，被降级为访客的客户端只有在没有任何已认证客户端可用时才会
  承接流量。
- 需要上传文件的请求——包括附件，以及需要以 `message.txt` 形式发送的超长输入——会被路由到
  已认证的客户端，即使已存储的会话原本会将其绑定到访客客户端。若没有可用的已认证客户端，
  请求会返回明确的错误说明，而不是 Google 的 `Permission denied`。
- Google 不为访客提供模型选择，因此所请求的模型会被替换为访客被允许使用的默认模型，并记录
  一条警告。`/v1/models` 只会公布客户端确实能够提供服务的模型。

`/health` 会将访客客户端报告为不健康——请刷新其 Cookie 以恢复完整能力。
`guest_mode` 设置决定这些不健康客户端如何影响就绪状态响应：

- `strict`（严格）：任一客户端不健康时返回 HTTP 503。
- `adaptive`（自适应，默认）：仅当所有客户端均不健康时返回 HTTP 503；若仍有健康客户端，
  则只记录警告并保持就绪。
- `permissive`（宽松）：只记录警告，不改变就绪状态，即使所有客户端均不健康也是如此。

三种模式都会记录不健康客户端。无论 `guest_mode` 为何，对话存储不可用时仍会返回 HTTP 503。

除此之外，以上规则**仅**在临时模式下生效：由已认证客户端开启的普通会话在用户手动删除之前
会一直由 Google 保留，因此其元数据可以长期重用，并且不受重启影响。

> [!WARNING]
> Google 可能在任意时刻、且不作任何提示地关闭临时会话窗口，包括在对话进行到一半时。
> 此时模型可能直接返回不含既有上下文的回复，而不会抛出错误，因此上下文丢失可能是静默的。
> 只要服务能够识别出该会话已失效，就会将完整历史重放到新会话中，但这种识别并非总能成功。
> 对于较长或对上下文较敏感的对话，建议使用 `normal`；`temporary` 的连续性应视为尽力而为。

由于临时会话可接受的负载更小，服务会在标准安全余量的基础上再收紧 10%，
因此有效输入上限为 `max_chars_per_request` 的 81%（而非 90%）。
两种模式下，超出有效上限的输入仍会以 `message.txt` 附件的形式发送。

环境变量等价写法：

```bash
export CONFIG_GEMINI__CHAT_MODE="temporary"
export CONFIG_GEMINI__GUEST_MODE="adaptive"
```

### 模型

模型在启动时从 Google 动态获取，无需任何配置。每个客户端会读取自己账号可用的模型，并在运行时
构建对应的请求头，因此只要 Google 提供了新发布的模型，服务就能立即使用。`GET /v1/models`
列出的是运行中的客户端确实能够提供服务的模型。

请求可以使用模型的规范名称（当前为 `gemini-pro`、`gemini-flash`、`gemini-flash-lite`）、别名或
显示名称（`pro`、`Flash Lite`），也可以使用其内部 id；所有写法都会解析到同一份会话历史。模型
名称来自 Google 而非本项目，可通过 `GET /v1/models` 查看你自己账号实际可用的列表。服务只提供
动态获取到的模型——若没有任何客户端提供该模型，则返回 `400`，而不会悄悄改用其他模型作答。

> [!NOTE]
> `models` 与 `model_strategy` 配置项已移除，底层库中基于静态 `Model` 枚举的名称查找也已删除。
> 它们的作用是在库尚未跟上新模型发布时手写模型请求头，而动态获取已让这一需求不再存在——如今
> 硬编码的请求头反而有把请求固定在过时模型上的风险。若配置文件或环境变量中仍保留这两个键，
> 将被直接忽略。

## 鸣谢

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - 底层 Gemini Web API 客户端
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - 本项目最初基于此仓库，经过深度重构与工程化改进，现已成为独立项目，并增加了多轮会话复用等新特性。在此表示特别感谢。

## 免责声明

本项目与 Google 或 OpenAI 无关，仅供学习和研究使用。本项目使用了逆向工程 API，可能不符合 Google 服务条款。使用风险自负。
