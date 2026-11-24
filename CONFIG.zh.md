# 配置说明

[ [English](CONFIG.md) | 中文 ]

每个配置项及其默认值都列在 [`config/config.yaml`](config/config.yaml) 中。本页说明注释里讲不清的部分。

## 配置文件与环境变量

服务读取 `config/config.yaml`，或 `CONFIG_PATH` 指定的文件。名为 `CONFIG_` 加上配置项路径的环境变量（层级之间用 `__` 分隔）会覆盖该配置项：

```bash
export CONFIG_SERVER__API_KEY="your-secure-api-key"
export CONFIG_GEMINI__CLIENTS__0__COOKIES='your-cookie-header'
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"
export CONFIG_GEMINI__CHAT_MODE="auto_delete"
```

- `CONFIG_GEMINI__CLIENTS__<n>__<key>` 修改配置文件中的第 `n` 个账号；`n` 是下一个空闲序号时，则新增一个账号。序号再往后会导致服务无法启动。
- `GEMINI_COOKIE_PATH` 指定服务刷新后的 Cookie 的缓存目录。
- 服务不接受的配置值会让它在启动时退出，日志会指出是哪一项。

## 账号

`gemini.clients` 下的每一项对应一个 Google 账号。

- `id`：该账号开启的每个会话都会记录这个值。更新 Cookie 时请保持不变，否则已保存的会话将无法继续。
- `cookies`：已登录的浏览器发往 Gemini 的请求中的 Cookie 请求头（[获取方法](README.zh.md#获取-cookie)）。不填则该客户端以访客身份运行。
- `proxy`：仅用于该账号的代理 URL，可选。
- `impersonate`：供 [`curl_cffi`](https://github.com/lexiforest/curl_cffi) 使用的浏览器指纹，例如 `chrome`，可选。

## 会话模式

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

## 访客与失效的 Cookie

没有配置 `cookies` 的客户端是访客。访客只能回答文本请求：不能上传文件，不能选择模型，也不保留历史。

- 优先使用有账号的客户端。只有在它们都不可用时，访客才会收到请求；如果只剩访客，需要上传的请求会返回 `503`。
- 配置了 Cookie 的客户端永远不会以访客身份运行。当 Google 不再接受它的 Cookie 时（无论是启动时，还是之后因过期或续期失败），错误会记入日志，该客户端停止服务，它正在处理的请求返回 `503`。之后每个本会用到它的请求都会先尝试重启它；只要 Google 仍拒绝它的 Cookie，请求就会交给其他客户端，没有可用客户端时返回 `503`。新的 Cookie 需要重启服务才会生效。
- 启动时如果没有任何客户端能启动，服务会退出。
- 客户端未运行，或配置了 Cookie 但 Google 不再接受时，`/health` 会把它报告为不健康。访客在运行时是健康的。`gemini.guest_mode` 决定何时因不健康的客户端返回 HTTP 503：

| `guest_mode` | `/health` 返回 503 的条件 |
| --- | --- |
| `strict` | 任一客户端不健康 |
| `adaptive`（默认） | 所有客户端都不健康 |
| `permissive` | 不会因客户端状态而返回 503 |

存储故障始终返回 503。

## 模型

模型在启动时从各个账号自动发现，无需配置。能用哪些模型、它们叫什么名字，取决于账号的订阅档位。

- 请求可以使用列表中的名称，也可以使用显示名称。
- 匹配是宽松的。版本号会被忽略，并且只要匹配到列表名称中的一个词即可，因此 `gemini-2.5-pro` 和 `pro` 都会解析为 `gemini-pro`。
- 匹配不到任何模型的名称会以 `400` 拒绝。服务不会改用其他模型。

## Gem

> [!WARNING]
> Google 已宣布[下线 Gem](https://support.google.com/gemini/answer/18560919)：个人账号自 2026-11-17 起只读，2026-12-17 移除；Workspace 账号为 2027 年 3 月，教育账号为 2027 年 6 月。服务不再创建 Gem，此功能只会在账号还保留 Gem 时，使用旧版本服务创建的 Gem。账号失去 Gem 后请把策略改回 `off`：被 Google 拒绝的 Gem 请求最多会重试约 15 秒，每次重试都会重启客户端，之后提示词才放在消息正文中发送，每个请求都是如此。

默认关闭。设置 `gemini.gems.policy: fetch_only` 后，`/v1/chat/completions` 或 `/v1/responses` 请求开头的系统提示词，会改为引用旧版本服务为它保存在 Google 账号里的 Gem（Gemini 中保存的自定义指令，名为 `fastapi_policy_...`），而不再放进消息正文。没有对应 Gem 或 Gem 无法使用时，提示词放在消息正文中发送。

- Gem 会一直保留在账号中，直到你删除。再设置 `cleanup.enabled: true` 后，超过 `cleanup.unused_days`（默认 `7`）未使用的 Gem 会被清理任务删除（启动时运行一次，之后每 6 小时一次），并且每个 Gem 的描述会被更新以记录最近使用时间。
- 访客会话和 Gemini 原生接口的请求始终把提示词放在消息正文中发送。

## 输入长度

输入超过 `gemini.max_chars_per_request` 的 90%（`temporary` 模式下为 81%）时，会作为 `message.txt` 附件发送给 Gemini。`server.max_request_body_bytes` 是另一项针对 HTTP 请求体的本地保护，与 Gemini 实际接受的长度无关。
