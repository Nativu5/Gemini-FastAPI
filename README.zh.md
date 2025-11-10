# Gemini-FastAPI

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ [English](README.md) | 中文 ]

基于 [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 的 Gemini Web API 封装，提供兼容 OpenAI 的接口。

**✅ 无需 API Key，也能免费调用 Gemini Web 模型！**

## 功能

- **🔐 无需 Google API Key**：直接使用浏览器 Cookie 即可访问 Gemini Web 模型。
- **🔍 内置 Google 搜索**：享受 Gemini Web 带来的实时搜索能力。
- **💾 对话持久化**：使用 LMDB 保存多轮对话历史。
- **🖼️ 多模态支持**：支持文本、图片、文件等多种输入形式。
- **🔧 灵活配置**：YAML 配置，可由环境变量覆盖。

## 快速开始

**若需 Docker 部署，请查看下方 [Docker 部署](#docker-部署) 章节。**

### 前置条件

- Python 3.12
- 拥有可使用 Gemini 的 Google 账号
- 从 Gemini 网页复制 `secure_1psid` 与 `secure_1psidts` Cookie

### 安装

#### 使用 uv（推荐）

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

编辑 `config/config.yaml` 并至少填写一组凭证：

```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
      proxy: null # 可选代理 URL（null/空字符串表示直连）
```

> 代理提示：
>
> - 不填写 `proxy` 或设置为 `null`/空字符串，即可保持直连。
> - 每个 client 都可以设置不同的代理。

> [!NOTE]
> 更多字段说明请参考下方 [配置](#配置) 章节。

### 运行服务

```bash
# 使用 uv
uv run python run.py

# 直接使用 Python
python run.py
```

默认监听 `http://localhost:8000`。

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
  -e CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080" \
  -e GEMINI_COOKIE_PATH="/app/cache" \
  ghcr.io/nativu5/gemini-fastapi
```

> [!TIP]
> 仅在需要代理时才设置 `CONFIG_GEMINI__CLIENTS__N__PROXY`，留空则保持直连。
>
> `GEMINI_COOKIE_PATH` 指向容器内保存刷新后 Cookie 的目录。绑定宿主机路径（例如 `-v $(pwd)/cache:/app/cache`）即可在重建/重新创建容器时保留 Cookie，避免重复登录。

### 使用 Docker Compose

创建 `docker-compose.yml`：

```yaml
services:
  gemini-fastapi:
    image: ghcr.io/nativu5/gemini-fastapi:latest
    ports:
      - "8000:8000"
    volumes:
      # - ./config:/app/config  # 如需自定义配置文件可取消注释
      # - ./certs:/app/certs        # 如需启用 HTTPS 可取消注释
      - ./data:/app/data
      - ./cache:/app/cache
    environment:
      - CONFIG_SERVER__HOST=0.0.0.0
      - CONFIG_SERVER__PORT=8000
      - CONFIG_SERVER__API_KEY=${API_KEY}
      - CONFIG_GEMINI__CLIENTS__0__ID=client-a
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID=${SECURE_1PSID}
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS=${SECURE_1PSIDTS}
      - CONFIG_GEMINI__CLIENTS__0__PROXY=socks5://127.0.0.1:1080 # 每个 client 可选代理
      - GEMINI_COOKIE_PATH=/app/cache # 需与上方 cache 卷保持一致
    restart: on-failure:3 # 避免过度重试
```

然后执行：

```bash
docker compose up -d
```

> [!IMPORTANT]
> 请务必挂载 `/app/data` 以在容器重启后保留对话数据。
> 同时挂载 `/app/cache`，即可让刷新后的 Cookie（含轮换的 1PSIDTS）在重建/重启容器后继续生效，无需再次登录。

## 配置

应用会读取 `config/config.yaml`。各字段说明可查看该文件中的注释。

### 环境变量覆盖

> [!TIP]
> 在 Docker/生产环境中，通过环境变量覆写可以与配置文件解耦敏感信息。

所有配置都可以使用 `CONFIG_` 前缀的环境变量覆写，使用双下划线 `__` 表示嵌套键，例如：

```bash
# 覆盖服务端配置
export CONFIG_SERVER__API_KEY="your-secure-api-key"

# 覆盖第一组 Gemini 凭证
export CONFIG_GEMINI__CLIENTS__0__ID="client-a"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts"
# 可选：为某个 client 设置代理
export CONFIG_GEMINI__CLIENTS__0__PROXY="socks5://127.0.0.1:1080"

# 覆盖会话存储空间大小
export CONFIG_STORAGE__MAX_SIZE=268435456  # 256 MB
```

### Client ID 与会话复用

对话会按照 client 的 `id` 进行存储。更新 Cookie 时保持 ID 一致，即可继续复用历史会话。

### Gemini 凭证

> [!WARNING]
> Cookie 等同于你的 Google 登录态，请妥善保管，不要提交到版本库。

操作步骤：

1. 在无痕/私密模式下打开 [Gemini](https://gemini.google.com/) 并登录。
2. 打开浏览器开发者工具（F12）。
3. 前往 **Application → Storage → Cookies**。
4. 复制以下键值：
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

> [!TIP]
> 更详细的图文步骤可参考 [HanaokaYuzu/Gemini-API 认证指南](https://github.com/HanaokaYuzu/Gemini-API?tab=readme-ov-file#authentication)。

## 致谢

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) —— 底层 Gemini Web API 客户端。
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) —— 项目早期基于此仓库，经过大量工程化改造后演化为现在的版本，感谢启发。

## 免责声明

本项目与 Google 或 OpenAI 无任何关联，仅供学习和研究。项目基于逆向接口实现，可能不符合 Google 的服务条款，请自行承担风险。
