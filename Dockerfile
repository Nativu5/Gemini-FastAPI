FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv <<EOF
set -eux
uv sync --locked --no-install-project --no-default-groups
EOF

FROM python:3.13-slim-trixie AS runtime

LABEL org.opencontainers.image.title="Gemini-FastAPI" \
      org.opencontainers.image.description="Web-based Gemini models wrapped into an OpenAI-compatible API."

# tally ignore=hadolint/DL3002
USER root

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,id=apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,id=aptlib,sharing=locked \
    <<EOF
set -eux
apt-get update
apt-get install -y --no-install-recommends ca-certificates tini
mkdir -p /app/cache /app/data
EOF

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /app/.venv .venv/
COPY app/ app/
COPY config/ config/
COPY run.py run.py

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=10m \
    --retries=3 \
    CMD ["python", "-c", "import urllib.request; from app.utils import g_config; urllib.request.urlopen(f'http://127.0.0.1:{g_config.server.port}/health', timeout=5).close()"]

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python", "run.py"]
