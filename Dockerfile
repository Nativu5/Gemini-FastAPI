FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

LABEL org.opencontainers.image.title="Gemini-FastAPI" \
      org.opencontainers.image.description="Web-based Gemini models wrapped into an OpenAI-compatible API."

USER root

WORKDIR /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml uv.lock ./
RUN uv sync --no-cache --frozen --no-install-project --no-dev

COPY app/ app/
COPY config/ config/
COPY run.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uv", "run", "--no-dev", "run.py"]
