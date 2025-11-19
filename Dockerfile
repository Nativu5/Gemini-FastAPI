FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

LABEL org.opencontainers.image.description="Web-based Gemini models wrapped into an OpenAI-compatible API."

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-cache --no-dev

COPY app/ app/
COPY config/ config/
COPY run.py .

# Command to run the application
CMD ["uv", "run", "--no-dev", "run.py"]
