FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Add the agent user early to keep it consistent
RUN adduser --disabled-password --gecos "" agent

# Set production environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/home/agent/src

USER agent
WORKDIR /home/agent

# Install dependencies first for better caching
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked --no-dev

# Copy source code
COPY src src

# Production entrypoint
ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]

EXPOSE 9009