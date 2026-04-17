FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Add the agent user early to keep it consistent
RUN adduser --disabled-password --gecos "" agent

# Set production environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/home/agent/src

USER agent
WORKDIR /home/agent

# Install third-party dependencies first for better caching.
# Do not install the local project yet, because setuptools expects `src/`
# to exist when building the package.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked --no-dev --no-install-project

# Copy source code
COPY src src

# Install the local project after source is present.
RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked --no-dev

# Production entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "engine.a2a.server"]
CMD ["--host", "0.0.0.0", "--port", "9009"]

EXPOSE 9009
