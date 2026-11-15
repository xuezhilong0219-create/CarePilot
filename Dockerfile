# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# 先只装依赖，源码改动不会让这一层缓存失效
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app --no-create-home app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app data/kb ./data/kb
# 治理模式启动时要在 /app/data 下创建 tickets/gateway/checkpoints 三个 SQLite 文件和 traces 目录。
# 容器以非 root 运行，/app 又是 WORKDIR 创建的（root 所有），所以不挂卷直接 docker run 时这里不显式授权就会 PermissionError。
# 用 compose 挂载 ./data 时，挂载会盖住这里；这一步针对的是不挂卷、或挂命名卷的场景。（未经构建验证：本机没有 Docker）
RUN mkdir -p /app/data/traces && chown -R app:app /app/data

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "carepilot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
