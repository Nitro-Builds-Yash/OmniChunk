# syntax=docker/dockerfile:1.7

# ── Rust extension: built once, installed as a wheel ──
FROM rust:1-slim AS rust-builder
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
 && pip3 install --break-system-packages "maturin>=1.7,<2" && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY rust/cleave_core ./
RUN maturin build --release -i python3 -o /wheels

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first so code edits do not re-download torch and Docling.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY cleave ./cleave
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
# abi3 wheel: built against the builder's Python, loads on any CPython ≥ 3.12.
COPY --from=rust-builder /wheels /wheels
RUN uv pip install --python /app/.venv/bin/python /wheels/*.whl && rm -rf /wheels

RUN useradd --create-home --uid 1000 cleave && mkdir -p /app/data && chown -R cleave /app/data
USER cleave
ENV PATH="/app/.venv/bin:$PATH" HOME=/home/cleave

EXPOSE 8321
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fsS http://127.0.0.1:8321/health/live || exit 1
# Inside a container the app must listen on 0.0.0.0, so `cleave serve`
# insists on CLEAVE_API_KEY being set (compose loads it from .env).
CMD ["cleave", "serve", "--host", "0.0.0.0", "--port", "8321"]
