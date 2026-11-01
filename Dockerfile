# Python 3.11 image for the test suite.
# Java 17 is not installed here. HermiT (Phase 1) needs it, and this image
# should gain a JDK only once that phase is implemented.
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    MPLBACKEND=Agg

COPY pyproject.toml uv.lock README.md Makefile ./
COPY src ./src
COPY tests ./tests
COPY data ./data

RUN uv sync --frozen --all-groups

CMD ["make", "test"]
