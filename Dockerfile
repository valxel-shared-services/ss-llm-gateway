# syntax=docker/dockerfile:1
# uv builds a locked virtualenv; the runtime image has Python and the venv only, as a non-root user.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src src
COPY README.md ./
RUN uv sync --locked --no-dev --no-editable

# The same Python as the build stage: the virtualenv is bound to the interpreter that made it.
FROM python:3.15-slim-bookworm
# No pip, setuptools or ensurepip at run time: nothing installs here, and their vendored packages
# (urllib3, msgpack) only add vulnerabilities.
RUN python -m pip uninstall -y pip setuptools wheel 2>/dev/null; rm -rf /usr/local/lib/python3.13/ensurepip \
    && useradd --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["llm-gateway"]
