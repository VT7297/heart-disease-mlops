# syntax=docker/dockerfile:1
# Heart-disease risk API. Build from the repo root:
#   podman build --format docker -f docker/Dockerfile -t heart-api:1.0 .
#   docker build -f docker/Dockerfile -t heart-api:1.0 .
ARG PYTHON_VERSION=3.11

# ---------- builder: install dependencies into an isolated venv ----------
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install -r /tmp/requirements-serve.txt

# ---------- runtime: slim image, no build tools, non-root ----------
FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="UCI heart disease risk classifier (FastAPI)" \
      org.opencontainers.image.source="https://github.com/VT7297/heart-disease-mlops"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
# Application code and model are root-owned and read-only for the app user.
COPY src/__init__.py src/inference.py ./src/
COPY api/ ./api/
COPY models/model.joblib models/model_metadata.json ./models/

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]

# One worker per container; scale with replicas (keeps Prometheus metrics per-pod).
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
