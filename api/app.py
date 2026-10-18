"""
Heart-disease risk API.

Endpoints
    POST /predict      patient JSON -> prediction + confidence
    GET  /health       liveness
    GET  /ready        readiness (model loaded)
    GET  /model-info   model version, features, evaluation metrics
    GET  /metrics      Prometheus metrics

Run locally:  uvicorn api.app:app --reload
"""
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.schemas import PatientFeatures, PredictionResponse  # noqa: E402
from src.inference import DEFAULT_MODEL_PATH, load_model  # noqa: E402


# --------------------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname,
                 "logger": record.name, "msg": record.getMessage()}
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    log = logging.getLogger("heart_api")
    log.handlers[:] = [handler]
    log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    log.propagate = False
    return log


logger = _setup_logging()

# --------------------------------------------------------------------------- metrics
REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["method", "path"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("predictions_total", "Predictions by outcome", ["label"])
PROBABILITY = Histogram("prediction_probability_disease", "Predicted probability of disease",
                        buckets=[i / 10 for i in range(1, 11)])
MISSING_INPUT = Counter("prediction_missing_feature_total",
                        "Optional fields omitted in requests (imputed)", ["feature"])
MODEL_INFO = Gauge("model_info", "Loaded model (value is always 1)", ["version", "model"])


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    try:
        app.state.model = load_model(path)
        m = app.state.model
        MODEL_INFO.labels(version=m.version, model=m.metadata.get("selected_model", "unknown")).set(1)
        logger.info("model loaded", extra={"fields": {"model_path": path, "version": m.version}})
    except Exception as exc:  # keep serving /health so the failure is visible, /ready reports 503
        app.state.model = None
        logger.error("model failed to load", extra={"fields": {"model_path": path, "error": str(exc)}})
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"  # template path keeps metric cardinality low
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(request.method, path).observe(elapsed)
            logger.info("request", extra={"fields": {
                "request_id": request_id, "method": request.method, "path": request.url.path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


def _model(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return model


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    m = _model(request)
    return {"status": "ready", "model_version": m.version}


@app.get("/model-info")
def model_info(request: Request):
    m = _model(request)
    keys = ("model_name", "version", "selected_model", "feature_variant", "calibration",
            "threshold", "test_metrics", "loso_metrics", "sklearn")
    return {"features": m.features, **{k: m.metadata.get(k) for k in keys}}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    m = _model(request)
    record = patient.model_dump()
    for f in m.features:
        if record.get(f) is None:
            MISSING_INPUT.labels(feature=f).inc()

    result = m.predict([record])[0]
    PREDICTIONS.labels(label=result["label"]).inc()
    PROBABILITY.observe(result["probability_disease"])
    # Patient fields are health data: log the outcome, never the inputs.
    logger.info("prediction", extra={"fields": {
        "request_id": request.state.request_id, "label": result["label"],
        "probability_disease": result["probability_disease"], "model_version": m.version}})
    return {**result, "model_version": m.version, "request_id": request.state.request_id}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
