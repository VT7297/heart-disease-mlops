"""API tests: endpoints, validation, metrics and failure modes (uses a model trained on synthetic data)."""
import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.features import TARGET, FeatureConfig, build_pipeline

HIGH_RISK = {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
             "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 2.3, "slope": 2}


@pytest.fixture
def model_dir(tmp_path, clean_df):
    # CSV round-trip so dtypes match what train.py sees in production
    path = tmp_path / "clean.csv"
    clean_df.to_csv(path, index=False)
    df = pd.read_csv(path)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(df, df[TARGET])
    out = tmp_path / "models"
    out.mkdir()
    joblib.dump(pipe, out / "model.joblib")
    (out / "model_metadata.json").write_text(json.dumps({
        "version": "test-1", "features": FeatureConfig().columns, "threshold": 0.5,
        "selected_model": "logreg", "test_metrics": {"test_roc_auc": 0.9}}))
    return out


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "model.joblib"))
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_version"] == "test-1"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] == ("disease" if body["prediction"] else "no_disease")
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["model_version"] == "test-1"
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_confidence_matches_probability(client):
    body = client.post("/predict", json=HIGH_RISK).json()
    expected = body["probability_disease"] if body["prediction"] == 1 else 1 - body["probability_disease"]
    assert body["confidence"] == pytest.approx(expected, abs=1e-4)


def test_optional_fields_can_be_omitted(client):
    r = client.post("/predict", json={"age": 57, "sex": 1, "cp": 3})
    assert r.status_code == 200


def test_request_id_is_propagated(client):
    r = client.post("/predict", json=HIGH_RISK, headers={"X-Request-ID": "abc123"})
    assert r.json()["request_id"] == "abc123"


@pytest.mark.parametrize("bad", [
    {**HIGH_RISK, "age": 5},              # out of range
    {**HIGH_RISK, "cp": 7},               # invalid category code
    {**HIGH_RISK, "chol": "high"},        # wrong type
    {k: v for k, v in HIGH_RISK.items() if k != "sex"},  # required field missing
    {**HIGH_RISK, "site": "cleveland"},   # unknown field rejected
])
def test_invalid_input_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["features"] == FeatureConfig().columns
    assert info["version"] == "test-1"


def test_metrics_exposed_and_counting(client):
    client.post("/predict", json=HIGH_RISK)
    client.post("/predict", json={"age": 50, "sex": 0, "cp": 2})
    text = client.get("/metrics").text
    assert "predictions_total" in text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert 'prediction_missing_feature_total{feature="chol"}' in text
    assert "prediction_probability_disease_bucket" in text
    assert 'model_info{model="logreg",version="test-1"} 1.0' in text


def test_missing_model_reports_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    from api.app import app
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=HIGH_RISK).status_code == 503
