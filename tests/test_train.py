"""Tests for src/train.py: evaluation helpers and an end-to-end smoke run."""
import json

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

import src.train as train
from src.features import FeatureConfig, build_pipeline


def test_loso_reports_every_site(clean_df):
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    res = train.evaluate_loso(pipe, clean_df, clean_df["target"], clean_df["site"])
    for site in clean_df["site"].unique():
        assert f"loso_{site}_accuracy" in res
    assert 0.0 <= res["loso_mean_roc_auc"] <= 1.0


def test_quick_model_space_has_at_least_two_models():
    assert len(train.model_space(quick=True)) >= 2
    assert len(train.model_space(quick=False)) >= 2


def test_full_run_compares_indicator_variants():
    assert set(train.feature_variants(quick=False)) == {"impute_only", "impute_plus_indicators"}


@pytest.fixture
def isolated_project(tmp_path, clean_df, monkeypatch):
    """Point train.py at a temp directory so the smoke test never touches real files."""
    data = tmp_path / "data" / "processed" / "heart_disease_clean.csv"
    data.parent.mkdir(parents=True)
    clean_df.to_csv(data, index=False)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "DATA_PATH", data)
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path / "models")
    return tmp_path


def test_end_to_end_quick_training(isolated_project, clean_df):
    assert train.main(quick=True, selection_metric="loso_mean_roc_auc") == 0

    model_path = isolated_project / "models" / "model.joblib"
    meta_path = isolated_project / "models" / "model_metadata.json"
    assert model_path.exists() and meta_path.exists()
    assert (isolated_project / "report" / "model_comparison.csv").exists()
    assert (isolated_project / "mlruns").exists()

    meta = json.loads(meta_path.read_text())
    assert meta["features"] == FeatureConfig().columns
    assert 0.0 <= meta["test_metrics"]["test_roc_auc"] <= 1.0

    model = joblib.load(model_path)
    proba = model.predict_proba(clean_df[meta["features"]].head(3))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    # synthetic data has a real signal, so the model should beat chance
    assert meta["test_metrics"]["test_roc_auc"] > 0.6
