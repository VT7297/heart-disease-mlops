"""
Model loading and prediction, shared by the API (api/app.py) and the CLI (src/predict.py).

The saved model is a full sklearn Pipeline (impute -> scale/encode -> calibrated
classifier), so it takes raw patient fields and handles missing values itself.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"
LABELS = {0: "no_disease", 1: "disease"}


@dataclass
class LoadedModel:
    pipeline: Any
    features: list[str]
    threshold: float
    version: str
    metadata: dict

    def predict(self, records: list[dict]) -> list[dict]:
        frame = to_frame(records, self.features)
        proba = self.pipeline.predict_proba(frame)[:, 1]
        results = []
        for p in proba:
            pred = int(p >= self.threshold)
            results.append({
                "prediction": pred,
                "label": LABELS[pred],
                "confidence": round(float(p if pred == 1 else 1 - p), 4),
                "probability_disease": round(float(p), 4),
            })
        return results


def to_frame(records: list[dict], features: list[str]) -> pd.DataFrame:
    """Build a float64 frame in training column order; absent fields become NaN."""
    rows = [{f: r.get(f) for f in features} for r in records]
    return pd.DataFrame(rows, columns=features).astype("float64")


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> LoadedModel:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found at {model_path} - run `python src/train.py` first")
    meta_path = model_path.with_name("model_metadata.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pipeline = joblib.load(model_path)
    features = metadata.get("features") or list(getattr(pipeline, "feature_names_in_", []))
    if not features:
        raise ValueError("cannot determine model input features (metadata missing)")
    return LoadedModel(
        pipeline=pipeline,
        features=features,
        threshold=float(metadata.get("threshold", 0.5)),
        version=str(metadata.get("version", "unknown")),
        metadata=metadata,
    )
