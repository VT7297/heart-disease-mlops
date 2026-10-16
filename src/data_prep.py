"""
Data cleaning for the UCI Heart Disease dataset (combined four sites).

Only fixes values that are clearly wrong in the source data. Imputation,
scaling and encoding happen later inside the sklearn Pipeline so they are
fit on training folds only (no leakage).

Usage:
    python src/data_prep.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease_combined_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_disease_clean.csv"

NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

# A value of 0 is physiologically impossible for these measurements; in this
# dataset it is how some sites (Switzerland, VA) recorded "not measured".
ZERO_MEANS_MISSING = ["chol", "trestbps"]

# Valid category codes per the UCI documentation.
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe."""
    out = df.copy()

    out = out.drop_duplicates().reset_index(drop=True)

    for col in ZERO_MEANS_MISSING:
        out.loc[out[col] == 0, col] = np.nan

    # Codes outside the documented range are treated as missing.
    for col, valid in VALID_CODES.items():
        out.loc[~out[col].isin(valid) & out[col].notna(), col] = np.nan
        out[col] = out[col].astype("Int64")

    if "num" in out.columns:
        out = out.drop(columns=["num"])  # replaced by binary `target`

    out[TARGET] = out[TARGET].astype(int)
    return out


def summarize(raw: pd.DataFrame, cleaned: pd.DataFrame) -> str:
    lines = [
        f"rows: {len(raw)} -> {len(cleaned)} (duplicates removed: {len(raw) - len(raw.drop_duplicates())})",
    ]
    for col in ZERO_MEANS_MISSING:
        lines.append(f"{col}: {(raw[col] == 0).sum()} zero values set to missing")
    lines.append(f"positive class rate: {cleaned[TARGET].mean():.3f}")
    return "\n".join(lines)


def main() -> None:
    raw = load_raw()
    cleaned = clean(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(summarize(raw, cleaned))
    print(f"saved -> {CLEAN_PATH}")


if __name__ == "__main__":
    main()
