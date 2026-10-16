"""Shared fixtures: small synthetic datasets with the same schema as the UCI data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SITES = ["cleveland", "hungary", "switzerland", "va_long_beach"]


def make_raw(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Raw-format frame (as written by download_data.py) with a learnable signal."""
    rng = np.random.default_rng(seed)
    cp = rng.integers(1, 5, n)
    exang = rng.integers(0, 2, n)
    thalach = rng.normal(140, 20, n).round()
    risk = 1.2 * (cp == 4) + 1.0 * exang - 0.03 * (thalach - 140) + rng.normal(0, 0.8, n)
    num = np.where(risk > 0.8, rng.integers(1, 5, n), 0)

    df = pd.DataFrame({
        "age": rng.integers(29, 78, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": cp.astype(float),
        "trestbps": rng.normal(132, 17, n).round(),
        "chol": rng.normal(240, 45, n).round(),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": thalach,
        "exang": exang.astype(float),
        "oldpeak": rng.normal(1.0, 1.0, n).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "num": num,
        "site": np.resize(SITES, n),
    })
    df["target"] = (df["num"] > 0).astype(int)
    # realistic quirks
    df.loc[df.site == "switzerland", "chol"] = 0
    df.loc[rng.random(n) < 0.3, "slope"] = np.nan
    df.loc[rng.random(n) < 0.6, ["ca", "thal"]] = np.nan
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    from src.data_prep import clean
    return clean(raw_df)
