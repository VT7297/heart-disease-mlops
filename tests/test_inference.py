"""Tests for src/inference.py (shared by the API and CLI)."""
import math

import pytest

from src.inference import load_model, to_frame


def test_to_frame_orders_columns_and_fills_missing():
    df = to_frame([{"b": 2, "a": 1}, {"a": 3}], ["a", "b", "c"])
    assert list(df.columns) == ["a", "b", "c"]
    assert df.dtypes.unique().tolist() == ["float64"]
    assert math.isnan(df.loc[1, "b"]) and df["c"].isna().all()


def test_to_frame_ignores_unknown_keys():
    assert list(to_frame([{"a": 1, "zzz": 9}], ["a"]).columns) == ["a"]


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")
