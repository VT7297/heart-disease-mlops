"""Unit tests for src/data_prep.py (data cleaning)."""
import numpy as np
import pandas as pd

from src.data_prep import CATEGORICAL_COLS, TARGET, VALID_CODES, ZERO_MEANS_MISSING, clean


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_duplicates_removed(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    assert len(clean(dup)) == len(raw_df)


def test_zero_cholesterol_becomes_missing(raw_df):
    n_zero = (raw_df["chol"] == 0).sum()
    assert n_zero > 0
    out = clean(raw_df)
    for col in ZERO_MEANS_MISSING:
        assert (out[col] == 0).sum() == 0
    assert out["chol"].isna().sum() >= n_zero


def test_invalid_category_codes_become_missing(raw_df):
    raw_df.loc[0, "cp"] = 9          # not a valid chest-pain code
    raw_df.loc[1, "thal"] = 5        # not a valid thal code
    out = clean(raw_df)
    for col, valid in VALID_CODES.items():
        present = out[col].dropna().astype(int)
        assert set(present.unique()) <= valid, col


def test_categoricals_are_nullable_ints(clean_df):
    for col in CATEGORICAL_COLS:
        assert str(clean_df[col].dtype) == "Int64"


def test_target_is_binary_and_num_dropped(clean_df):
    assert "num" not in clean_df.columns
    assert set(clean_df[TARGET].unique()) <= {0, 1}
    assert clean_df[TARGET].dtype == np.int64


def test_missing_values_are_preserved_not_imputed(raw_df):
    """Imputation belongs in the training pipeline, not in cleaning (avoids leakage)."""
    out = clean(raw_df)
    assert out["slope"].isna().sum() == raw_df["slope"].isna().sum()
