"""Unit tests for src/features.py (preprocessing pipeline)."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import (
    CLEVELAND_ONLY_FEATURES,
    GROUP_COL,
    TARGET,
    FeatureConfig,
    build_pipeline,
    build_preprocessor,
)


def test_default_config_excludes_site_and_cleveland_only_columns():
    cols = FeatureConfig().columns
    assert GROUP_COL not in cols
    assert TARGET not in cols
    assert not set(CLEVELAND_ONLY_FEATURES) & set(cols)


def test_include_ca_thal_adds_columns():
    assert set(CLEVELAND_ONLY_FEATURES) <= set(FeatureConfig(include_ca_thal=True).columns)


def test_preprocessor_output_has_no_missing_values(clean_df):
    X = build_preprocessor().fit_transform(clean_df)
    assert not np.isnan(X).any()
    assert X.shape[0] == len(clean_df)


def test_numeric_features_are_standardised(clean_df):
    prep = build_preprocessor().fit(clean_df)
    X = prep.transform(clean_df)
    names = list(prep.get_feature_names_out())
    age = X[:, names.index("num__age")]
    assert abs(age.mean()) < 1e-6
    assert abs(age.std() - 1) < 1e-6


def test_missing_indicators_add_columns(clean_df):
    n_plain = build_preprocessor(FeatureConfig()).fit_transform(clean_df).shape[1]
    n_ind = build_preprocessor(FeatureConfig(add_missing_indicators=True)).fit_transform(clean_df).shape[1]
    assert n_ind > n_plain


def test_unseen_category_does_not_crash(clean_df):
    prep = build_preprocessor().fit(clean_df)
    row = clean_df.head(1).copy()
    row["restecg"] = 2 if (clean_df["restecg"] != 2).all() else row["restecg"]
    row["cp"] = np.nan
    assert prep.transform(row).shape[0] == 1


def test_pipeline_ignores_extra_columns(clean_df):
    """`site` and other columns are dropped, so callers can pass the full frame."""
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(clean_df, clean_df[TARGET])
    proba = pipe.predict_proba(clean_df.head(5))
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("missing_col", ["chol", "slope", "fbs", "trestbps"])
def test_pipeline_handles_missing_input(clean_df, missing_col):
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(clean_df, clean_df[TARGET])
    row = clean_df.head(1).copy()
    row[missing_col] = np.nan
    assert 0.0 <= pipe.predict_proba(row)[0, 1] <= 1.0
