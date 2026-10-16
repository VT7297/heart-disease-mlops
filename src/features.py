"""
Feature definitions and the preprocessing pipeline.

Design decisions (see notebooks/01_eda.ipynb, section 8):
  * `site` is NOT a model feature. Prevalence ranges 36%-93% by site, so the
    model would learn "which hospital" instead of "which patient".
  * `ca` and `thal` are excluded by default: 95-99% missing at three of four
    sites. They can be switched back on for a Cleveland-only comparison.
  * Missing values are imputed inside the pipeline (fit on training folds
    only). Missing-indicator columns are optional and OFF by default because
    missingness is mostly a proxy for site.
"""
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
NOMINAL_FEATURES = ["cp", "restecg", "slope"]
CLEVELAND_ONLY_FEATURES = ["ca", "thal"]

TARGET = "target"
GROUP_COL = "site"


@dataclass(frozen=True)
class FeatureConfig:
    add_missing_indicators: bool = False
    include_ca_thal: bool = False

    @property
    def nominal(self) -> list[str]:
        return NOMINAL_FEATURES + (CLEVELAND_ONLY_FEATURES if self.include_ca_thal else [])

    @property
    def columns(self) -> list[str]:
        return NUMERIC_FEATURES + BINARY_FEATURES + self.nominal


def build_preprocessor(cfg: FeatureConfig | None = None) -> ColumnTransformer:
    """Impute -> scale numerics; impute -> one-hot encode categoricals."""
    cfg = cfg or FeatureConfig()
    ind = cfg.add_missing_indicators

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=ind)),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=ind)),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=ind)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", binary, BINARY_FEATURES),
            ("nom", nominal, cfg.nominal),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(model, cfg: FeatureConfig | None = None) -> Pipeline:
    """Full pipeline: raw patient columns in, prediction out."""
    return Pipeline([("prep", build_preprocessor(cfg)), ("model", model)])
