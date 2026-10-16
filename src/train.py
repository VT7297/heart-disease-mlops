"""
Train, tune and compare classifiers with MLflow tracking, then package the
best model (preprocessing + calibrated classifier) for serving.

Usage:
    python src/train.py                 # full run
    python src/train.py --quick         # small grids, used in CI
    mlflow ui --backend-store-uri mlruns  # view results at http://127.0.0.1:5000
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import GROUP_COL, TARGET, FeatureConfig, build_pipeline  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "heart_disease_clean.csv"
MODEL_DIR = ROOT / "models"
EXPERIMENT = "heart-disease-classifier"
SEED = 42
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def model_space(quick: bool) -> dict:
    """Candidate models and their hyper-parameter grids."""
    if quick:
        return {
            "logreg": (LogisticRegression(max_iter=2000), {"model__C": [0.1, 1.0]}),
            "random_forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                              {"model__n_estimators": [100], "model__max_depth": [5]}),
        }
    return {
        "logreg": (
            LogisticRegression(max_iter=2000, solver="liblinear"),
            {"model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0], "model__penalty": ["l1", "l2"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, None],
             "model__min_samples_leaf": [1, 5, 10], "model__max_features": ["sqrt", 0.5]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3], "model__subsample": [0.8, 1.0]},
        ),
    }


def feature_variants(quick: bool) -> dict[str, FeatureConfig]:
    variants = {"impute_only": FeatureConfig(add_missing_indicators=False)}
    if not quick:
        variants["impute_plus_indicators"] = FeatureConfig(add_missing_indicators=True)
    return variants


# --------------------------------------------------------------------------- plots
def save_fig(fig, path: Path) -> Path:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_diagnostics(y, proba, out_dir: Path, title: str) -> list[Path]:
    pred = (proba >= 0.5).astype(int)
    paths = []

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["No disease", "Disease"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion matrix\n{title}")
    paths.append(save_fig(fig, out_dir / "confusion_matrix.png"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=title)
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax.set_title("ROC curve")
    paths.append(save_fig(fig, out_dir / "roc_curve.png"))

    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label=title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed disease rate")
    ax.set_title("Calibration curve")
    ax.legend(fontsize=8)
    paths.append(save_fig(fig, out_dir / "calibration_curve.png"))
    return paths


def plot_importance(pipeline, out_dir: Path) -> Path | None:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        vals, label = model.feature_importances_, "importance"
    elif hasattr(model, "coef_"):
        vals, label = model.coef_.ravel(), "coefficient"
    else:
        return None
    s = pd.Series(vals, index=names).sort_values(key=abs).tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    s.plot.barh(ax=ax, color=["#C8553D" if v > 0 else "#4C9F70" for v in s])
    ax.set_title(f"Top features ({label})")
    return save_fig(fig, out_dir / "feature_importance.png")


# --------------------------------------------------------------------------- evaluation
def evaluate_loso(pipeline, X, y, groups) -> dict:
    """Leave-one-site-out: train on 3 sites, test on the 4th."""
    out = {}
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        site = groups.iloc[te].iloc[0]
        fitted = sklearn.base.clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        proba = fitted.predict_proba(X.iloc[te])[:, 1]
        yt = y.iloc[te]
        out[f"loso_{site}_accuracy"] = accuracy_score(yt, proba >= 0.5)
        if yt.nunique() > 1:
            out[f"loso_{site}_roc_auc"] = roc_auc_score(yt, proba)
    aucs = [v for k, v in out.items() if k.endswith("_roc_auc")]
    accs = [v for k, v in out.items() if k.endswith("_accuracy")]
    out["loso_mean_roc_auc"] = float(np.mean(aucs))
    out["loso_mean_accuracy"] = float(np.mean(accs))
    return out


def run_candidate(name, estimator, grid, variant, cfg, X, y, groups, cv, work_dir):
    run_name = f"{name}__{variant}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"model_family": name, "feature_variant": variant, "stage": "tuning"})
        mlflow.log_params({"model": name, "feature_variant": variant,
                           "add_missing_indicators": cfg.add_missing_indicators,
                           "include_ca_thal": cfg.include_ca_thal,
                           "n_features_in": len(cfg.columns), "cv_folds": cv.get_n_splits(),
                           "grid_size": int(np.prod([len(v) for v in grid.values()]))})

        search = GridSearchCV(build_pipeline(estimator, cfg), grid, cv=cv, scoring="roc_auc",
                              n_jobs=-1, refit=True)
        search.fit(X, y)
        best = search.best_estimator_
        mlflow.log_params({k.replace("model__", "best_"): v for k, v in search.best_params_.items()})

        cv_res = cross_validate(best, X, y, cv=cv, scoring=SCORING)
        metrics = {}
        for m in SCORING:
            metrics[f"cv_{m}_mean"] = float(cv_res[f"test_{m}"].mean())
            metrics[f"cv_{m}_std"] = float(cv_res[f"test_{m}"].std())
        metrics.update(evaluate_loso(best, X, y, groups))
        mlflow.log_metrics(metrics)

        art = work_dir / run_name
        art.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
            art / "grid_search_results.csv", index=False)
        oof = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]
        plot_diagnostics(y, oof, art, run_name)
        plot_importance(best, art)
        mlflow.log_artifacts(str(art), artifact_path="evaluation")

        print(f"  {run_name:42s} cv_auc={metrics['cv_roc_auc_mean']:.3f} "
              f"loso_auc={metrics['loso_mean_roc_auc']:.3f}")
        return {"run_id": run.info.run_id, "run_name": run_name, "model": name, "variant": variant,
                "cfg": cfg, "estimator": best, "best_params": search.best_params_, **metrics}


# --------------------------------------------------------------------------- main
def main(quick: bool, selection_metric: str) -> int:
    df = pd.read_csv(DATA_PATH)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    X, y, groups = df_train, df_train[TARGET], df_train[GROUP_COL]
    X_test, y_test = df_test, df_test[TARGET]
    print(f"train={len(df_train)} test={len(df_test)}  positive rate={y.mean():.3f}")

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    work_dir = ROOT / "reports_tmp"

    results = []
    for variant, cfg in feature_variants(quick).items():
        for name, (est, grid) in model_space(quick).items():
            results.append(run_candidate(name, est, grid, variant, cfg, X, y, groups, cv, work_dir))

    summary = pd.DataFrame([{k: v for k, v in r.items() if k not in ("cfg", "estimator")}
                            for r in results]).sort_values(selection_metric, ascending=False)
    summary_path = ROOT / "report" / "model_comparison.csv"
    summary_path.parent.mkdir(exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print("\n" + summary[["run_name", "cv_roc_auc_mean", "cv_accuracy_mean", "cv_precision_mean",
                          "cv_recall_mean", "loso_mean_roc_auc"]].round(3).to_string(index=False))

    best = max(results, key=lambda r: r[selection_metric])
    print(f"\nselected: {best['run_name']} (by {selection_metric})")

    # ---- final model: calibrated, evaluated once on the untouched test split
    with mlflow.start_run(run_name=f"final__{best['run_name']}") as run:
        mlflow.set_tags({"stage": "final", "selected_from": best["run_id"]})
        calibrated = CalibratedClassifierCV(best["estimator"], method="sigmoid", cv=5)
        calibrated.fit(X, y)
        proba = calibrated.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        test_metrics = {
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_f1": f1_score(y_test, pred),
            "test_roc_auc": roc_auc_score(y_test, proba),
            "test_brier": brier_score_loss(y_test, proba),
            "test_brier_uncalibrated": brier_score_loss(
                y_test, best["estimator"].predict_proba(X_test)[:, 1]),
        }
        mlflow.log_params({"selected_model": best["model"], "feature_variant": best["variant"],
                           "selection_metric": selection_metric, "calibration": "sigmoid",
                           **{k.replace("model__", "best_"): v for k, v in best["best_params"].items()}})
        mlflow.log_metrics(test_metrics)

        art = work_dir / "final"
        art.mkdir(parents=True, exist_ok=True)
        plot_diagnostics(y_test, proba, art, "final (test set)")
        mlflow.log_artifacts(str(art), artifact_path="evaluation")
        mlflow.log_artifact(str(summary_path))

        feature_cols = best["cfg"].columns
        # float64 everywhere so the schema accepts missing values (NaN) at inference time
        example = X_test[feature_cols].head(3).astype("float64")
        schema_sample = X[feature_cols].astype("float64")
        mlflow.sklearn.log_model(
            calibrated, artifact_path="model",
            signature=infer_signature(schema_sample, calibrated.predict_proba(schema_sample)),
            input_example=example,
            registered_model_name=None,
        )

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(calibrated, MODEL_DIR / "model.joblib")
        metadata = {
            "model_name": "heart-disease-classifier",
            "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "selected_model": best["model"],
            "feature_variant": best["variant"],
            "features": feature_cols,
            "threshold": 0.5,
            "calibration": "sigmoid",
            "mlflow_run_id": run.info.run_id,
            "train_rows": len(X), "test_rows": len(X_test),
            "cv_metrics": {k: round(v, 4) for k, v in best.items() if k.startswith("cv_")},
            "loso_metrics": {k: round(v, 4) for k, v in best.items() if k.startswith("loso_")},
            "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        }
        (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(MODEL_DIR / "model_metadata.json"))

    print("\ntest metrics:", {k: round(float(v), 3) for k, v in test_metrics.items()})
    print(f"saved -> {MODEL_DIR / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="small grids (CI)")
    p.add_argument("--select-by", default="loso_mean_roc_auc",
                   choices=["loso_mean_roc_auc", "cv_roc_auc_mean"],
                   help="metric used to pick the final model")
    a = p.parse_args()
    sys.exit(main(a.quick, a.select_by))
