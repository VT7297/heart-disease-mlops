"""
Labelled scatter of normal CV AUC vs leave-one-site-out AUC for every tuning run.

Usage:
    python src/plot_comparison.py   # reads report/model_comparison.csv
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "model_comparison.csv"
OUT = ROOT / "report" / "figures" / "07_cv_vs_loso.png"

LABELS = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
          "gradient_boosting": "Gradient Boosting"}
COLORS = {"logreg": "#2E6FAB", "random_forest": "#4C9F70", "gradient_boosting": "#C8553D"}
# label offsets (points) so nearby labels don't collide; default is up-right
OFFSETS = {("gradient_boosting", "impute_only"): (-10, 6, "right"),
           ("random_forest", "impute_only"): (0, 12, "center"),
           ("gradient_boosting", "impute_plus_indicators"): (8, -26, "left")}


def main() -> None:
    df = pd.read_csv(SRC)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model, g in df.groupby("model"):
        g = g.set_index("variant")
        if {"impute_only", "impute_plus_indicators"} <= set(g.index):
            a, b = g.loc["impute_only"], g.loc["impute_plus_indicators"]
            ax.annotate("", xy=(b.cv_roc_auc_mean, b.loso_mean_roc_auc),
                        xytext=(a.cv_roc_auc_mean, a.loso_mean_roc_auc),
                        arrowprops=dict(arrowstyle="->", color=COLORS[model], alpha=0.5, lw=1.5))
        for variant, r in g.iterrows():
            marker = "o" if variant == "impute_only" else "s"
            ax.scatter(r.cv_roc_auc_mean, r.loso_mean_roc_auc, s=90, marker=marker,
                       color=COLORS[model], edgecolor="black", linewidth=0.6, zorder=3)
            dx, dy, ha = OFFSETS.get((model, variant), (8, 4, "left"))
            ax.annotate(f"{LABELS[model]}\n{'impute only' if variant == 'impute_only' else '+ indicators'}",
                        (r.cv_roc_auc_mean, r.loso_mean_roc_auc), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=8)

    best = df.loc[df.loso_mean_roc_auc.idxmax()]
    ax.scatter(best.cv_roc_auc_mean, best.loso_mean_roc_auc, s=320, facecolors="none",
               edgecolors="gold", linewidth=2.5, zorder=4, label="selected (best leave-one-site-out)")

    ax.scatter([], [], marker="o", color="grey", label="impute only")
    ax.scatter([], [], marker="s", color="grey", label="impute + missing indicators")
    ax.set_xlabel("5-fold stratified CV ROC-AUC (same-distribution)")
    ax.set_ylabel("Leave-one-site-out ROC-AUC (unseen hospital)")
    ax.set_title("Missing indicators raise CV AUC but don't improve cross-site generalisation")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.margins(x=0.25, y=0.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
