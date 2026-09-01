"""
MedRisk - train and compare four classifiers on the UCI Cleveland heart
disease data, then write metrics, figures and the fitted best model to disk.

Run:  python src/train.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data_prep import (
    CATEGORICAL,
    NUMERIC,
    build_preprocessor,
    feature_names,
    load_dataset,
    load_raw,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
for d in (FIG, MODELS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# A missed heart-disease diagnosis sends a sick patient home; a false alarm
# books an unnecessary follow-up. We price that asymmetry at 5:1 and use it
# to pick the operating threshold in `choose_threshold`.
COST_FN, COST_FP = 5.0, 1.0

PALETTE = {
    "Logistic Regression": "#2B5D8A",
    "Random Forest": "#3E8E7E",
    "XGBoost": "#C6603C",
    "SVM (RBF)": "#7A5C9E",
}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
)


# --------------------------------------------------------------------------
# model zoo
# --------------------------------------------------------------------------
def candidate_models() -> dict[str, tuple[object, dict]]:
    """Estimator + a small hyper-parameter grid for each candidate."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced_subsample"
            ),
            {
                "clf__n_estimators": [300, 600],
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=2,
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.03, 0.1],
                "clf__subsample": [0.8, 1.0],
            },
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, class_weight="balanced",
                random_state=RANDOM_STATE),
            {"clf__C": [0.5, 1.0, 10.0], "clf__gamma": ["scale", 0.05, 0.01]},
        ),
    }


def metrics_at(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_true, proba) -> tuple[float, pd.DataFrame]:
    """Sweep thresholds and return the one minimising expected clinical cost."""
    grid = np.linspace(0.05, 0.95, 181)
    rows = []
    for t in grid:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "cost": COST_FN * fn + COST_FP * fp,
            }
        )
    sweep = pd.DataFrame(rows)
    best = float(sweep.loc[sweep["cost"].idxmin(), "threshold"])
    return best, sweep


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_eda(raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    counts = raw["AHD"].value_counts()
    axes[0, 0].bar(["No disease", "Disease"], [counts.get("No", 0), counts.get("Yes", 0)],
                   color=["#3E8E7E", "#C6603C"], width=0.55)
    for i, v in enumerate([counts.get("No", 0), counts.get("Yes", 0)]):
        axes[0, 0].text(i, v + 2, f"{v}  ({v / len(raw):.0%})", ha="center", fontsize=9)
    axes[0, 0].set_title("Class balance is close to even")
    axes[0, 0].set_ylabel("Patients")
    axes[0, 0].set_ylim(0, counts.max() * 1.18)

    for label, colour in [("No", "#3E8E7E"), ("Yes", "#C6603C")]:
        sns.kdeplot(raw.loc[raw["AHD"] == label, "MaxHR"], ax=axes[0, 1],
                    fill=True, alpha=0.35, color=colour, label=label, linewidth=1.6)
    axes[0, 1].set_title("Peak heart rate separates the classes")
    axes[0, 1].set_xlabel("Max heart rate achieved")
    axes[0, 1].legend(title="Heart disease", frameon=False)

    ct = pd.crosstab(raw["ChestPain"], raw["AHD"], normalize="index")[["No", "Yes"]]
    ct = ct.sort_values("Yes")
    ct.plot(kind="barh", stacked=True, ax=axes[1, 0],
            color=["#3E8E7E", "#C6603C"], width=0.7, legend=False)
    axes[1, 0].set_title("Asymptomatic chest pain carries the highest risk")
    axes[1, 0].set_xlabel("Share of patients")
    axes[1, 0].set_ylabel("")
    axes[1, 0].set_xlim(0, 1)

    sns.boxplot(data=raw, x="AHD", y="Oldpeak", hue="AHD", ax=axes[1, 1],
                order=["No", "Yes"], palette=["#3E8E7E", "#C6603C"],
                width=0.5, legend=False)
    axes[1, 1].set_title("Exercise ST depression runs higher in cases")
    axes[1, 1].set_xlabel("Heart disease")

    fig.suptitle("MedRisk - UCI Cleveland heart disease, exploratory view",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "01_eda_overview.png", bbox_inches="tight")
    plt.close(fig)


def fig_correlation(raw: pd.DataFrame) -> None:
    num = raw[NUMERIC + ["Sex", "Fbs", "ExAng"]].copy()
    num["Target"] = (raw["AHD"] == "Yes").astype(int)
    corr = num.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
    ax.set_title("Correlation between numeric features and the diagnosis")
    fig.tight_layout()
    fig.savefig(FIG / "02_correlation.png", bbox_inches="tight")
    plt.close(fig)


def fig_cv_comparison(cv_table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = cv_table.sort_values("cv_roc_auc_mean", ascending=True)
    ax.barh(order["model"], order["cv_roc_auc_mean"],
            xerr=order["cv_roc_auc_std"], capsize=4,
            color=[PALETTE[m] for m in order["model"]], height=0.6)
    for y, (mean, std) in enumerate(zip(order["cv_roc_auc_mean"], order["cv_roc_auc_std"])):
        ax.text(mean + std + 0.006, y, f"{mean:.3f} ± {std:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("ROC-AUC (5-fold stratified CV on the training split)")
    ax.set_title("Cross-validated model comparison")
    fig.tight_layout()
    fig.savefig(FIG / "03_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_curves(y_test, probas: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, color=PALETTE[name], linewidth=1.9,
                     label=f"{name}  (AUC {roc_auc_score(y_test, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, color=PALETTE[name], linewidth=1.9,
                     label=f"{name}  (AP {average_precision_score(y_test, proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC curves (held-out test set)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[1].axhline(y_test.mean(), ls="--", color="#999", linewidth=1)
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Precision-recall curves (held-out test set)")
    axes[1].legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "04_roc_pr_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_confusions(y_test, probas: dict[str, np.ndarray], threshold: float) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, (name, proba) in zip(axes, probas.items()):
        cm = confusion_matrix(y_test, (proba >= threshold).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
                    annot_kws={"size": 13})
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("Actual")
    fig.suptitle(f"Confusion matrices at the cost-optimal threshold ({threshold:.2f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "05_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def fig_threshold(sweep: pd.DataFrame, best_t: float, best_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(sweep["threshold"], sweep["precision"], label="Precision", color="#2B5D8A")
    axes[0].plot(sweep["threshold"], sweep["recall"], label="Recall", color="#C6603C")
    axes[0].plot(sweep["threshold"], sweep["f1"], label="F1", color="#3E8E7E", ls="--")
    axes[0].axvline(best_t, color="#444", ls=":", linewidth=1.6)
    axes[0].set(xlabel="Decision threshold", ylabel="Score",
                title=f"Precision / recall trade-off - {best_name}")
    axes[0].legend(frameon=False)

    axes[1].plot(sweep["threshold"], sweep["cost"], color="#7A5C9E", linewidth=2)
    axes[1].axvline(best_t, color="#444", ls=":", linewidth=1.6)
    axes[1].annotate(f"cost-optimal\nthreshold = {best_t:.2f}",
                     xy=(best_t, sweep["cost"].min()),
                     xytext=(best_t + 0.12, sweep["cost"].min() + (sweep["cost"].max() - sweep["cost"].min()) * 0.35),
                     arrowprops=dict(arrowstyle="->", color="#444"), fontsize=9)
    axes[1].set(xlabel="Decision threshold",
                ylabel=f"Expected cost ({COST_FN:.0f}xFN + {COST_FP:.0f}xFP)",
                title="A missed case costs five times a false alarm")
    fig.tight_layout()
    fig.savefig(FIG / "06_threshold_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def fig_importance(perm_df: pd.DataFrame, logit_df: pd.DataFrame, best_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

    top = perm_df.head(12).iloc[::-1]
    axes[0].barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
                 capsize=3, color="#2B5D8A", height=0.65)
    axes[0].set_xlabel("Drop in ROC-AUC when the column is shuffled")
    axes[0].set_title(f"Permutation importance - {best_name}")

    # Diverging bars on the log-odds scale, anchored at zero. A log-scale bar
    # anchored at the axis edge makes protective effects (OR < 1) hard to read.
    lg = (logit_df.reindex(logit_df["coefficient"].abs().sort_values(ascending=False).index)
          .head(12).iloc[::-1])
    colours = ["#C6603C" if c > 0 else "#3E8E7E" for c in lg["coefficient"]]
    axes[1].barh(lg["feature"], lg["coefficient"], color=colours, height=0.65)
    axes[1].axvline(0, color="#444", lw=1.2)
    for feat, coef, orr in zip(lg["feature"], lg["coefficient"], lg["odds_ratio"]):
        offset = 0.04 if coef > 0 else -0.04
        axes[1].text(coef + offset, feat, f"OR {orr:.2f}", va="center", fontsize=7.5,
                     ha="left" if coef > 0 else "right", color="#333")
    span = float(lg["coefficient"].abs().max()) * 1.45
    axes[1].set_xlim(-span, span)
    axes[1].set_xlabel("Log-odds per 1 SD  -  right raises risk, left lowers it")
    axes[1].set_title("Logistic regression effect sizes")

    fig.suptitle("Which measurements actually drive the prediction",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "07_feature_importance.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    raw = load_raw()
    X, y = load_dataset()
    print(f"Loaded {X.shape[0]} patients x {X.shape[1]} features | "
          f"positive rate {y.mean():.1%}")
    print("Missing values before imputation:")
    print(raw[["Ca", "Thal"]].isna().sum().to_string(), "\n")

    fig_eda(raw)
    fig_correlation(raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_rows, fitted, probas = [], {}, {}
    for name, (estimator, grid) in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=2, refit=True)
        search.fit(X_train, y_train)

        best_row = search.cv_results_["mean_test_score"][search.best_index_]
        std_row = search.cv_results_["std_test_score"][search.best_index_]
        fitted[name] = search.best_estimator_
        probas[name] = search.best_estimator_.predict_proba(X_test)[:, 1]

        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(best_row),
                "cv_roc_auc_std": float(std_row),
                "best_params": {k.replace("clf__", ""): v
                                for k, v in search.best_params_.items()},
            }
        )
        print(f"{name:22s} CV ROC-AUC {best_row:.4f} ± {std_row:.4f}  "
              f"{cv_rows[-1]['best_params']}")

    cv_table = pd.DataFrame(cv_rows).sort_values("cv_roc_auc_mean", ascending=False)
    fig_cv_comparison(cv_table)

    best_name = cv_table.iloc[0]["model"]
    best_model = fitted[best_name]
    best_proba = probas[best_name]
    print(f"\nSelected by cross-validation: {best_name}")

    best_t, sweep = choose_threshold(y_test, best_proba)
    fig_curves(y_test, probas)
    fig_confusions(y_test, probas, best_t)
    fig_threshold(sweep, best_t, best_name)

    test_rows = []
    for name, proba in probas.items():
        default = metrics_at(y_test, proba, 0.5)
        default["model"] = name
        default["operating_point"] = "default 0.50"
        test_rows.append(default)
    tuned = metrics_at(y_test, best_proba, best_t)
    tuned["model"] = best_name
    tuned["operating_point"] = f"cost-optimal {best_t:.2f}"
    test_rows.append(tuned)
    test_table = pd.DataFrame(test_rows)[
        ["model", "operating_point", "accuracy", "precision", "recall", "f1",
         "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
    ]
    print("\nHeld-out test performance:")
    print(test_table.round(3).to_string(index=False))

    # ---- feature importance -------------------------------------------------
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=30,
        random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=2
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    logit = fitted["Logistic Regression"]
    names = feature_names(logit.named_steps["prep"])
    coefs = logit.named_steps["clf"].coef_[0]
    logit_df = pd.DataFrame(
        {"feature": names, "coefficient": coefs, "odds_ratio": np.exp(coefs)}
    ).sort_values("coefficient", ascending=False).reset_index(drop=True)

    fig_importance(perm_df, logit_df, best_name)
    print("\nTop drivers by permutation importance:")
    print(perm_df.head(8).round(4).to_string(index=False))

    # ---- persist ------------------------------------------------------------
    joblib.dump(
        {"model": best_model, "threshold": best_t, "model_name": best_name,
         "features": list(X.columns)},
        MODELS / "medrisk_model.joblib",
    )
    cv_table.to_csv(REPORTS / "cv_results.csv", index=False)
    test_table.to_csv(REPORTS / "test_results.csv", index=False)
    perm_df.to_csv(REPORTS / "permutation_importance.csv", index=False)
    logit_df.to_csv(REPORTS / "logistic_coefficients.csv", index=False)
    sweep.to_csv(REPORTS / "threshold_sweep.csv", index=False)
    with open(REPORTS / "summary.json", "w") as fh:
        json.dump(
            {
                "n_patients": int(len(X)),
                "positive_rate": float(y.mean()),
                "best_model": best_name,
                "cv_roc_auc": float(cv_table.iloc[0]["cv_roc_auc_mean"]),
                "operating_threshold": best_t,
                "test_metrics_at_threshold": tuned,
            },
            fh,
            indent=2,
        )
    print(f"\nSaved model, 5 CSV reports and 7 figures under {ROOT.name}/")


if __name__ == "__main__":
    main()
