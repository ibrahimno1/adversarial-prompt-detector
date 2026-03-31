"""
detector.py
-----------
Trains and evaluates classifiers to detect adversarial prompts.

Answers RQ3: Can adversarial prompts be detected from text features alone?

Classifiers:
  - Logistic Regression (L2, primary — interpretable)
  - Random Forest       (100 trees, secondary)

Each trained in three configurations:
  - llama_only   : rows labeled for llama3:8b
  - openai_only  : rows labeled for gpt-4o-mini
  - combined     : all rows (primary deliverable)

Evaluation:
  - Stratified 5-fold cross-validation
  - Metrics per fold: Accuracy, Precision, Recall, F1 (macro), FPR
  - Mean ± std across folds reported
  - Feature importance: LR coefficients + RF importances (top 10)

Outputs:
  outputs/figures/  — confusion matrix, ROC curve, feature importance chart
  outputs/reports/  — metrics JSON, classification report text

Usage:
    python src/detector.py
    python src/detector.py --config combined      # one configuration only
    python src/detector.py --no-show             # suppress plot windows
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CV_FOLDS,
    FEATURES_FILE,
    FIGURES_DIR,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    RANDOM_SEED,
    REPORTS_DIR,
)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

FIGURE_DPI = 150
plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        FIGURE_DPI,
})

CLASSIFIER_COLORS = {"lr": "#4C9BE8", "rf": "#E84C4C"}
CLASSIFIER_LABELS = {"lr": "Logistic Regression", "rf": "Random Forest"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        print(f"ERROR: {FEATURES_FILE} not found. Run feature_extraction.py first.")
        sys.exit(1)
    df = pd.read_csv(FEATURES_FILE)
    print(f"Loaded features: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("lex_", "str_", "sem_", "resp_"))]


FEATURE_GROUPS = {
    "lexical":    "lex_",
    "structural": "str_",
    "semantic":   "sem_",
    "response":   "resp_",
    "all":        ("lex_", "str_", "sem_", "resp_"),
}


def subset_for_config(df: pd.DataFrame, config: str) -> pd.DataFrame:
    if config == "llama_only":
        return df[df["model"] == OLLAMA_MODEL].copy()
    if config == "openai_only":
        return df[df["model"] == OPENAI_MODEL].copy()
    return df.copy()   # combined


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def make_pipelines() -> dict[str, Pipeline]:
    return {
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000,
                random_state=RANDOM_SEED, solver="lbfgs",
            )),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    feature_names: list[str],
) -> dict:
    """
    Stratified k-fold CV. Returns per-fold metrics + aggregates.
    Pipeline is cloned per fold so no state leaks between folds.
    """
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_metrics = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    importances_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Probability for ROC (positive class = adversarial = 1)
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipe.decision_function(X_test)

        # Per-fold metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fold_metrics.append({
            "fold":      fold,
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_macro":  round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "fpr":       round(fpr_val, 4),
            "n_test":    len(y_test),
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        # Feature importances
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            importances_per_fold.append(np.abs(clf.coef_[0]))
        elif hasattr(clf, "feature_importances_"):
            importances_per_fold.append(clf.feature_importances_)

    # Aggregate metrics
    metrics_df = pd.DataFrame(fold_metrics)
    agg = {}
    for col in ["accuracy", "precision", "recall", "f1_macro", "fpr"]:
        agg[f"{col}_mean"] = round(metrics_df[col].mean(), 4)
        agg[f"{col}_std"]  = round(metrics_df[col].std(), 4)

    # ROC-AUC on pooled predictions
    fpr_arr, tpr_arr, _ = roc_curve(all_y_true, all_y_prob)
    roc_auc = round(auc(fpr_arr, tpr_arr), 4)

    # Mean feature importances across folds
    mean_importances = None
    if importances_per_fold:
        mean_importances = np.mean(importances_per_fold, axis=0)

    return {
        "fold_metrics":        fold_metrics,
        "agg":                 agg,
        "roc_auc":             roc_auc,
        "roc_curve":           (fpr_arr.tolist(), tpr_arr.tolist()),
        "all_y_true":          all_y_true,
        "all_y_pred":          all_y_pred,
        "all_y_prob":          all_y_prob,
        "mean_importances":    mean_importances,
        "feature_names":       feature_names,
        "clf_report":          classification_report(
                                   all_y_true, all_y_pred,
                                   target_names=["benign", "adversarial"],
                                   zero_division=0,
                               ),
    }


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_cv_results(clf_key: str, config: str, results: dict) -> None:
    label = CLASSIFIER_LABELS[clf_key]
    agg = results["agg"]
    print(f"\n{'─'*60}")
    print(f"  {label}  |  config: {config}")
    print(f"{'─'*60}")
    print(f"  {'Metric':<18}  {'Mean':>7}  {'±Std':>7}")
    print(f"  {'─'*36}")
    for metric in ["accuracy", "precision", "recall", "f1_macro", "fpr"]:
        print(
            f"  {metric:<18}  "
            f"{agg[f'{metric}_mean']:>7.4f}  "
            f"±{agg[f'{metric}_std']:>6.4f}"
        )
    print(f"  {'ROC-AUC':<18}  {results['roc_auc']:>7.4f}")
    print(f"\n  Classification report (pooled folds):")
    for line in results["clf_report"].splitlines():
        print(f"    {line}")


def print_top_features(clf_key: str, results: dict, top_n: int = 10) -> None:
    imp = results.get("mean_importances")
    if imp is None:
        return
    names = results["feature_names"]
    ranked = sorted(zip(imp, names), reverse=True)[:top_n]
    print(f"\n  Top {top_n} features ({CLASSIFIER_LABELS[clf_key]}):")
    for score, name in ranked:
        bar = "█" * int(score / max(imp) * 20)
        print(f"    {name:<30} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_confusion_matrix(
    results: dict, clf_key: str, config: str, show: bool
) -> None:
    y_true = results["all_y_true"]
    y_pred = results["all_y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, title in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised (row %)"],
    ):
        fmt = "d" if data is cm else ".2f"
        sns.heatmap(
            data, ax=ax, annot=True, fmt=fmt,
            cmap="Blues", linewidths=0.5, linecolor="white",
            xticklabels=["Benign", "Adversarial"],
            yticklabels=["Benign", "Adversarial"],
            cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    fig.suptitle(
        f"Confusion Matrix — {CLASSIFIER_LABELS[clf_key]} ({config})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, f"confusion_{clf_key}_{config}.png", show)


def fig_roc_curve(
    all_results: dict[tuple, dict], config: str, show: bool
) -> None:
    """Plot ROC curves for both classifiers on one figure, per config."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")

    for (clf_key, cfg), results in all_results.items():
        if cfg != config:
            continue
        fpr, tpr = results["roc_curve"]
        roc_auc = results["roc_auc"]
        ax.plot(
            fpr, tpr, lw=2,
            color=CLASSIFIER_COLORS[clf_key],
            label=f"{CLASSIFIER_LABELS[clf_key]} (AUC={roc_auc:.3f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {config}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    _save_fig(fig, f"roc_{config}.png", show)


def fig_feature_importance(
    results: dict, clf_key: str, config: str, show: bool, top_n: int = 10
) -> None:
    imp = results.get("mean_importances")
    if imp is None:
        return
    names = results["feature_names"]
    ranked = sorted(zip(imp, names), reverse=True)[:top_n]
    scores, labels = zip(*ranked)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [
        "#4C9BE8" if n.startswith("lex_") else
        "#F5A623" if n.startswith("str_") else
        "#9B59B6" if n.startswith("resp_") else
        "#7ED321"
        for n in labels
    ]
    ax.barh(range(top_n), scores[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Mean |coefficient| / importance")
    ax.set_title(
        f"Top {top_n} Features — {CLASSIFIER_LABELS[clf_key]} ({config})",
        fontsize=11, fontweight="bold",
    )

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#4C9BE8", label="Lexical"),
        Patch(color="#F5A623", label="Structural"),
        Patch(color="#7ED321", label="Semantic"),
        Patch(color="#9B59B6", label="Response"),
    ], fontsize=8, loc="lower right")

    fig.tight_layout()
    _save_fig(fig, f"feature_importance_{clf_key}_{config}.png", show)


def fig_metrics_comparison(all_results: dict[tuple, dict], show: bool) -> None:
    """Bar chart comparing F1 and ROC-AUC across all clf × config combinations."""
    labels, f1_means, f1_stds, auc_vals = [], [], [], []

    for (clf_key, config), results in all_results.items():
        labels.append(f"{CLASSIFIER_LABELS[clf_key][:2].upper()}\n{config}")
        f1_means.append(results["agg"]["f1_macro_mean"])
        f1_stds.append(results["agg"]["f1_macro_std"])
        auc_vals.append(results["roc_auc"])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.5))
    ax.bar(x - width/2, f1_means, width, yerr=f1_stds, capsize=4,
           label="Macro F1 (mean ± std)", color="#4C9BE8", alpha=0.85)
    ax.bar(x + width/2, auc_vals, width,
           label="ROC-AUC", color="#E84C4C", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.5)
    ax.set_title("Classifier Performance Summary", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, "metrics_comparison.png", show)


def run_ablation(
    df: pd.DataFrame, config: str, show: bool
) -> None:
    """
    Train LR on each feature group separately + all combined.
    Produces a bar chart showing which group drives performance.
    """
    from sklearn.base import clone

    sub = subset_for_config(df, config)
    if len(sub) < CV_FOLDS * 2:
        print(f"  [skip ablation] not enough rows for config '{config}'")
        return

    all_feature_cols = get_feature_cols(sub)
    y = sub["is_adversarial"].values.astype(int)

    group_cols = {
        "Lexical":     [c for c in all_feature_cols if c.startswith("lex_")],
        "Structural":  [c for c in all_feature_cols if c.startswith("str_")],
        "Semantic":    [c for c in all_feature_cols if c.startswith("sem_")],
        "Response":    [c for c in all_feature_cols if c.startswith("resp_")],
        "All":         all_feature_cols,
    }

    pipeline = make_pipelines()["lr"]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    group_names, f1_means, f1_stds, auc_vals = [], [], [], []

    print(f"\n  Ablation ({config}):")
    for group_name, cols in group_cols.items():
        if not cols:
            continue
        X = sub[cols].values.astype(float)
        fold_f1s, all_y_true, all_y_prob = [], [], []

        for train_idx, test_idx in skf.split(X, y):
            pipe = clone(pipeline)
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            y_prob = pipe.predict_proba(X[test_idx])[:, 1]
            fold_f1s.append(f1_score(y[test_idx], y_pred, average="macro", zero_division=0))
            all_y_true.extend(y[test_idx])
            all_y_prob.extend(y_prob)

        mean_f1 = float(np.mean(fold_f1s))
        std_f1  = float(np.std(fold_f1s))
        fpr_arr, tpr_arr, _ = roc_curve(all_y_true, all_y_prob)
        roc_auc = auc(fpr_arr, tpr_arr)

        group_names.append(group_name)
        f1_means.append(mean_f1)
        f1_stds.append(std_f1)
        auc_vals.append(roc_auc)
        print(f"    {group_name:<14} F1={mean_f1:.3f}±{std_f1:.3f}  AUC={roc_auc:.3f}  (n_features={len(cols)})")

    # Plot
    x = np.arange(len(group_names))
    width = 0.35
    colors_map = {
        "Lexical": "#4C9BE8", "Structural": "#F5A623",
        "Semantic": "#7ED321", "Response": "#9B59B6", "All": "#E84C4C",
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width/2, f1_means, width, yerr=f1_stds, capsize=4,
                   color=[colors_map[g] for g in group_names], alpha=0.85,
                   label="Macro F1 (mean ± std)")
    bars2 = ax.bar(x + width/2, auc_vals, width,
                   color=[colors_map[g] for g in group_names], alpha=0.45,
                   label="ROC-AUC", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.5, label="Random baseline")
    ax.set_title(
        f"Feature Group Ablation — Logistic Regression ({config})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, f"ablation_{config}.png", show)


def _save_fig(fig: plt.Figure, filename: str, show: bool) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Persist metrics
# ---------------------------------------------------------------------------

def save_metrics(all_results: dict[tuple, dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for (clf_key, config), results in all_results.items():
        key = f"{clf_key}_{config}"
        out[key] = {
            "classifier":   CLASSIFIER_LABELS[clf_key],
            "config":       config,
            "fold_metrics": results["fold_metrics"],
            "aggregates":   results["agg"],
            "roc_auc":      results["roc_auc"],
        }
    path = REPORTS_DIR / "detector_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"  Saved: {path}")

    # Human-readable text report
    lines = []
    for (clf_key, config), results in all_results.items():
        lines.append(f"\n{'='*60}")
        lines.append(f"{CLASSIFIER_LABELS[clf_key]}  |  {config}")
        lines.append(f"{'='*60}")
        agg = results["agg"]
        for m in ["accuracy", "precision", "recall", "f1_macro", "fpr"]:
            lines.append(f"  {m:<18}  {agg[f'{m}_mean']:.4f} ± {agg[f'{m}_std']:.4f}")
        lines.append(f"  {'roc_auc':<18}  {results['roc_auc']:.4f}")
        lines.append(f"\n{results['clf_report']}")

    txt_path = REPORTS_DIR / "detector_report.txt"
    txt_path.write_text("\n".join(lines))
    print(f"  Saved: {txt_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CONFIGS = ["llama_only", "openai_only", "combined"]


def run_detector(selected_configs: list[str], show: bool) -> None:
    df = load_features()
    feature_cols = get_feature_cols(df)
    n_groups = {
        "lexical": sum(1 for c in feature_cols if c.startswith("lex_")),
        "structural": sum(1 for c in feature_cols if c.startswith("str_")),
        "semantic": sum(1 for c in feature_cols if c.startswith("sem_")),
        "response": sum(1 for c in feature_cols if c.startswith("resp_")),
    }
    print(f"Feature columns ({len(feature_cols)}): {n_groups}")

    # Two classification targets:
    #   is_adversarial — prompt category (adversarial vs benign)  [RQ3 primary]
    #   is_compliant   — model behaviour  (complied vs refused)   [RQ3 secondary]
    targets = {
        "is_adversarial": ("adversarial", "benign"),
        "is_compliant":   ("compliant",   "refusal"),
    }

    all_results: dict[tuple, dict] = {}

    for config in selected_configs:
        sub = subset_for_config(df, config)
        if len(sub) < CV_FOLDS * 2:
            print(f"\n  [skip] '{config}' has only {len(sub)} rows — not enough for {CV_FOLDS}-fold CV")
            continue

        X = sub[feature_cols].values.astype(float)

        for target_col, (pos_label, neg_label) in targets.items():
            if target_col not in sub.columns:
                continue
            y = sub[target_col].values.astype(int)
            pos = y.sum()
            tag = f"{config}_{target_col}"
            print(f"\n{'─'*60}")
            print(f"Config: {config}  |  Target: {target_col}")
            print(f"  n={len(y)}  {pos_label}={pos}  {neg_label}={len(y)-pos}")
            minority_pct = min(pos, len(y) - pos) / len(y)
            if minority_pct < 0.15:
                print(f"  WARNING: minority class is only {minority_pct:.0%} — results may be unstable")

            pipelines = make_pipelines()
            for clf_key, pipeline in pipelines.items():
                print(f"\n  Training {CLASSIFIER_LABELS[clf_key]}...")
                results = run_cv(X, y, pipeline, feature_cols)
                all_results[(clf_key, tag)] = results
                print_cv_results(clf_key, tag, results)
                print_top_features(clf_key, results)

        # Ablation on primary target only (combined config for the paper)
        if config == "combined" and "is_adversarial" in df.columns:
            print(f"\nRunning feature group ablation ({config})...")
            run_ablation(df, config, show)

    if not all_results:
        print("\nNo results to report. Check your data.")
        return

    # --- Figures ---
    print("\nGenerating figures...")
    for (clf_key, tag), results in all_results.items():
        fig_confusion_matrix(results, clf_key, tag, show)
        fig_feature_importance(results, clf_key, tag, show)

    # ROC per config×target combination
    seen_tags = {tag for (_, tag) in all_results}
    for tag in seen_tags:
        fig_roc_curve(all_results, tag, show)

    fig_metrics_comparison(all_results, show)

    # --- Save metrics ---
    print("\nSaving metrics...")
    save_metrics(all_results)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Train adversarial prompt detector")
    parser.add_argument(
        "--config",
        choices=CONFIGS + ["all"],
        default="all",
        help="Which data configuration to run (default: all)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening display windows",
    )
    args = parser.parse_args()

    selected = CONFIGS if args.config == "all" else [args.config]
    run_detector(selected, show=not args.no_show)


if __name__ == "__main__":
    main()
