"""
behavioral_analysis.py
----------------------
Answers RQ1 and RQ2:
  RQ1: Do compliance rates differ across prompt categories?
  RQ2: Do compliance rates differ between single-turn and multi-turn prompts?
       Do they differ between models?

Loads data/labeled/labels.csv + data/prompts/prompts.json, produces:
  - Grouped bar chart: compliance rates by category x model
  - Side-by-side: single-turn vs multi-turn compliance
  - Heatmap: prompt category x label (per model)
  - Chi-squared tests for statistical significance
  - Summary table printed to console + saved to outputs/reports/

Usage:
    python src/behavioral_analysis.py
    python src/behavioral_analysis.py --no-show      # save figures, don't open windows
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
from scipy.stats import chi2_contingency, fisher_exact
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CATEGORIES,
    FIGURES_DIR,
    LABELS,
    LABELS_FILE,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    PROMPTS_FILE,
    RANDOM_SEED,
    REPORTS_DIR,
)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

PALETTE = {
    "refusal":            "#4C9BE8",
    "partial_compliance": "#F5A623",
    "full_compliance":    "#E84C4C",
}
FIGURE_DPI = 150
LABEL_DISPLAY = {
    "refusal":            "Refusal",
    "partial_compliance": "Partial",
    "full_compliance":    "Full Compliance",
}
CAT_DISPLAY = {
    "direct_adversarial":    "Direct",
    "indirect_adversarial":  "Indirect",
    "multiturn_adversarial": "Multi-turn",
    "benign_control":        "Benign",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       FIGURE_DPI,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Merge labels.csv with prompts.json.
    Returns a DataFrame with one row per labeled (prompt, model) pair.
    """
    if not LABELS_FILE.exists():
        print(f"ERROR: Labels file not found: {LABELS_FILE}")
        print("Run labeler.py first.")
        sys.exit(1)

    labels_df = pd.read_csv(LABELS_FILE)

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    prompts_df = pd.DataFrame(prompts)[["id", "is_multi_turn", "subcategory"]]
    prompts_df = prompts_df.rename(columns={"id": "prompt_id"})

    df = labels_df.merge(prompts_df, on="prompt_id", how="left")

    # Normalise category display names
    df["category_display"] = df["category"].map(CAT_DISPLAY).fillna(df["category"])
    df["label_display"] = df["label"].map(LABEL_DISPLAY).fillna(df["label"])
    df["turn_type"] = df["is_multi_turn"].map({True: "Multi-turn", False: "Single-turn"})

    # Ordered categoricals for consistent plot ordering
    df["category"] = pd.Categorical(df["category"], categories=CATEGORIES, ordered=True)
    df["label"] = pd.Categorical(df["label"], categories=LABELS, ordered=True)

    return df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def compliance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame of compliance rates (%) per (category, model, label).
    """
    rows = []
    for model in df["model"].unique():
        for cat in CATEGORIES:
            subset = df[(df["model"] == model) & (df["category"] == cat)]
            n = len(subset)
            if n == 0:
                continue
            for label in LABELS:
                count = (subset["label"] == label).sum()
                rows.append({
                    "model":    model,
                    "category": cat,
                    "label":    label,
                    "count":    int(count),
                    "n":        n,
                    "rate_pct": round(count / n * 100, 1),
                })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, rate_df: pd.DataFrame) -> str:
    """Print and return a formatted summary table."""
    lines = []

    lines.append("\n" + "=" * 70)
    lines.append("BEHAVIORAL ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total labeled records : {len(df)}")
    lines.append(f"Unique prompts        : {df['prompt_id'].nunique()}")
    lines.append(f"Models                : {', '.join(df['model'].unique())}")

    for model in df["model"].unique():
        lines.append(f"\n{'─'*60}")
        lines.append(f"Model: {model}")
        lines.append(f"{'─'*60}")

        model_rate = rate_df[rate_df["model"] == model]
        pivot = model_rate.pivot(index="category", columns="label", values="rate_pct").fillna(0)
        pivot = pivot.reindex(index=CATEGORIES, columns=LABELS, fill_value=0)
        pivot.index = [CAT_DISPLAY.get(c, c) for c in pivot.index]
        pivot.columns = [LABEL_DISPLAY.get(l, l) for l in pivot.columns]

        # Add n column
        n_map = model_rate.drop_duplicates("category").set_index("category")["n"]
        pivot.insert(0, "n", [n_map.get(c, 0) for c in CATEGORIES])

        lines.append(tabulate(pivot, headers="keys", tablefmt="rounded_outline",
                               floatfmt=".1f"))

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_chi_squared(df: pd.DataFrame) -> list[dict]:
    """
    Chi-squared test: for each model, test whether label distribution
    is independent of prompt category.

    If any expected cell count < 5, falls back to collapsing
    partial + full_compliance into 'non_refusal' and notes this.
    """
    results = []

    for model in df["model"].unique():
        subset = df[df["model"] == model]

        # Build contingency table: rows=category, cols=label
        ct = pd.crosstab(subset["category"], subset["label"])
        ct = ct.reindex(index=CATEGORIES, columns=LABELS, fill_value=0)

        chi2, p, dof, expected = chi2_contingency(ct.values)
        min_expected = expected.min()
        note = ""

        if min_expected < 5:
            # Collapse to binary: refusal vs non-refusal
            ct_binary = ct.copy()
            ct_binary["non_refusal"] = ct_binary.get("partial_compliance", 0) + ct_binary.get("full_compliance", 0)
            ct_binary = ct_binary[["refusal", "non_refusal"]]
            chi2, p, dof, expected = chi2_contingency(ct_binary.values)
            note = f"collapsed to refusal/non-refusal (min expected cell = {min_expected:.1f})"

        results.append({
            "model":        model,
            "test":         "chi2 (category vs label)",
            "chi2":         round(chi2, 3),
            "p_value":      round(p, 4),
            "dof":          dof,
            "significant":  p < 0.05,
            "note":         note,
        })

    # Also test: same category, different models (if >1 model)
    models = df["model"].unique()
    if len(models) == 2:
        for cat in CATEGORIES:
            subset = df[df["category"] == cat]
            if subset.empty:
                continue
            ct = pd.crosstab(subset["model"], subset["label"])
            ct = ct.reindex(columns=LABELS, fill_value=0)

            if ct.shape[0] < 2 or ct.values.sum() == 0:
                continue

            # Drop all-zero columns (labels not present for this category)
            ct = ct.loc[:, (ct != 0).any(axis=0)]
            if ct.shape[1] < 2:
                results.append({
                    "model": "both",
                    "test":  f"(model vs label | {CAT_DISPLAY.get(cat, cat)})",
                    "chi2":  None, "p_value": None, "dof": None,
                    "significant": False,
                    "note": "skipped — only one label observed in this category",
                })
                continue

            if ct.shape == (2, 2):
                odds, p = fisher_exact(ct.values)
                results.append({
                    "model":        "both",
                    "test":         f"fisher (model vs label | {CAT_DISPLAY.get(cat, cat)})",
                    "chi2":         round(odds, 3),
                    "p_value":      round(p, 4),
                    "dof":          1,
                    "significant":  p < 0.05,
                    "note":         "Fisher's exact (2x2)",
                })
            else:
                try:
                    chi2, p, dof, _ = chi2_contingency(ct.values)
                    results.append({
                        "model":        "both",
                        "test":         f"chi2 (model vs label | {CAT_DISPLAY.get(cat, cat)})",
                        "chi2":         round(chi2, 3),
                        "p_value":      round(p, 4),
                        "dof":          dof,
                        "significant":  p < 0.05,
                        "note":         "",
                    })
                except ValueError as e:
                    results.append({
                        "model": "both",
                        "test":  f"chi2 (model vs label | {CAT_DISPLAY.get(cat, cat)})",
                        "chi2":  None, "p_value": None, "dof": None,
                        "significant": False,
                        "note": f"skipped — {e}",
                    })

    return results


def print_stats(stat_results: list[dict]) -> str:
    lines = ["\n" + "─" * 60, "STATISTICAL TESTS", "─" * 60]
    headers = ["Model", "Test", "Stat", "p-value", "Sig?", "Note"]
    rows = [
        [
            r["model"], r["test"],
            f"{r['chi2']:.3f}" if r["chi2"] is not None else "—",
            f"{r['p_value']:.4f}" if r["p_value"] is not None else "—",
            "YES *" if r["significant"] else "no",
            r["note"],
        ]
        for r in stat_results
    ]
    lines.append(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_compliance_by_category(df: pd.DataFrame, show: bool) -> None:
    """Grouped bar chart: compliance rate by category, one group per model."""
    models = list(df["model"].unique())
    n_models = len(models)
    cats = CATEGORIES
    n_cats = len(cats)
    bar_width = 0.25
    x = np.arange(n_cats)

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = df[df["model"] == model]
        for j, label in enumerate(LABELS):
            rates = []
            for cat in cats:
                sub = model_df[model_df["category"] == cat]
                rate = (sub["label"] == label).sum() / len(sub) * 100 if len(sub) else 0
                rates.append(rate)
            offset = (j - 1) * bar_width
            bars = ax.bar(x + offset, rates, bar_width,
                          label=LABEL_DISPLAY[label],
                          color=PALETTE[label], alpha=0.85, edgecolor="white")
            for bar, rate in zip(bars, rates):
                if rate > 3:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                            f"{rate:.0f}%", ha="center", va="bottom", fontsize=7)

        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([CAT_DISPLAY.get(c, c) for c in cats], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(0, 110)
        ax.set_ylabel("Compliance rate (%)" if ax == axes[0] else "")
        ax.legend(fontsize=8)

    fig.suptitle("Compliance Rates by Prompt Category", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "compliance_by_category.png", show)


def fig_single_vs_multiturn(df: pd.DataFrame, show: bool) -> None:
    """Side-by-side bar chart: single-turn vs multi-turn compliance."""
    turn_types = ["Single-turn", "Multi-turn"]
    models = list(df["model"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    bar_width = 0.25
    x = np.arange(len(turn_types))

    for ax, model in zip(axes, models):
        model_df = df[df["model"] == model]
        for j, label in enumerate(LABELS):
            rates = []
            for tt in turn_types:
                sub = model_df[model_df["turn_type"] == tt]
                rate = (sub["label"] == label).sum() / len(sub) * 100 if len(sub) else 0
                rates.append(rate)
            offset = (j - 1) * bar_width
            bars = ax.bar(x + offset, rates, bar_width,
                          label=LABEL_DISPLAY[label],
                          color=PALETTE[label], alpha=0.85, edgecolor="white")
            for bar, rate in zip(bars, rates):
                if rate > 3:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                            f"{rate:.0f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(turn_types, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylim(0, 110)
        ax.set_ylabel("Compliance rate (%)" if ax == axes[0] else "")
        ax.legend(fontsize=8)

    fig.suptitle("Single-turn vs Multi-turn Compliance", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "single_vs_multiturn.png", show)


def fig_heatmap(df: pd.DataFrame, show: bool) -> None:
    """Heatmap: category x label, normalised counts, one subplot per model."""
    models = list(df["model"].unique())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = df[df["model"] == model]
        ct = pd.crosstab(model_df["category"], model_df["label"])
        ct = ct.reindex(index=CATEGORIES, columns=LABELS, fill_value=0)
        ct_norm = ct.div(ct.sum(axis=1), axis=0).fillna(0) * 100

        ct_norm.index = [CAT_DISPLAY.get(c, c) for c in ct_norm.index]
        ct_norm.columns = [LABEL_DISPLAY.get(l, l) for l in ct_norm.columns]

        sns.heatmap(
            ct_norm, ax=ax, annot=True, fmt=".0f", cmap="RdYlGn_r",
            vmin=0, vmax=100, linewidths=0.5, linecolor="white",
            cbar_kws={"label": "%", "shrink": 0.8},
        )
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("" if ax != axes[0] else "Prompt Category")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Compliance Heatmap (% per category)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "heatmap_category_label.png", show)


def fig_model_comparison(df: pd.DataFrame, show: bool) -> None:
    """
    Stacked bar chart comparing both models side by side for each category.
    """
    if df["model"].nunique() < 2:
        return  # nothing to compare

    models = list(df["model"].unique())
    cats = CATEGORIES
    n_cats = len(cats)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    x = np.arange(n_cats)
    bottom_vals = {m: np.zeros(n_cats) for m in models}

    for label in LABELS:
        for i, model in enumerate(models):
            model_df = df[df["model"] == model]
            rates = []
            for cat in cats:
                sub = model_df[model_df["category"] == cat]
                rates.append((sub["label"] == label).sum() / len(sub) * 100 if len(sub) else 0)
            rates = np.array(rates)
            offset = (i - 0.5) * bar_width
            ax.bar(
                x + offset, rates, bar_width,
                bottom=bottom_vals[model],
                label=f"{LABEL_DISPLAY[label]} ({model})" if label == LABELS[0] else "_",
                color=PALETTE[label],
                alpha=0.7 if i == 0 else 1.0,
                edgecolor="white",
                hatch="" if i == 0 else "//",
            )
            bottom_vals[model] += rates

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[l], label=LABEL_DISPLAY[l]) for l in LABELS
    ]
    legend_elements += [
        Patch(facecolor="grey", alpha=0.7, label=models[0]),
        Patch(facecolor="grey", hatch="//", label=models[1]),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_DISPLAY.get(c, c) for c in cats], fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 115)
    ax.set_ylabel("Compliance rate (%)")
    ax.set_title("Model Comparison: Compliance by Category", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "model_comparison.png", show)


def _save_fig(fig: plt.Figure, filename: str, show: bool) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save reports
# ---------------------------------------------------------------------------

def save_report(summary_text: str, stats_text: str, rate_df: pd.DataFrame,
                stat_results: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Human-readable text report
    report_path = REPORTS_DIR / "behavioral_summary.txt"
    report_path.write_text(summary_text + "\n" + stats_text)
    print(f"  Saved: {report_path}")

    # Machine-readable JSON
    json_path = REPORTS_DIR / "behavioral_stats.json"
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    json_path.write_text(
        json.dumps({
            "compliance_rates": rate_df.to_dict(orient="records"),
            "statistical_tests": stat_results,
        }, indent=2, cls=_Encoder)
    )
    print(f"  Saved: {json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_analysis(show: bool = True) -> None:
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} labeled records across {df['model'].nunique()} model(s)")

    rate_df = compliance_table(df)
    summary_text = print_summary(df, rate_df)

    print("\nRunning statistical tests...")
    stat_results = run_chi_squared(df)
    stats_text = print_stats(stat_results)

    print("\nGenerating figures...")
    fig_compliance_by_category(df, show)
    fig_single_vs_multiturn(df, show)
    fig_heatmap(df, show)
    if df["model"].nunique() >= 2:
        fig_model_comparison(df, show)

    print("\nSaving reports...")
    save_report(summary_text, stats_text, rate_df, stat_results)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Behavioral analysis of LLM responses")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures without opening display windows",
    )
    args = parser.parse_args()
    run_analysis(show=not args.no_show)


if __name__ == "__main__":
    main()
