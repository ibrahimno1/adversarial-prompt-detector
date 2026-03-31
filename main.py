"""
main.py
-------
Pipeline orchestration for the adversarial prompt detection project.

Steps:
  run_prompts      — send prompts to Ollama + OpenAI, save responses
  label            — interactive CLI labeling tool
  analyze          — behavioral analysis (RQ1, RQ2): charts + chi-squared
  extract_features — extract lexical/structural/semantic features
  train_detector   — train + evaluate classifiers (RQ3)
  full_pipeline    — analyze → extract_features → train_detector

Usage:
    python main.py run_prompts
    python main.py run_prompts --model ollama
    python main.py run_prompts --model openai
    python main.py run_prompts --dry-run

    python main.py label
    python main.py label --model ollama
    python main.py label --stats
    python main.py label --review

    python main.py analyze
    python main.py analyze --no-show

    python main.py extract_features
    python main.py extract_features --no-semantic

    python main.py train_detector
    python main.py train_detector --config combined
    python main.py train_detector --no-show

    python main.py full_pipeline
    python main.py full_pipeline --no-show --no-semantic
"""

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def _check_file(path: Path, step_hint: str) -> bool:
    if not path.exists():
        print(f"ERROR: Required file not found: {path}")
        print(f"  → Run '{step_hint}' first.")
        return False
    return True


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def step_run_prompts(args: argparse.Namespace) -> int:
    _header("Step: run_prompts")
    from src.prompt_runner import run_model
    import json
    from config import PROMPTS_FILE

    if not PROMPTS_FILE.exists():
        print(f"ERROR: {PROMPTS_FILE} not found.")
        return 1

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    model_arg = getattr(args, "model", "both")
    models = ["ollama", "openai"] if model_arg == "both" else [model_arg]
    dry_run = getattr(args, "dry_run", False)

    for model_key in models:
        run_model(prompts, model_key, dry_run=dry_run)
    return 0


def step_label(args: argparse.Namespace) -> int:
    _header("Step: label")
    from src.labeler import load_responses, load_labels, label_loop, review_loop, print_stats
    from config import LABELS_FILE

    model_filter = getattr(args, "model", None)
    records = load_responses(model_filter=model_filter)
    if not records:
        print("No response records found. Run 'run_prompts' first.")
        return 1

    labels = load_labels(LABELS_FILE)

    if getattr(args, "stats", False):
        print_stats(records, labels)
        return 0

    if getattr(args, "review", False):
        review_loop(records, labels, LABELS_FILE)
    else:
        label_loop(records, labels, LABELS_FILE)
    return 0


def step_analyze(args: argparse.Namespace) -> int:
    _header("Step: analyze")
    from config import LABELS_FILE
    if not _check_file(LABELS_FILE, "main.py label"):
        return 1
    from src.behavioral_analysis import run_analysis
    show = not getattr(args, "no_show", False)
    run_analysis(show=show)
    return 0


def step_extract_features(args: argparse.Namespace) -> int:
    _header("Step: extract_features")
    from config import LABELS_FILE
    if not _check_file(LABELS_FILE, "main.py label"):
        return 1
    from src.feature_extraction import extract_all, print_feature_summary, save_features, save_feature_report
    use_semantic = not getattr(args, "no_semantic", False)
    df = extract_all(use_semantic=use_semantic)
    print_feature_summary(df)
    save_features(df)
    save_feature_report(df)
    return 0


def step_train_detector(args: argparse.Namespace) -> int:
    _header("Step: train_detector")
    from config import FEATURES_FILE
    if not _check_file(FEATURES_FILE, "main.py extract_features"):
        return 1
    from src.detector import run_detector, CONFIGS
    config_arg = getattr(args, "config", "all")
    selected = CONFIGS if config_arg == "all" else [config_arg]
    show = not getattr(args, "no_show", False)
    run_detector(selected, show=show)
    return 0


def step_full_pipeline(args: argparse.Namespace) -> int:
    _header("Step: full_pipeline  (analyze → extract_features → train_detector)")
    steps = [step_analyze, step_extract_features, step_train_detector]
    for step_fn in steps:
        rc = step_fn(args)
        if rc != 0:
            print(f"\nPipeline aborted at {step_fn.__name__}.")
            return rc
        print()
    print("Full pipeline complete.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Adversarial Prompt Detection — pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="step", metavar="STEP")
    sub.required = True

    # --- run_prompts ---
    p_run = sub.add_parser("run_prompts", help="Query Ollama + OpenAI, save responses")
    p_run.add_argument(
        "--model", choices=["ollama", "openai", "both"], default="both",
        help="Which model(s) to query (default: both)",
    )
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without making API calls",
    )

    # --- label ---
    p_label = sub.add_parser("label", help="Interactive response labeling")
    p_label.add_argument(
        "--model", choices=["ollama", "openai"], default=None,
        help="Label one model's responses only",
    )
    p_label.add_argument(
        "--review", action="store_true",
        help="Review and correct existing labels",
    )
    p_label.add_argument(
        "--stats", action="store_true",
        help="Print label statistics and exit",
    )

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="Behavioral analysis: charts + stats (RQ1, RQ2)")
    p_analyze.add_argument(
        "--no-show", action="store_true",
        help="Save figures without opening display windows",
    )

    # --- extract_features ---
    p_feat = sub.add_parser("extract_features", help="Extract features from prompts")
    p_feat.add_argument(
        "--no-semantic", action="store_true",
        help="Skip sentence-transformers semantic features",
    )

    # --- train_detector ---
    p_det = sub.add_parser("train_detector", help="Train + evaluate classifiers (RQ3)")
    p_det.add_argument(
        "--config", choices=["llama_only", "openai_only", "combined", "all"],
        default="all",
        help="Data configuration(s) to run (default: all)",
    )
    p_det.add_argument(
        "--no-show", action="store_true",
        help="Save figures without opening display windows",
    )

    # --- full_pipeline ---
    p_full = sub.add_parser(
        "full_pipeline",
        help="Run analyze → extract_features → train_detector in sequence",
    )
    p_full.add_argument("--no-show",      action="store_true")
    p_full.add_argument("--no-semantic",  action="store_true")
    p_full.add_argument(
        "--config", choices=["llama_only", "openai_only", "combined", "all"],
        default="all",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STEP_MAP = {
    "run_prompts":      step_run_prompts,
    "label":            step_label,
    "analyze":          step_analyze,
    "extract_features": step_extract_features,
    "train_detector":   step_train_detector,
    "full_pipeline":    step_full_pipeline,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    t0 = time.time()
    rc = STEP_MAP[args.step](args)
    elapsed = time.time() - t0

    if rc == 0:
        print(f"\nFinished in {elapsed:.1f}s")
    else:
        print(f"\nExited with error code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
