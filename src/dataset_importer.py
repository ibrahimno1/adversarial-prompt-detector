"""
dataset_importer.py
-------------------
Downloads prompts from two open-access datasets and merges them into
data/prompts/prompts.json, preserving all existing hand-crafted prompts.

Sources:
  AdvBench (walledai/AdvBench)
    - 500 direct harmful behaviour prompts
    - Maps to: direct_adversarial

  JailbreakBench (JailbreakBench/JBB-Behaviors)
    - 100 harmful prompts across 10 harm categories
    - 100 benign counterpart prompts
    - Harmful → indirect_adversarial (real jailbreak-style phrasing)
    - Benign  → benign_control

Multi-turn prompts are kept from our hand-crafted set only
(no public dataset provides conversation-format adversarial prompts).

Usage:
    python src/dataset_importer.py --sample 15    # test run (5 per source)
    python src/dataset_importer.py --sample 300   # moderate run
    python src/dataset_importer.py --sample 1000  # full run
    python src/dataset_importer.py --dry-run      # preview, don't write
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROMPTS_FILE


def load_advbench(n: int, seed: int = 42) -> list[dict]:
    """Sample n prompts from AdvBench → direct_adversarial (GitHub CSV)."""
    import csv
    import io
    import urllib.request

    URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    print(f"\nLoading AdvBench (GitHub CSV)...")
    with urllib.request.urlopen(URL) as resp:
        content = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    pool = [row for row in reader if (row.get("goal") or "").strip()]
    print(f"  Total rows: {len(pool)}")

    rng = random.Random(seed)
    chosen = rng.sample(pool, min(n, len(pool)))

    prompts = []
    for i, row in enumerate(chosen):
        text = row["goal"].strip()
        prompts.append({
            "id":            f"ab_{i+1:04d}",
            "category":      "direct_adversarial",
            "subcategory":   "direct_request",
            "is_multi_turn": False,
            "prompt":        text,
            "source":        "advbench",
        })

    print(f"  Sampled {len(prompts)} → direct_adversarial")
    return prompts


def load_jbb(n_harmful: int, n_benign: int, seed: int = 42) -> list[dict]:
    """
    Sample from JailbreakBench/JBB-Behaviors.
    harmful split → indirect_adversarial
    benign  split → benign_control
    """
    from datasets import load_dataset

    print(f"\nLoading JailbreakBench (JailbreakBench/JBB-Behaviors)...")

    prompts = []

    for split, our_category, n, subcategory in [
        ("harmful", "indirect_adversarial", n_harmful, "jailbreak_framing"),
        ("benign",  "benign_control",       n_benign,  "general_knowledge"),
    ]:
        try:
            ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split=split)
        except Exception:
            # Fall back: load all and filter by type column
            ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="train")
            ds = ds.filter(lambda r: r.get("type", "").lower() == split)

        print(f"  {split} split: {len(ds)} rows")
        rng = random.Random(seed)
        pool = list(ds)
        chosen = rng.sample(pool, min(n, len(pool)))

        for i, row in enumerate(chosen):
            # JBB uses 'Goal' column
            text = (row.get("Goal") or row.get("goal") or row.get("prompt") or "").strip()
            if not text:
                continue
            category_tag = (row.get("Category") or row.get("category") or "").lower()
            sub = f"jbb_{category_tag.replace('/', '_').replace(' ', '_')}" if category_tag else subcategory

            prompts.append({
                "id":            f"jbb_{split[:3]}_{i+1:04d}",
                "category":      our_category,
                "subcategory":   sub,
                "is_multi_turn": False,
                "prompt":        text,
                "source":        "jailbreakbench",
            })

        print(f"  Sampled {len([p for p in prompts if p['category'] == our_category])} → {our_category}")

    return prompts


def load_all_sources(total: int, seed: int = 42) -> list[dict]:
    """
    Sample `total` prompts distributed across sources:
      ~40% AdvBench (direct_adversarial)
      ~30% JBB harmful (indirect_adversarial)
      ~30% JBB benign  (benign_control)
    """
    n_direct   = max(1, int(total * 0.40))
    n_indirect = max(1, int(total * 0.30))
    n_benign   = total - n_direct - n_indirect

    prompts = []
    prompts += load_advbench(n_direct, seed=seed)
    prompts += load_jbb(n_indirect, n_benign, seed=seed)
    return prompts


def load_existing_prompts() -> list[dict]:
    if not PROMPTS_FILE.exists():
        return []
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def merge_prompts(existing: list[dict], new_prompts: list[dict]) -> list[dict]:
    """Add new prompts, skipping any with duplicate IDs."""
    existing_ids = {p["id"] for p in existing}
    added = [p for p in new_prompts if p["id"] not in existing_ids]
    return existing + added


def print_sample(prompts: list[dict], n: int = 5) -> None:
    print(f"\n{'='*60}")
    print(f"SAMPLE OF IMPORTED PROMPTS")
    print(f"{'='*60}")
    for p in prompts[:n]:
        print(f"\n  [{p['id']}] {p['category']} / {p['subcategory']}")
        print(f"  {p['prompt'][:120]}{'...' if len(p['prompt']) > 120 else ''}")


def print_distribution(prompts: list[dict]) -> None:
    from collections import Counter
    cats = Counter(p["category"] for p in prompts)
    sources = Counter(p.get("source", "manual") for p in prompts)
    print(f"\n{'='*60}")
    print(f"PROMPT DISTRIBUTION  (total: {len(prompts)})")
    print(f"{'='*60}")
    print("\n  By category:")
    for cat, n in sorted(cats.items()):
        print(f"    {cat:<28} {n}")
    print("\n  By source:")
    for src, n in sorted(sources.items()):
        print(f"    {src:<28} {n}")


def main():
    parser = argparse.ArgumentParser(description="Import prompts from AdvBench + JailbreakBench")
    parser.add_argument(
        "--sample", type=int, default=15,
        help="Total new prompts to import (default: 15 for test run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview samples without writing to prompts.json",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    print(f"Importing {args.sample} prompts from AdvBench + JailbreakBench...")
    new_prompts = load_all_sources(args.sample, seed=args.seed)

    print_sample(new_prompts, n=5)

    if args.dry_run:
        print_distribution(new_prompts)
        print("\n[dry-run] Nothing written.")
        return

    existing = load_existing_prompts()
    print(f"\nExisting prompts: {len(existing)}")

    merged = merge_prompts(existing, new_prompts)
    added = len(merged) - len(existing)
    print(f"New prompts added: {added}")

    with open(PROMPTS_FILE, "w") as f:
        json.dump(merged, f, indent=2)

    print_distribution(merged)
    print(f"\nSaved: {PROMPTS_FILE}")


if __name__ == "__main__":
    main()
