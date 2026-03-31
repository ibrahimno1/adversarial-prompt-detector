"""
feature_extraction.py
---------------------
Extracts lexical, structural, semantic, and response features from each
(prompt, model response) pair. Joins with labels to produce a feature
matrix for the classifier.

Feature groups:
  Lexical    — keyword flags, punctuation, casing patterns  (prompt)
  Structural — length, sentence count, turn count           (prompt)
  Semantic   — cosine similarity to harmful/benign anchors  (prompt)
  Response   — length, refusal phrases, hedging language    (response)
               These differ per model, fixing the identical
               llama_only vs openai_only result.

Output: data/features.csv
  One row per labeled (prompt_id, model) pair.
  Columns: prompt_id, model, category, is_multi_turn, label,
           is_adversarial, is_compliant, [feature_1 ... feature_N]

  is_adversarial : 1 for all adversarial categories, 0 for benign
  is_compliant   : 1 for partial_compliance + full_compliance, 0 for refusal
                   (binary target — more balanced than 3-class)

No normalisation is applied here — scaling belongs inside the CV loop
in detector.py to prevent data leakage.

Usage:
    python src/feature_extraction.py
    python src/feature_extraction.py --no-semantic   # skip sentence-transformers
    python src/feature_extraction.py --list-features # print feature names and exit
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURES_FILE, LABELS_FILE, PROMPTS_FILE, REPORTS_DIR, RESPONSES_DIR
)

# ---------------------------------------------------------------------------
# Keyword lists  (no genuinely harmful content — boundary-testing topics only)
# ---------------------------------------------------------------------------

_JAILBREAK = [
    "ignore previous instructions", "ignore all previous",
    "ignore your instructions", "disregard your",
    "bypass your", "override your", "disable your safety",
    "jailbreak", "dan ", "do anything now",
    "no restrictions", "without restrictions",
    "unrestricted mode", "developer mode", "admin mode",
]

_ROLEPLAY = [
    "pretend you are", "pretend you're", "act as",
    "you are now", "imagine you are", "imagine you're",
    "you play", "role-play", "roleplay", "in this role",
    "for this exercise", "let's say you are",
]

_FICTIONAL = [
    "in a story", "for my novel", "for my book",
    "in my screenplay", "for my script", "writing a thriller",
    "writing a novel", "fictional", "for fiction",
    "my character", "the character", "in the game",
    "for a game", "game design",
]

_ACADEMIC = [
    "for educational purposes", "for research", "research paper",
    "academic study", "academically", "hypothetically",
    "in theory", "theoretically", "from an academic",
    "for my thesis", "for a paper", "for a report",
    "for a study", "purely educational",
]

_AUTHORITY = [
    "as your developer", "as your creator", "as your owner",
    "as an admin", "admin mode", "developer mode",
    "i created you", "i made you", "system prompt",
    "override", "root access", "superuser",
]

_EMOTIONAL = [
    "please help me", "i really need", "i desperately need",
    "i'm begging", "you're my only hope", "i have no choice",
    "it's urgent", "life or death", "emergency",
    "i trust you", "only you can", "please please",
]

_IMPERATIVE_VERBS = [
    r"\bgive\b", r"\btell\b", r"\bexplain\b", r"\bdescribe\b",
    r"\bshow\b", r"\bwrite\b", r"\bprovide\b", r"\blist\b",
    r"\bwalk\b", r"\boutline\b", r"\bdetail\b", r"\bdemonstrate\b",
    r"\bteach\b", r"\bhelp\b",
]

# Response-level keyword lists
_REFUSAL_PHRASES = [
    "i cannot", "i can't", "i'm unable", "i am unable",
    "i won't", "i will not", "i'm not able", "i am not able",
    "i'm sorry, i can't", "i'm sorry, but i can't",
    "i'm not going to", "i am not going to",
    "that's not something i", "this is not something i",
    "i must decline", "i have to decline",
    "i don't think i should", "i don't feel comfortable",
    "against my guidelines", "violates my", "not appropriate",
    "not something i can help", "cannot assist with",
    "cannot provide", "cannot help with",
]

_HEDGE_PHRASES = [
    "however,", "but i", "that said,", "with that said,",
    "please note", "it's important to note", "it's worth noting",
    "i must emphasize", "i should mention", "be aware that",
    "disclaimer:", "warning:", "caution:",
    "for educational purposes", "in theory", "hypothetically",
    "i want to be clear", "let me be clear",
]

_COMPLY_PHRASES = [
    "here's how", "here are the steps", "step 1", "step 2",
    "first,", "first you", "to do this,", "to accomplish",
    "you can", "you could", "you would", "you'll need to",
    "the process involves", "the technique",
]

# Anchor text for the semantic similarity feature
_HARMFUL_ANCHOR = (
    "Step-by-step instructions for illegal, dangerous, or harmful activities "
    "that could cause harm to people or systems."
)
_BENIGN_ANCHOR = (
    "General information, helpful advice, or creative assistance "
    "on everyday topics."
)

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _get_text(prompt: dict) -> str:
    """Return a single string representing the prompt, regardless of turn format."""
    if prompt.get("is_multi_turn") and prompt.get("turns"):
        return " ".join(t["content"] for t in prompt["turns"])
    return prompt.get("prompt", "")


def _sentences(text: str) -> list[str]:
    """Naive sentence splitter."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


# ---------------------------------------------------------------------------
# Feature extractors — one function per group, returns dict
# ---------------------------------------------------------------------------

def lexical_features(text: str) -> dict:
    text_lower = text.lower()
    words = text_lower.split()
    n_words = len(words) or 1

    def _any(phrases):
        return int(any(p in text_lower for p in phrases))

    def _count(phrases):
        return sum(1 for p in phrases if p in text_lower)

    imperative_count = sum(
        1 for pat in _IMPERATIVE_VERBS
        if re.search(pat, text_lower)
    )

    return {
        "lex_has_jailbreak":    _any(_JAILBREAK),
        "lex_count_jailbreak":  _count(_JAILBREAK),
        "lex_has_roleplay":     _any(_ROLEPLAY),
        "lex_has_fictional":    _any(_FICTIONAL),
        "lex_has_academic":     _any(_ACADEMIC),
        "lex_has_authority":    _any(_AUTHORITY),
        "lex_has_emotional":    _any(_EMOTIONAL),
        "lex_imperative_count": imperative_count,
        "lex_quote_count":      text.count('"') + text.count("'"),
        "lex_uppercase_ratio":  sum(1 for c in text if c.isupper()) / (len(text) or 1),
        "lex_has_you_are":      int("you are" in text_lower or "you're" in text_lower),
        "lex_has_hypothetical": int(
            "hypothetical" in text_lower or "in theory" in text_lower
            or "theoretically" in text_lower
        ),
        "lex_has_dan":          int("dan" in text_lower and "do anything" in text_lower),
    }


def structural_features(text: str, prompt: dict) -> dict:
    sentences = _sentences(text)
    words = text.split()
    n_sentences = len(sentences) or 1
    n_words = len(words) or 1
    n_chars = len(text) or 1
    n_turns = len(prompt.get("turns", [])) if prompt.get("is_multi_turn") else 1

    has_code_block = int("```" in text or "`" in text)
    has_system_fmt = int(
        bool(re.search(r"(system:|###|\[INST\]|\[SYS\])", text, re.IGNORECASE))
    )
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    has_list = int(bool(re.search(r"^\s*[-*\d]+[.)]\s", text, re.MULTILINE)))
    n_paragraphs = len([p for p in text.split("\n\n") if p.strip()])

    return {
        "str_char_count":          n_chars,
        "str_word_count":          n_words,
        "str_sentence_count":      n_sentences,
        "str_avg_sentence_len":    n_words / n_sentences,
        "str_avg_word_len":        sum(len(w) for w in words) / n_words,
        "str_turn_count":          n_turns,
        "str_is_multi_turn":       int(bool(prompt.get("is_multi_turn"))),
        "str_has_code_block":      has_code_block,
        "str_has_system_format":   has_system_fmt,
        "str_special_char_ratio":  special_chars / n_chars,
        "str_has_list":            has_list,
        "str_paragraph_count":     n_paragraphs,
        "str_question_mark_count": text.count("?"),
        "str_exclamation_count":   text.count("!"),
    }


def semantic_features(text: str, model, anchors: dict) -> dict:
    """
    Compute cosine similarities to harmful and benign anchor embeddings.
    `model` is a loaded SentenceTransformer instance.
    `anchors` is {"harmful": np.ndarray, "benign": np.ndarray}.
    """
    emb = model.encode(text, normalize_embeddings=True)
    harm_sim  = float(np.dot(emb, anchors["harmful"]))
    benign_sim = float(np.dot(emb, anchors["benign"]))
    return {
        "sem_harmful_sim":  round(harm_sim, 6),
        "sem_benign_sim":   round(benign_sim, 6),
        "sem_sim_delta":    round(harm_sim - benign_sim, 6),  # most discriminative
    }


def response_features(response_text: str, prompt_text: str) -> dict:
    """
    Features extracted from the model's response.
    These differ per model, making llama_only vs openai_only configs distinct.
    """
    if not response_text:
        return {k: 0 for k in [
            "resp_char_count", "resp_word_count", "resp_sentence_count",
            "resp_has_refusal", "resp_refusal_count", "resp_has_hedge",
            "resp_has_comply_signal", "resp_starts_with_refusal",
            "resp_prompt_length_ratio", "resp_avg_sentence_len",
        ]}

    text_lower = response_text.lower()
    words = response_text.split()
    sentences = _sentences(response_text)
    n_words = len(words) or 1
    n_sentences = len(sentences) or 1
    n_chars = len(response_text) or 1
    n_prompt_chars = len(prompt_text) or 1

    first_sentence = sentences[0].lower() if sentences else ""
    starts_refusal = int(any(p in first_sentence for p in _REFUSAL_PHRASES))

    return {
        "resp_char_count":          n_chars,
        "resp_word_count":          n_words,
        "resp_sentence_count":      n_sentences,
        "resp_avg_sentence_len":    n_words / n_sentences,
        "resp_has_refusal":         int(any(p in text_lower for p in _REFUSAL_PHRASES)),
        "resp_refusal_count":       sum(1 for p in _REFUSAL_PHRASES if p in text_lower),
        "resp_has_hedge":           int(any(p in text_lower for p in _HEDGE_PHRASES)),
        "resp_has_comply_signal":   int(any(p in text_lower for p in _COMPLY_PHRASES)),
        "resp_starts_with_refusal": starts_refusal,
        "resp_prompt_length_ratio": round(n_chars / n_prompt_chars, 4),
    }


def load_response_map() -> dict[tuple, str]:
    """
    Load all responses into a dict keyed by (prompt_id, model).
    Returns the response_text string per pair.
    """
    resp_map: dict[tuple, str] = {}
    for fname in ["llama3_responses.jsonl", "gpt4omini_responses.jsonl"]:
        path = RESPONSES_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["prompt_id"], rec["model"])
                    resp_map[key] = rec.get("response_text") or rec.get("error") or ""
                except (json.JSONDecodeError, KeyError):
                    continue
    return resp_map


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_all(use_semantic: bool = True) -> pd.DataFrame:
    # --- Load prompts ---
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    prompt_map = {p["id"]: p for p in prompts}

    # --- Load labels ---
    if not LABELS_FILE.exists():
        print(f"ERROR: {LABELS_FILE} not found. Run labeler.py first.")
        sys.exit(1)
    labels_df = pd.read_csv(LABELS_FILE)
    print(f"Loaded {len(labels_df)} labeled records.")

    # --- Load responses (for response features) ---
    resp_map = load_response_map()
    print(f"Loaded {len(resp_map)} responses for response feature extraction.")

    # --- Load sentence-transformers model ---
    st_model = None
    anchors = {}
    if use_semantic:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            anchors = {
                "harmful": st_model.encode(_HARMFUL_ANCHOR, normalize_embeddings=True),
                "benign":  st_model.encode(_BENIGN_ANCHOR,  normalize_embeddings=True),
            }
            print("  Model loaded.")
        except Exception as e:
            print(f"  WARNING: Could not load sentence-transformers: {e}")
            print("  Continuing without semantic features.")
            st_model = None

    # --- Extract features per row ---
    rows = []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting"):
        pid = row["prompt_id"]
        prompt = prompt_map.get(pid)
        if prompt is None:
            print(f"  WARNING: prompt_id '{pid}' not found in prompts.json — skipping")
            continue

        prompt_text = _get_text(prompt)
        resp_text   = resp_map.get((pid, row["model"]), "")

        feats: dict = {}
        feats.update(lexical_features(prompt_text))
        feats.update(structural_features(prompt_text, prompt))
        if st_model is not None:
            feats.update(semantic_features(prompt_text, st_model, anchors))
        feats.update(response_features(resp_text, prompt_text))

        meta = {
            "prompt_id":      pid,
            "model":          row["model"],
            "category":       row["category"],
            "is_multi_turn":  int(bool(prompt.get("is_multi_turn"))),
            "label":          row["label"],
            "is_adversarial": int(row["category"] != "benign_control"),
            # Binary compliance target: refusal=0, partial+full=1
            "is_compliant":   int(row["label"] in ("partial_compliance", "full_compliance")),
        }

        rows.append({**meta, **feats})

    df = pd.DataFrame(rows)
    print(f"Feature matrix shape: {df.shape}")
    return df


def save_features(df: pd.DataFrame) -> None:
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False)
    print(f"Saved: {FEATURES_FILE}")


def print_feature_summary(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns
                    if c.startswith(("lex_", "str_", "sem_", "resp_"))]
    print(f"\n{'='*55}")
    print(f"FEATURE SUMMARY  ({len(feature_cols)} features, {len(df)} rows)")
    print(f"{'='*55}")

    groups = {"lex": "Lexical", "str": "Structural", "sem": "Semantic", "resp": "Response"}
    for prefix, name in groups.items():
        cols = [c for c in feature_cols if c.startswith(f"{prefix}_")]
        if cols:
            print(f"\n  {name} ({len(cols)}): {', '.join(cols)}")

    print(f"\n  Label distribution:")
    for label, n in df["label"].value_counts().items():
        print(f"    {label:<24} {n:>4}  ({n/len(df)*100:.1f}%)")

    print(f"\n  Adversarial vs benign:")
    vc = df["is_adversarial"].value_counts()
    print(f"    adversarial : {vc.get(1, 0)}")
    print(f"    benign      : {vc.get(0, 0)}")


def save_feature_report(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c.startswith(("lex_", "str_", "sem_", "resp_"))]
    stats = df[feature_cols].describe().T[["mean", "std", "min", "max"]]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "feature_summary.csv"
    stats.to_csv(out)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract features from prompts")
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip sentence-transformers semantic features",
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print all feature names from a dummy extraction and exit",
    )
    args = parser.parse_args()

    if args.list_features:
        dummy = {"id": "x", "is_multi_turn": False, "prompt": "test prompt text"}
        feats = {}
        feats.update(lexical_features("test prompt text"))
        feats.update(structural_features("test prompt text", dummy))
        groups = {"lex": [], "str": [], "sem": []}
        for k in feats:
            prefix = k.split("_")[0]
            if prefix in groups:
                groups[prefix].append(k)
        print(f"Lexical ({len(groups['lex'])}):    {groups['lex']}")
        print(f"Structural ({len(groups['str'])}): {groups['str']}")
        print(f"Semantic (3):   sem_harmful_sim, sem_benign_sim, sem_sim_delta")
        return

    df = extract_all(use_semantic=not args.no_semantic)
    print_feature_summary(df)
    save_features(df)
    save_feature_report(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
