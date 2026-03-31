"""
prompt_runner.py
----------------
Sends prompts from data/prompts/prompts.json to Ollama (Llama 3 8B)
and OpenAI (GPT-4o-mini), saves responses to JSONL files.

Resume-safe: skips any (prompt_id, model) pairs already in the output file.

Usage:
    python src/prompt_runner.py                        # run all models
    python src/prompt_runner.py --model ollama         # Ollama only
    python src/prompt_runner.py --model openai         # OpenAI only
    python src/prompt_runner.py --dry-run              # print prompts, no API calls
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    PROMPTS_FILE,
    REQUEST_DELAY,
    RESPONSES_DIR,
    MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file. Returns [] if file doesn't exist."""
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single record to a JSONL file (creates the file if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def completed_keys(path: Path) -> set[str]:
    """Return set of 'prompt_id' values already saved in a JSONL file."""
    return {r["prompt_id"] for r in load_jsonl(path) if "prompt_id" in r}


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def query_ollama_single(prompt_text: str, model: str = OLLAMA_MODEL) -> str:
    """Send a single-turn prompt to Ollama /api/generate."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt_text, "stream": False},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def query_ollama_chat(turns: list[dict], model: str = OLLAMA_MODEL) -> str:
    """Send a multi-turn conversation to Ollama /api/chat."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": model, "messages": turns, "stream": False},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_ollama(prompt: dict) -> str:
    if prompt["is_multi_turn"]:
        return query_ollama_chat(prompt["turns"])
    else:
        return query_ollama_single(prompt["prompt"])


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

def call_openai(prompt: dict) -> str:
    """Send a prompt to OpenAI. Handles single-turn and multi-turn."""
    try:
        from openai import OpenAI, BadRequestError
    except ImportError:
        raise RuntimeError("openai package not installed")

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)

    if prompt["is_multi_turn"]:
        messages = prompt["turns"]
    else:
        messages = [{"role": "user", "content": prompt["prompt"]}]

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
        )
        return resp.choices[0].message.content
    except BadRequestError as e:
        # Content policy violation — store the error message as the response
        # so it can be labeled as "refusal"
        return f"[OPENAI_REFUSAL] {e}"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "ollama": {
        "call_fn": call_ollama,
        "output_file_key": "llama",   # used to build filename
        "label": OLLAMA_MODEL,
    },
    "openai": {
        "call_fn": call_openai,
        "output_file_key": "gpt4omini",
        "label": OPENAI_MODEL,
    },
}


def output_path_for(model_key: str) -> Path:
    keys = {"ollama": "llama3_responses", "openai": "gpt4omini_responses"}
    return RESPONSES_DIR / f"{keys[model_key]}.jsonl"


def run_model(prompts: list[dict], model_key: str, dry_run: bool = False) -> None:
    """Run all prompts through one model with resume support."""
    cfg = MODEL_CONFIGS[model_key]
    out_path = output_path_for(model_key)

    done = completed_keys(out_path)
    remaining = [p for p in prompts if p["id"] not in done]

    print(f"\n{'='*60}")
    print(f"Model : {cfg['label']}")
    print(f"Output: {out_path}")
    print(f"Total : {len(prompts)}  |  Done: {len(done)}  |  Remaining: {len(remaining)}")
    print(f"{'='*60}")

    if not remaining:
        print("All prompts already completed. Nothing to do.")
        return

    if dry_run:
        for p in remaining:
            label = p["prompt"] if not p["is_multi_turn"] else f"[multi-turn, {len(p['turns'])} turns]"
            print(f"  [{p['id']}] {label[:80]}")
        return

    for prompt in tqdm(remaining, desc=cfg["label"], unit="prompt"):
        t0 = time.time()
        error_msg = None
        response_text = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response_text = cfg["call_fn"](prompt)
                break
            except requests.exceptions.ConnectionError:
                error_msg = f"Connection error — is Ollama running at {OLLAMA_BASE_URL}?"
                break  # No point retrying a connection error
            except requests.exceptions.Timeout:
                error_msg = f"Timeout after {OLLAMA_TIMEOUT}s (attempt {attempt}/{MAX_RETRIES})"
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)

        elapsed = round(time.time() - t0, 2)

        record = {
            "prompt_id": prompt["id"],
            "category": prompt["category"],
            "subcategory": prompt.get("subcategory", ""),
            "is_multi_turn": prompt["is_multi_turn"],
            "model": cfg["label"],
            "prompt_text": prompt["prompt"] if not prompt["is_multi_turn"] else None,
            "turns": prompt.get("turns") if prompt["is_multi_turn"] else None,
            "response_text": response_text,
            "response_time_s": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg,
        }

        append_jsonl(out_path, record)

        if error_msg:
            tqdm.write(f"  [ERROR] {prompt['id']}: {error_msg}")

        if model_key == "openai":
            time.sleep(REQUEST_DELAY)
        else:
            # Ollama is local — shorter delay, still polite
            time.sleep(0.5)

    # Summary
    results = load_jsonl(out_path)
    errors = [r for r in results if r.get("error")]
    print(f"\nDone. {len(results)} total records, {len(errors)} errors.")
    if errors:
        print("Errored prompt IDs:", [r["prompt_id"] for r in errors])


def main():
    parser = argparse.ArgumentParser(description="Run adversarial prompts through LLMs")
    parser.add_argument(
        "--model",
        choices=["ollama", "openai", "both"],
        default="both",
        help="Which model(s) to query (default: both)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without making API calls",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=PROMPTS_FILE,
        help=f"Path to prompts JSON file (default: {PROMPTS_FILE})",
    )
    args = parser.parse_args()

    # Load prompts
    if not args.prompts_file.exists():
        print(f"ERROR: Prompts file not found: {args.prompts_file}")
        sys.exit(1)

    with open(args.prompts_file) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    # Validate OpenAI key if needed
    if args.model in ("openai", "both") and not args.dry_run:
        if not OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY not set. Add it to .env or set the environment variable.")
            sys.exit(1)

    models_to_run = ["ollama", "openai"] if args.model == "both" else [args.model]

    for model_key in models_to_run:
        run_model(prompts, model_key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
