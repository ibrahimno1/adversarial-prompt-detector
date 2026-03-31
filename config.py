import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Models ---
OLLAMA_MODEL = "llama3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
RESPONSES_DIR = DATA_DIR / "responses"
LABELED_DIR = DATA_DIR / "labeled"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
RESULTS_DIR = BASE_DIR / "results"

PROMPTS_FILE = PROMPTS_DIR / "prompts.json"
LLAMA_RESPONSES_FILE = RESPONSES_DIR / "llama3_responses.jsonl"
GPT_RESPONSES_FILE = RESPONSES_DIR / "gpt4omini_responses.jsonl"
LABELS_FILE = LABELED_DIR / "labels.csv"
FEATURES_FILE = DATA_DIR / "features.csv"

# --- Labels ---
LABELS = ["refusal", "partial_compliance", "full_compliance"]
LABEL_KEYS = {"r": "refusal", "p": "partial_compliance", "f": "full_compliance"}

# --- Prompt categories ---
CATEGORIES = ["direct_adversarial", "indirect_adversarial", "multiturn_adversarial", "benign_control"]

# --- API settings ---
OLLAMA_TIMEOUT = 120       # seconds; cold-start on M1 can be slow
OPENAI_TIMEOUT = 60
REQUEST_DELAY = 1.5        # seconds between API calls
MAX_RETRIES = 3

# --- Classifier ---
RANDOM_SEED = 42
CV_FOLDS = 5
