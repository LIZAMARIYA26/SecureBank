"""
Project-wide paths and constants.

Every module imports from here so file locations are never hardcoded
in multiple places. Paths are resolved from this file's directory.
"""

from pathlib import Path

# Root folder of the project (wherever this file lives)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data and artefact folders
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UTILS_DIR = PROJECT_ROOT / "utils"

DATASET_PATH = DATASET_DIR / "banksim.csv"
SAMPLE_CSV_PATH = DATASET_DIR / "sample_transactions.csv"

# Trained artefacts (created by train_model.py)
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
LOOKUPS_PATH = MODELS_DIR / "lookups.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
COMPARISON_PATH = MODELS_DIR / "comparison.json"
HISTORY_PATH = OUTPUTS_DIR / "prediction_history.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.pkl"

# Train / test split used in the research paper
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Undersample the majority class so training finishes in a reasonable time
# while keeping EVERY fraud row (minority class).
MAX_GENUINE_SAMPLES = 30000

# Random Forest defaults (used before GridSearchCV)
RF_PARAMS = {
    "n_estimators": 120,
    "max_depth": 16,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": "balanced",
}

# Small grid so hyperparameter tuning is practical on a laptop
PARAM_GRID = {
    "n_estimators": [80, 140],
    "max_depth": [12, 18],
    "min_samples_split": [5],
    "min_samples_leaf": [1, 2],
}

# BankSim age codes → human-readable bands (used in the web form)
AGE_BANDS = {
    0: "0–17",
    1: "18–25",
    2: "26–35",
    3: "36–45",
    4: "46–55",
    5: "56–65",
    6: "65+",
}

# Extra categorical values injected so the web form matches the model
PAYMENT_MODES = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"]
DEVICE_TYPES = ["Mobile", "Desktop", "POS Terminal", "ATM"]
LOCATIONS = ["Madrid", "Barcelona", "Valencia", "Seville", "Bilbao", "Malaga"]
SPENDING_CATEGORIES = ["Low", "Medium", "High", "Premium"]

# Make sure output folders exist when the app starts
for folder in (DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, STATIC_DIR, TEMPLATES_DIR):
    folder.mkdir(parents=True, exist_ok=True)
