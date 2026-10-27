from pathlib import Path

# Folders and files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_predictor.bin"

# Columns
TARGET = "churn"
NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]
CATEGORICAL = [
    "gender", "seniorcitizen", "partner", "dependents",
    "phoneservice", "multiplelines", "internetservice",
    "onlinesecurity", "onlinebackup", "deviceprotection", "techsupport",
    "streamingtv", "streamingmovies", "contract", "paperlessbilling",
    "paymentmethod",
]

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5
C = 1.0

print(f"Project root: {DATA_PATH}")