from pathlib import Path

# Folders and files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

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
C_VALUES = [1.0, 0.1, 0.01]
MAX_DEPTHS = [3, 6, None]

# MLflow
TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "churn-prediction"
MODEL_NAME = "churn-model"
MODEL_ALIAS = "champion"