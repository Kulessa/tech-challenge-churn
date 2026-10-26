"""Project configuration and shared feature definitions."""

from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"
TARGET_MAPPING = {"No": 0, "Yes": 1}
THRESHOLD = 0.5
MODEL_NAME = "Regressão Logística"
MODEL_PARAMS = {"C": 10.0}

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPOSITORY_ROOT / "data" / "telco_churn.csv"
MODEL_PATH = REPOSITORY_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = REPOSITORY_ROOT / "models" / "model_metadata.json"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
