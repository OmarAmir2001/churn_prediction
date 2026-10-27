import pandas as pd
import structlog

from churn_prediction import config

log = structlog.get_logger()


def load_data(path=config.DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the Telco CSV file in the data/ folder."
        )
    df = pd.read_csv(path)
    log.info("data_loaded", rows=len(df), columns=df.shape[1])
    return df


def clean_data(df):
    df = df.copy()

    df.columns = df.columns.str.lower().str.replace(" ", "_")
    text_columns = list(df.select_dtypes(exclude="number").columns)
    for c in text_columns:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce").fillna(0)
    df.churn = (df.churn == "yes").astype(int)

    log.info("data_cleaned", rows=len(df), churn_rate=round(float(df.churn.mean()), 3))
    return df