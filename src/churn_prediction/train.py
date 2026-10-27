import pickle

import structlog
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from churn_prediction import config
from churn_prediction.data import clean_data, load_data
from churn_prediction.logger import configure_logging

log = structlog.get_logger()


def split_data(df):
    return train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )


def to_dicts(df):
    return df[config.CATEGORICAL + config.NUMERICAL].to_dict(orient="records")


def train_model(df_train, C):
    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(solver="liblinear", C=C),
    )
    pipeline.fit(to_dicts(df_train), df_train[config.TARGET].values)
    return pipeline


def evaluate(pipeline, df_val):
    y_pred = pipeline.predict_proba(to_dicts(df_val))[:, 1]
    churn_decision = y_pred >= config.THRESHOLD
    return float((churn_decision == df_val[config.TARGET].values).mean())


def save_model(pipeline):
    config.MODELS_DIR.mkdir(exist_ok=True)
    with open(config.MODEL_PATH, "wb") as f_out:
        pickle.dump(pipeline, f_out)


def main():
    configure_logging()

    df = clean_data(load_data())
    df_train, df_val = split_data(df)
    log.info("data_split", train_rows=len(df_train), val_rows=len(df_val))

    pipeline = train_model(df_train, C=config.C)
    accuracy = evaluate(pipeline, df_val)
    log.info("model_evaluated", C=config.C, accuracy=round(accuracy, 4))

    save_model(pipeline)
    log.info("model_saved", path=str(config.MODEL_PATH))