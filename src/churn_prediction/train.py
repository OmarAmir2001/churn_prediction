import mlflow
import structlog
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

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


def train_model(df_train, model):
    pipeline = make_pipeline(DictVectorizer(sparse=False), model)
    pipeline.fit(to_dicts(df_train), df_train[config.TARGET].values)
    return pipeline


def evaluate(pipeline, df_val):
    y_pred = pipeline.predict_proba(to_dicts(df_val))[:, 1]
    churn_decision = y_pred >= config.THRESHOLD
    return float((churn_decision == df_val[config.TARGET].values).mean())


def run_experiment(df_train, df_val, model, params):
    with mlflow.start_run():
        pipeline = train_model(df_train, model)
        accuracy = evaluate(pipeline, df_val)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            pipeline, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

    log.info("run_logged", **params, accuracy=round(accuracy, 4))


def main():
    configure_logging()
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = clean_data(load_data())
    df_train, df_val = split_data(df)
    log.info("data_split", train_rows=len(df_train), val_rows=len(df_val))

    for C in config.C_VALUES:
        model = LogisticRegression(solver="liblinear", C=C)
        params = {"model": "logistic_regression", "C": C}
        run_experiment(df_train, df_val, model, params)

    for max_depth in config.MAX_DEPTHS:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=config.RANDOM_STATE)
        params = {"model": "decision_tree", "max_depth": max_depth}
        run_experiment(df_train, df_val, model, params)