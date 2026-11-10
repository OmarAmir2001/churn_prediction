import mlflow
import structlog
from mlflow import MlflowClient

from churn_prediction import config
from churn_prediction.logger import configure_logging

log = structlog.get_logger()


def find_best_model(experiment_id):
    models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
        max_results=1,
        output_format="list",
    )
    if not models:
        raise RuntimeError("No trained models found. Run `uv run train` first.")
    return models[0]


def main():
    configure_logging()
    mlflow.set_tracking_uri(config.TRACKING_URI)
    experiment = mlflow.set_experiment(config.EXPERIMENT_NAME)

    best = find_best_model(experiment.experiment_id)
    log.info("best_model_found", model_id=best.model_id, **best.params)

    version = mlflow.register_model(best.model_uri, config.MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        config.MODEL_NAME, config.MODEL_ALIAS, version.version
    )
    log.info(
        "model_registered",
        name=config.MODEL_NAME,
        version=version.version,
        alias=config.MODEL_ALIAS,
    )