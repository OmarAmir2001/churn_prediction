import pickle

import structlog

from churn_prediction import config
from churn_prediction.logger import configure_logging
from churn_prediction.schemas import Customer

log = structlog.get_logger()


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `uv run train` first.")
    with open(path, "rb") as f_in:
        return pickle.load(f_in)


def predict(model, customer: Customer) -> float:
    customer_dict = customer.model_dump()
    return float(model.predict_proba([customer_dict])[0, 1])


def main():
    configure_logging()

    model = load_model()

    customer = Customer(
        gender="female",
        seniorcitizen=0,
        partner="yes",
        dependents="no",
        phoneservice="no",
        multiplelines="no_phone_service",
        internetservice="dsl",
        onlinesecurity="no",
        onlinebackup="no",
        deviceprotection="no",
        techsupport="no",
        streamingtv="no",
        streamingmovies="no",
        contract="month-to-month",
        paperlessbilling="yes",
        paymentmethod="electronic_check",
        tenure=1,
        monthlycharges=24.8,
        totalcharges=24.8,
    )

    probability = predict(model, customer)
    log.info("prediction_made", churn_probability=round(probability, 4))