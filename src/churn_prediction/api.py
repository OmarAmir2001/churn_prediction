from fastapi import FastAPI

from churn_prediction import config
from churn_prediction.logger import configure_logging
from churn_prediction.predict import load_model, predict
from churn_prediction.schemas import Customer, Prediction

configure_logging()
model = load_model()

app = FastAPI(title="Churn prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: Customer) -> Prediction:
    probability = predict(model, customer)
    return Prediction(
        churn_probability=round(probability, 4),
        churn=probability >= config.THRESHOLD,
    )