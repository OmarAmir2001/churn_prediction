from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["yes", "no"]
InternetExtra = Literal["yes", "no", "no_internet_service"]


class Customer(BaseModel):
    gender: Literal["female", "male"]
    seniorcitizen: Literal[0, 1]
    partner: YesNo
    dependents: YesNo
    phoneservice: YesNo
    multiplelines: Literal["yes", "no", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: InternetExtra
    onlinebackup: InternetExtra
    deviceprotection: InternetExtra
    techsupport: InternetExtra
    streamingtv: InternetExtra
    streamingmovies: InternetExtra
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: YesNo
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(ge=0)
    monthlycharges: float = Field(gt=0)
    totalcharges: float = Field(ge=0)