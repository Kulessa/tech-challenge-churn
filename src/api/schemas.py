"""Pydantic schemas for churn API requests and responses."""

from typing import Literal

from pydantic import BaseModel, Field

Gender = Literal["Female", "Male"]
YesNo = Literal["No", "Yes"]
MultipleLinesValue = Literal["No", "No phone service", "Yes"]
InternetServiceValue = Literal["DSL", "Fiber optic", "No"]
InternetOption = Literal["No", "No internet service", "Yes"]
ContractValue = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodValue = Literal[
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]
SeniorCitizenValue = Literal[0, 1]


class CustomerData(BaseModel):
    """Customer attributes accepted by the churn prediction endpoint."""

    gender: Gender = Field(description="Gender as present in the training dataset.")
    SeniorCitizen: SeniorCitizenValue = Field(
        description="Whether the customer is a senior citizen: 0 or 1."
    )
    Partner: YesNo = Field(description="Whether the customer has a partner.")
    Dependents: YesNo = Field(description="Whether the customer has dependents.")
    tenure: int = Field(ge=0, description="Customer tenure in months.")
    PhoneService: YesNo = Field(description="Whether phone service is enabled.")
    MultipleLines: MultipleLinesValue = Field(
        description="Multiple-line service status."
    )
    InternetService: InternetServiceValue = Field(description="Internet service type.")
    OnlineSecurity: InternetOption = Field(description="Online security service status.")
    OnlineBackup: InternetOption = Field(description="Online backup service status.")
    DeviceProtection: InternetOption = Field(
        description="Device protection service status."
    )
    TechSupport: InternetOption = Field(description="Technical support service status.")
    StreamingTV: InternetOption = Field(description="Streaming TV service status.")
    StreamingMovies: InternetOption = Field(
        description="Streaming movies service status."
    )
    Contract: ContractValue = Field(description="Customer contract term.")
    PaperlessBilling: YesNo = Field(description="Whether paperless billing is enabled.")
    PaymentMethod: PaymentMethodValue = Field(
        description="Customer payment method."
    )
    MonthlyCharges: float = Field(ge=0, description="Monthly charge amount.")
    TotalCharges: float | None = Field(
        default=None,
        ge=0,
        description="Accumulated charge amount; null values are imputed by the model.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 85.5,
                    "TotalCharges": 1026.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response containing the churn score, decision, and model version."""

    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: bool
    threshold: float = Field(ge=0, le=1)
    model_version: str
