from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Insurance Claim Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # OK for assignment/demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

preprocessor = model.named_steps["preprocess"]
classifier = model.named_steps["classifier"]

# Extract feature groups
num_features = preprocessor.transformers_[0][2]
cat_features = preprocessor.transformers_[1][2]
ohe = preprocessor.named_transformers_["cat"]


FEATURE_REASON_MAP = {
    "policy_tenure": "Short policy tenure increases claim risk",
    "population_density": "High population density increases accident exposure",
    "ncap_rating": "Lower vehicle safety rating increases injury and damage risk",
    "airbags": "Fewer airbags reduce occupant protection",
    "age_of_policyholder": "Higher driver age slightly increases claim probability",
}

# Area clusters (grouped explanation)
AREA_CLUSTER_PREFIX = "area_cluster_"


def build_feature_template():
    data = {}

    for col in num_features:
        data[col] = 0

    for col, categories in zip(cat_features, ohe.categories_):
        data[col] = categories[0]  # baseline category

    return data


class PolicyInput(BaseModel):
    policy_tenure: float
    age_of_car: float
    age_of_policyholder: int
    population_density: float

    area_cluster: str
    segment: str
    fuel_type: str
    transmission_type: str

    airbags: int
    ncap_rating: int


@app.post("/predict")
def predict_claim(data: PolicyInput):
    # Build full feature vector
    feature_data = build_feature_template()
    feature_data.update(data.dict())

    input_df = pd.DataFrame([feature_data])

    # Predict probability
    prob_claim = model.predict_proba(input_df)[0][1]

    # Threshold
    threshold = 0.4
    prediction = int(prob_claim >= threshold)

    # Risk banding
    if prob_claim >= 0.4:
        risk_level = "High"
    elif prob_claim >= 0.2:
        risk_level = "Medium"
    else:
        risk_level = "Low"


    X_transformed = preprocessor.transform(input_df)
    contributions = X_transformed.toarray()[0] * classifier.coef_[0]

    feature_names = (
        list(num_features) +
        list(ohe.get_feature_names_out(cat_features))
    )

    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions
    })

    # Keep only positive contributors
    positive_contribs = contrib_df[contrib_df["contribution"] > 0]
    positive_contribs = positive_contribs.sort_values(
        "contribution", ascending=False
    )

    reasons = []

    for feature in positive_contribs["feature"]:
        if feature.startswith(AREA_CLUSTER_PREFIX):
            reasons.append("Geographic area has historically higher claim frequency")
        elif feature in FEATURE_REASON_MAP:
            reasons.append(FEATURE_REASON_MAP[feature])

        if len(reasons) == 2:
            break


    return {
        "claim_probability": round(prob_claim, 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold_used": threshold,
        "top_risk_drivers": reasons
    }


@app.options("/predict")
def options_predict():
    return {}