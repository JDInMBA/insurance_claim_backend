from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Insurance Claim Prediction API")

# CORS configuration (required for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins (OK for assignment/demo)
    allow_credentials=True,
    allow_methods=["*"],          # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)

# -----------------------------
# Load trained model
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

preprocessor = model.named_steps["preprocess"]

# Extract feature groups from preprocessing pipeline
num_features = preprocessor.transformers_[0][2]
cat_features = preprocessor.transformers_[1][2]
ohe = preprocessor.named_transformers_["cat"]

# -----------------------------
# Helper: build full feature template
# -----------------------------
def build_feature_template():
    data = {}

    # Numerical defaults
    for col in num_features:
        data[col] = 0

    # Categorical defaults (baseline category)
    for col, categories in zip(cat_features, ohe.categories_):
        data[col] = categories[0]

    return data

# -----------------------------
# Request schema
# -----------------------------
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

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_claim(data: PolicyInput):
    # Build full feature vector
    feature_data = build_feature_template()

    # Override defaults with user input
    feature_data.update(data.dict())

    # Convert to DataFrame
    input_df = pd.DataFrame([feature_data])

    # Predict probability
    prob_claim = model.predict_proba(input_df)[0][1]

    # Business threshold
    threshold = 0.4
    prediction = int(prob_claim >= threshold)

    # Risk banding
    if prob_claim >= 0.4:
        risk_level = "High"
    elif prob_claim >= 0.2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "claim_probability": round(prob_claim, 4),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold_used": threshold
    }

# -----------------------------
# Explicit OPTIONS handler (CRITICAL FIX)
# -----------------------------
@app.options("/predict")
def options_predict():
    return {}