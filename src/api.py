# src/api.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime
import json

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_xgb.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_names.joblib")
LOG_PATH = os.path.join(BASE_DIR, "..", "models", "predictions.csv")

# load model
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

app = FastAPI(title="Fraud Detection API (demo)")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple API key check (set API_KEY env var)
API_KEY = os.environ.get("API_KEY", None)

class Txn(BaseModel):
    features: Dict[str, float]
    threshold: Optional[float] = 0.5  # classification threshold

@app.post("/predict")
async def predict(txn: Txn, x_api_key: Optional[str] = Header(None)):
    # basic API key auth
    if API_KEY:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ensure feature set complete
    missing = [f for f in feature_names if f not in txn.features]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # create DataFrame aligned with training features
    X = pd.DataFrame([{f: float(txn.features[f]) for f in feature_names}])
    prob = float(model.predict_proba(X)[:, 1][0])
    flag = prob >= float(txn.threshold)

    # append to log
    log_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "features": json.dumps(txn.features),
        "fraud_probability": prob,
        "fraud_flag": bool(flag)
    }
    df_log = pd.DataFrame([log_row])
    if os.path.exists(LOG_PATH):
        df_log.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        df_log.to_csv(LOG_PATH, index=False)

    return {"fraud_probability": prob, "fraud_flag": bool(flag)}

@app.get("/health")
async def health():
    return {"status": "ok"}
