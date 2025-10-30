from fastapi import FastAPI
from pydantic import BaseModel
import time
import orjson
from joblib import load
import numpy as np

app = FastAPI(title="Tip20 Classifier")

class Request(BaseModel):
    hour: int
    dow: int
    month: int
    year: int
    trip_distance: float
    passenger_count: float
    PULocationID: float
    DOLocationID: float
    payment_type: float
    VendorID: float
    RatecodeID: float
    model: str = "xgb"

class Response(BaseModel):
    proba: float
    latency_ms: float

MODELS = {}

def get_model(name: str):
    if name not in MODELS:
        # lazy load default seed
        MODELS[name] = load(f"models/seed_42/{name}.joblib")
    return MODELS[name]

@app.post("/predict", response_model=Response)
def predict(req: Request):
    mdl = get_model(req.model)
    x = np.array([[req.hour, req.dow, req.month, req.year, req.trip_distance, req.passenger_count,
                   req.PULocationID, req.DOLocationID, req.payment_type, req.VendorID, req.RatecodeID]])
    start = time.time()
    proba = float(mdl.predict_proba(x)[0,1])
    latency_ms = (time.time() - start) * 1000.0
    return Response(proba=proba, latency_ms=latency_ms)
