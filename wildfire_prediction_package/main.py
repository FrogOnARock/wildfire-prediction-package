from datetime import datetime
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import uvicorn
from train import train_model  # Import external training function
from prediction_cluster import predict_cluster_model  # Import cluster prediction function
from prediction_hectare import predict_hectare_model  # Import hectare prediction function

# Define paths
KMEANS_MODEL_PATH = "models/kmeans_model.pkl"
REGRESSION_MODEL_PATH = "models/regression_model.pkl"
PREDICTIONS_PATH = "app/predictions.pkl"

app = FastAPI(
    title="Wildfire Prediction API",
    description="API for training models and predicting wildfire clusters and hectares burned",
    version="0.3.0"
)

# Input and output schemas
class ClusterInput(BaseModel):
    features: List[float]

class RegressionInput(BaseModel):
    cluster: int
    features: List[float]

class PredictionOutput(BaseModel):
    prediction: float

@app.post("/train")
def train_models():
    """Trigger the external training function to train models."""
    try:
        train_model()  # Call the function from train.py
        return {"message": "Training completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/cluster", response_model=PredictionOutput)
def predict_cluster(input_data: ClusterInput):
    """Predict wildfire cluster based on input features."""
    try:
        prediction = predict_cluster_model(input_data.features)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/hectares", response_model=PredictionOutput)
def predict_hectares(input_data: RegressionInput):
    """Predict final hectares burned based on cluster and features."""
    try:
        prediction = predict_hectare_model(input_data.cluster, input_data.features)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
