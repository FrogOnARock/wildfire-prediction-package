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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KMEANS_MODEL_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR,"models", "regression_model.pkl")
PREDICTIONS_PATH = "app/predictions.pkl"

app = FastAPI(
    title="Wildfire Prediction API",
    description="API for training models and predicting wildfire clusters and hectares burned",
    version="0.3.0"
)

#configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wildfire-prediction-api")

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
        logger.info("Beginning model training...")
        train_model()  # Call the function from train.py
        logger.info("Training completed successfully")
        return {"message": "Training completed successfully."}
    except Exception as e:
        logger.info(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict/cluster", response_model=PredictionOutput)
def predict_cluster(input_data: ClusterInput):
    """Predict wildfire cluster based on input features."""
    if not os.path.exists(KMEANS_MODEL_PATH):
        logger.error("KMeans model file is missing")
        raise HTTPException(status_code=500, detail="KMeans model file missing")
    try:
        logger.info(f"Predicting cluster for features: {input_data.features}")
        prediction = predict_cluster_model(input_data.features)
        logger.info(f"Prediction: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during cluster prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during cluster prediction: {e}")

@app.post("/predict/hectares", response_model=PredictionOutput)
def predict_hectares(input_data: RegressionInput):
    """Predict final hectares burned based on cluster and features."""
    if not os.path.exists(REGRESSION_MODEL_PATH):
        logger.error("Regression model file is missing")
        raise HTTPException(status_code=500, detail="Regression model file missing")
    try:
        logger.info(f"Predicting hectares burned for features: {input_data.features}")
        prediction = predict_hectare_model(input_data.cluster, input_data.features)
        logger.info(f"Prediction: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
