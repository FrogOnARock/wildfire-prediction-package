from datetime import datetime
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import pickle
from src.train import train_model  # Import external training function
from src.prediction_cluster import predict_cluster_model  # Import cluster prediction function
from src.prediction_hectare import predict_hectare_model  # Import hectare prediction function
from src.preprocess import  preprocess_model
from fastapi.requests import Request
from fastapi import BackgroundTasks
import numpy as np

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
KMEANS_MODEL_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_model.pkl")
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR, "models", "regression_model.pkl")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "predictions", "predictions.pkl")

app = FastAPI(
    title="Wildfire Prediction API",
    description="API for training models and predicting wildfire clusters and hectares burned",
    version="0.1.0",
)

#configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wildfire-prediction-api")

@app.on_event("startup")
def startup_event():
    logger.info("Application has started succesfully!")
    for route in app.routes:
        logger.info(f"{route.path} - {route.name}")



@app.get("/", summary="Root Endpoint")
def read_root():
    logger.info("Root endpoint Accessed.")
    """Root endpoint for the Wildfire Prediction API."""
    return {
        "message": "Welcome to the Wildfire Prediction API!",
        "usage": {
            "Preprocessing Data": "/preprocess (POST)",
            "Train Models": "/train (POST)",
            "Predict Wildfire Cluster": "/predict/cluster (POST)",
            "Predict Hectares Burned": "/predict/hectares (POST)"
        },
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request URL: {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response



# Input and output schemas
class ClusterInput(BaseModel):
    national_park: str = Field(..., description="Yes/No for National Park.")
    temp_avg: float = Field(..., description="Temperature Average (7-days).")
    temp_min: float = Field(..., description="Temperature Minimum (7-days).")
    temp_max: float = Field(..., description="Temperature Maximum (7-days).")
    precip: float = Field(..., description="Precipitation Average (7-days).")
    windspeed: float = Field(..., description="Windspeed Average (7-days).")
    max_wind_gust: float = Field(..., description="Maximum Wind Gust Average (7-days).")
    pressure: float = Field(..., description="Pressure Average (7-days).")
    cause: str = Field(..., description="H, N, H-PB, or U.")
    region: int = Field(..., description="Region identifier.")
    response: str = Field(..., description="FUL, MOD, or MON.")
    month: int = Field(..., description="Month (1–12).")

    class Config:
        schema_extra = {
            "example": {
                "national_park": "Y",
                "temp_avg": 25.0,
                "temp_min": 20.0,
                "temp_max": 30.0,
                "precip": 5.0,
                "windspeed": 10.0,
                "max_wind_gust": 50.0,
                "pressure": 1015.0,
                "cause": "N",
                "region": 1,
                "response": "MOD",
                "month": 7
            }
        }

class RegressionInput(BaseModel):
    cluster: int
    national_park: str = Field(..., description="Yes/No for National Park.")
    temp_avg: float = Field(..., description="Temperature Average (7-days).")
    temp_min: float = Field(..., description="Temperature Minimum (7-days).")
    temp_max: float = Field(..., description="Temperature Maximum (7-days).")
    precip: float = Field(..., description="Precipitation Average (7-days).")
    windspeed: float = Field(..., description="Windspeed Average (7-days).")
    max_wind_gust: float = Field(..., description="Maximum Wind Gust Average (7-days).")
    pressure: float = Field(..., description="Pressure Average (7-days).")
    cause: str = Field(..., description="H, N, H-PB, or U.")
    region: int = Field(..., description="Region identifier.")
    response: str = Field(..., description="FUL, MOD, or MON.")
    month: int = Field(..., description="Month (1–12).")

    class Config:
        schema_extra = {
            "example": {
                "cluster": 2,
                "national_park": "Y",
                "temp_avg": 25.0,
                "temp_min": 20.0,
                "temp_max": 30.0,
                "precip": 5.0,
                "windspeed": 10.0,
                "max_wind_gust": 50.0,
                "pressure": 1015.0,
                "cause": "N",
                "region": 1,
                "response": "MOD",
                "month": 7
            }
        }

class PredictionOutput(BaseModel):
    prediction: float

def transform_inputs(input_data):
    try:
        # Define column names and initialize feature vector
        column_names = [
            'NAT_PARK BINARY', 'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'wpgt', 'pres',
            'CAUSE_H-PB', 'CAUSE_N', 'CAUSE_U',
            'Region_Region 1', 'Region_Region 11', 'Region_Region 12', 'Region_Region 13',
            'Region_Region 2', 'Region_Region 3', 'Region_Region 4', 'Region_Region 5', 'Region_Region 6',
            'Region_Region 7', 'Region_Region 8',
            'RESPONSE_MOD', 'RESPONSE_MON',
            'MONTH_2', 'MONTH_3', 'MONTH_4', 'MONTH_5', 'MONTH_6',
            'MONTH_7', 'MONTH_8', 'MONTH_9', 'MONTH_10', 'MONTH_11', 'MONTH_12', 'cluster_1', 'cluster_2'
        ]
        feature_vector = np.zeros(len(column_names))
        column_index = {col: idx for idx, col in enumerate(column_names)}

        # Validate and map inputs
        # 1. NAT_PARK BINARY
        if input_data.national_park.upper() not in ["Y", "N"]:
            raise ValueError("NAT_PARK BINARY must be 'Y' or 'N'")
        feature_vector[column_index["NAT_PARK BINARY"]] = 1 if input_data.national_park.upper() == "Y" else 0

        # 2. tavg, tmin, tmax
        for field, name in [(input_data.temp_avg, "tavg"), (input_data.temp_min, "tmin"), (input_data.temp_max, "tmax")]:
            if not (-40 <= field <= 40):
                raise ValueError(f"{name} must be between -40 and 40")
            feature_vector[column_index[name]] = field

        # 3. prcp
        if not (0 <= input_data.precip <= 50):
            raise ValueError("prcp must be between 0 and 50")
        feature_vector[column_index["prcp"]] = input_data.precip

        # 4. wspd
        if not (0 <= input_data.windspeed <= 80):
            raise ValueError("wspd must be between 0 and 80")
        feature_vector[column_index["wspd"]] = input_data.windspeed

        # 5. wpgt
        if not (0 <= input_data.max_wind_gust <= 400):
            raise ValueError("wpgt must be between 0 and 400")
        feature_vector[column_index["wpgt"]] = input_data.max_wind_gust

        # 6. pres
        if not (980 <= input_data.pressure <= 1100):
            raise ValueError("pres must be between 980 and 1100")
        feature_vector[column_index["pres"]] = input_data.pressure

        # 7. cause
        cause_mapping = {"H": None, "H-PB": "CAUSE_H-PB", "N": "CAUSE_N", "U": "CAUSE_U"}
        if input_data.cause not in cause_mapping:
            raise ValueError("cause must be one of 'H', 'N', 'H-PB', or 'U'")
        if input_data.cause != "H":
            cause_col = cause_mapping[input_data.cause]
            feature_vector[column_index[cause_col]] = 1

        # 8. region
        if not (1 <= input_data.region <= 13):
            raise ValueError("region must be between 1 and 13")
        if input_data.region != 9:
            region_col = f"Region_Region {input_data.region}"
            feature_vector[column_index[region_col]] = 1

        # 9. response
        response_mapping = {"MON": "RESPONSE_MON", "MOD": "RESPONSE_MOD", "FUL": None}
        if input_data.response not in response_mapping:
            raise ValueError("response must be one of 'MON', 'MOD', or 'FUL'")
        if input_data.response != "FUL":
            response_col = response_mapping[input_data.response]
            feature_vector[column_index[response_col]] = 1

        # 10. month
        if not (1 <= input_data.month <= 12):
            raise ValueError("month must be between 1 and 12")
        if input_data.month != 1:
            month_col = f"MONTH_{input_data.month}"
            feature_vector[column_index[month_col]] = 1

        # 11. cluster
        try:
            if not (0 <= input_data.cluster <= 2):
                raise ValueError("cluster must be between 0 and 2")

            cluster_col = f"cluster_{input_data.cluster}"
            if cluster_col in column_index and input_data.cluster != 0:
                feature_vector[column_index[cluster_col]] = 1

        except Exception as e:
            feature_vector[column_index['cluster_1']] = 0

        return feature_vector

    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")

        raise HTTPException(status_code=400,
            detail={
                "error": "Invalid input data",
                "message": str(e),
                "hint": "Please ensure all inputs are within the expected range."
                   }
            )


@app.post("/preprocess")
def train_models():
    """Trigger the external training function to train models."""
    try:
        logger.info("Beginning data preprocessing...")
        preprocess_model()
        logger.info("Preprocessing completed successfully")
        return {"message": "Preprocessing completed successfully."}

    except Exception as e:
        logger.info(f"Error during preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@app.post("/train")
def train_models():
    """Trigger the external training function to train models."""
    try:
        logger.info("Beginning model training...")
        train_model()
        logger.info("Training completed successfully")
        return {"message": "Training completed successfully."}

    except Exception as e:
        logger.info(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict/cluster", response_model=PredictionOutput)
def predict_cluster(input_data: ClusterInput):
    """Predict wildfire cluster based on input features."""
    features = transform_inputs(input_data)

    if not os.path.exists(KMEANS_MODEL_PATH):
        logger.error("KMeans model file is missing")
        raise HTTPException(status_code=500, detail="KMeans model file missing")
    try:
        logger.info(f"Predicting cluster for features: {features}")
        prediction = predict_cluster_model(features)
        logger.info(f"Prediction: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during cluster prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during cluster prediction: {e}")

@app.post("/predict/hectares", response_model=PredictionOutput)
def predict_hectares(input_data: RegressionInput):

    features = transform_inputs(input_data)

    """Predict final hectares burned based on cluster and features."""
    if not os.path.exists(REGRESSION_MODEL_PATH):
        logger.error("Regression model file is missing")
        raise HTTPException(status_code=500, detail="Regression model file missing")
    try:
        logger.info(f"Predicting hectares burned for features: {features}")
        prediction = predict_hectare_model(features)
        logger.info(f"Prediction: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wildfire_prediction_package.main:app", port=8060, reload=True)
