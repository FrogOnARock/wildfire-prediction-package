import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch

# Load KMeans model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
KMEANS_MODEL_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_model.pkl")
AE_PATH = os.path.join(BASE_DIR, "models", "encoder.pth")


def predict_cluster_model(features):
    """Predict wildfire cluster based on input features."""
    if not os.path.exists(KMEANS_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {KMEANS_MODEL_PATH}")

    with open(KMEANS_MODEL_PATH, "rb") as f:
        kmeans_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    ae_model = torch.load(AE_PATH)
    ae_model.eval()


    # Split numerical and categorical features
    numerical_features = np.array(features[1:8]).reshape(1, -1)  # Indices 2–7 are numerical
    categorical_features = np.hstack((features[:1],features[8:-2])).reshape(1, -1)  # Remaining features are categorical


    # Scale numerical features
    numerical_features_scaled = scaler.transform(numerical_features)

    # Combine scaled numerical features with categorical features
    combined_features = np.hstack((numerical_features_scaled, categorical_features))


    #conver to pytorch sensor
    combined_features_tensor = torch.tensor(combined_features, dtype=torch.float32)

    # Apply the autoencoder's encoder to reduce dimensions
    with torch.no_grad():
        features_reduced = ae_model(combined_features_tensor).numpy()

    # Use the reduced features for KMeans clustering
    prediction = kmeans_model.predict(features_reduced)[0]

    return prediction