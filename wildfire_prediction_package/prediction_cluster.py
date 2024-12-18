import pickle
import os

# Load KMeans model
KMEANS_MODEL_PATH = "models/kmeans_model.pkl"


def predict_cluster_model(features):
    """Predict wildfire cluster based on input features."""
    if not os.path.exists(KMEANS_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {KMEANS_MODEL_PATH}")

    with open(KMEANS_MODEL_PATH, "rb") as f:
        kmeans_model = pickle.load(f)

    # Reshape features for prediction
    features = [features]
    prediction = kmeans_model.predict(features)[0]
    return prediction