import pickle
import os

# Load Regression model
REGRESSION_MODEL_PATH = "models/regression_model.pkl"


def predict_hectare_model(cluster, features):
    """Predict final hectares burned based on cluster and input features."""
    if not os.path.exists(REGRESSION_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {REGRESSION_MODEL_PATH}")

    with open(REGRESSION_MODEL_PATH, "rb") as f:
        regression_model = pickle.load(f)

    # Combine cluster and features
    input_features = [cluster] + features
    input_features = [input_features]  # Reshape for prediction
    prediction = regression_model.predict(input_features)[0]
    return prediction