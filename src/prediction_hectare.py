import pickle
import os
import numpy as np

# Load Regression model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR, "models", "regression_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_model.pkl")




def predict_hectare_model(features):
    """Predict final hectares burned based on cluster and input features."""
    if not os.path.exists(REGRESSION_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {REGRESSION_MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {REGRESSION_MODEL_PATH}")

    with open(REGRESSION_MODEL_PATH, "rb") as f:
        regression_model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler_model = pickle.load(f)

    # Split numerical and categorical features
    numerical_features = np.array(features[1:8]).reshape(1, -1)  # Indices 2–7 are numerical
    categorical_features = np.hstack((features[:1], features[8:])).reshape(1, -1)  # Remaining features are categorical

    # Scale numerical features
    numerical_features_scaled = scaler_model.transform(numerical_features)

    # Combine cluster and features
    input_features = np.hstack((numerical_features_scaled, categorical_features))

    prediction = regression_model.predict(input_features)[0]

    return prediction