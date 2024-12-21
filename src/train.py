# train.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from meteostat import Point, Daily
import cv2
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.svm import SVR
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import logging

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


# Configure logger
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

logger = logging.getLogger(__name__)

# Define the PyTorch Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Load and preprocess dataset
def train_model():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")

    fire_data_path = os.path.join(DATA_DIR, "processed_data.csv")
    fire_data_encoded = pd.read_csv(fire_data_path, sep=",", header=0)

    X = fire_data_encoded.drop(columns='SIZE_HA')

    logger.info("Beginning auto-encoding...")

    # Prepare data for PyTorch
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # PyTorch Autoencoder Parameters
    input_dim = X.shape[1]
    encoding_dim = 2

    # Initialize PyTorch Autoencoder
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the PyTorch Autoencoder
    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = batch[0]  # Extract the data from TensorDataset
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()

    # Use the trained encoder for dimensionality reduction
    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(X_tensor).numpy()

    logger.info("Auto-encoding complete")
    logger.info("Beginning clustering...")

    # Final KMeans clustering on reduced dimensions
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(encoded_data)

    fire_data_encoded['cluster'] = kmeans.labels_

    logger.info("Clustering complete")

    fire_data_regression = fire_data_encoded.copy()


    # Set predictors and target
    X = fire_data_encoded.drop(columns='SIZE_HA')
    y = fire_data_regression['SIZE_HA']

    logger.info("Applying regression...")

    svr_best = SVR(
        kernel='rbf',
        C=0.8911151541387614,
        epsilon=0.08156308736323525
    )

    # Fit the SVR model
    svr_best.fit(X, y)

    logger.info("Model fit, saving models and predictions...")

    #make directory for models, fail safe to ensure even if not created using Docker it is here.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assignment_2_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(assignment_2_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

# Save models
    with open(os.path.join(models_dir, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(models_dir, "regression_model.pkl"), "wb") as f:
        pickle.dump(svr_best, f)
    torch.save(model.encoder, os.path.join(models_dir, "encoder.pth")) #save py torch model

    # Save predictions

    predictions_dir = os.path.join(assignment_2_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    predictions = pd.DataFrame({"actual": y, "predicted": svr_best.predict(X)})
    predictions.to_pickle(os.path.join(predictions_dir, "predictions.pkl"))


if __name__ == "__main__":
    train_model()
