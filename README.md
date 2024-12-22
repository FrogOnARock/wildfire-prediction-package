# Wildfire Prediction Application

The Wildfire Prediction Application is a comprehensive tool for training machine learning models to predict wildfire clusters and the number of hectares burned. The app provides an interactive interface using **Streamlit** and an API backend powered by **FastAPI**.

## Features
- **Predict Wildfire Cluster**: Classifies a wildfire into specific clusters based on environmental conditions.
- **Predict Hectares Burned**: Estimates the number of hectares affected by a wildfire.
- **Train Models**: Retrains the machine learning models on new data that is processed by using the preprocess data service. It is suggested to use preprocess data prior to training.
- **Shutdown Endpoint**: Gracefully stops the application via the Streamlit interface.

---
# Project documentation

## Repository

[GitHub Repository](https://github.com/FrogOnARock/wildfire_prediction-package)

## Requirements
- **Python**: `>=3.10`
- **Dependencies**: Managed via [Poetry](https://python-poetry.org/)
```bash
pip install poetry
```
- **Docker** (optional): For containerized deployment (not currently being utilized)

### Key Python Libraries
- **FastAPI**: Backend framework
- **Uvicorn**: ASGI server
- **Streamlit**: Interactive frontend
- **Scikit-learn**: Machine learning models
- **Pandas**: Data processing
- **OpenCV**: Image processing
- **Torch**: ANN library

## Installation and Setup

### Option 1: Run Locally (Currently the only option)

#### **Step 1: Clone the Repository**
Clone the repository and navigate to the correct directory
```bash
git clone https://github.com/FrogOnARock/wildfire_prediction-package.git
cd Wildfire Prediction
```
#### **Step 2: Install Poetry**
Install Poetry if not already installed:

```bash
pip install poetry
```

#### **Step 3: Install Dependencies**
Use Poetry to install all required packages:

```bash
poetry install
```

Step 4: Start the Application
For Windows:

```bash
run.bat
```

For Linux/MacOS:

```bash
poetry run uvicorn src.main:app --reload --port 8060 &
poetry run streamlit run frontend/frontend.py --server.port 8501
```
# Usage
## Access the Application
FastAPI Backend:
- API Docs: http://localhost:8060/docs

Streamlit Frontend:
- Interactive App: http://localhost:8501


# Features
### 1. Predict Cluster
- Navigate to the "Predict Cluster" tab in the Streamlit interface.
- Input environmental factors (temperature, precipitation, etc.).
- Click "Predict Cluster" to see the prediction.
### 2. Predict Hectares Burned
- Navigate to the "Predict Hectares Burned" tab.
- Input cluster ID and relevant conditions.
- Click "Predict Hectares" to estimate the burned area.
### 3. Train Models
- Use the "Train Models" tab to:
- Preprocess new data.
- Retrain the models.
### 4. Shutdown Application
- Use the "Shutdown" tab in Streamlit to gracefully terminate the application.

# Troubleshooting
### 1. "ModuleNotFoundError: No module named 'src'"
- Ensure you're running the application from the project root directory.
- Use poetry run to execute commands within the virtual environment.
### 2. Port Already in Use
- Stop any processes using the port (e.g., 8060 or 8501):
```bash
lsof -i :8060
kill -9 <PID>
```
### 3. Connection Errors
- Verify that both the backend (FastAPI) and frontend (Streamlit) are running.
- Check the BASE_URL in frontend/frontend.py and ensure it matches the backend's address.

# Contribution
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```
3. Commit your changes:
```bash
git commit -m "Add your feature"
```
4. Push to the branch:
```bash
git push origin feature/your-feature-name
```
5. Open a pull request.

# Contact
For questions or support, contact:

Author: Mackenzie Rock \
Email: [mac2rock@gmail.com]

### **How to Use This README**
1. Replace `<repository-url>` with the URL to your repository.
2. Update the **Contact** section with your details.
3. Add a `LICENSE` file (if needed) and update the **License** section.
