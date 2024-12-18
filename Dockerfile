# Base image
FROM python:3.8-slim AS base

# Set working directory
WORKDIR /app

# Install Poetry (or switch to requirements.txt if needed)
RUN pip install --no-cache-dir poetry

# Copy Poetry files and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-root

# Stage 1: Train Model
FROM base AS train

# Copy necessary files for training
COPY ./data ./data
COPY ./wildfire_prediction_package ./wildfire_prediction_package
RUN mkdir -p models  # Create the models directory
RUN poetry run python wildfire_prediction_package/train.py

# Stage 2: Serve Application
FROM base AS serve

# Copy trained models from the training stage
COPY --from=train /app/models /app/models

# Copy the application code
COPY ./wildfire_prediction_package ./wildfire_prediction_package

# Expose FastAPI port
EXPOSE 8080

# Command to run the FastAPI application
CMD ["poetry", "run", "uvicorn", "wildfire_prediction_package.main:app", "--host", "0.0.0.0", "--port", "8080"]

