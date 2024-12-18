# Dockerfile
FROM python:3.8-slim AS base

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage 1: Train Model
FROM base AS train
COPY ./data ./data
COPY wildfire_prediction_package/train.py ./train.py
COPY ./models ./models
RUN python train.py


# Stage 2: Serve Application
FROM base AS serve
COPY --from=train /app/models /app/models
COPY wildfire_prediction_package/main.py ./main.py
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

