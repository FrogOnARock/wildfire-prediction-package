# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install Poetry
COPY pyproject.toml poetry.lock ./  # Copy poetry files
RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install --no-root

# Copy application files
COPY src ./src
COPY frontend ./frontend

# Expose ports
EXPOSE 8060 8501

# Command to run both FastAPI and Streamlit
CMD ["sh", "-c", "poetry run uvicorn src.main:app --host 0.0.0.0 --port 8060 & poetry run streamlit run frontend/frontend.py --server.port 8501 --server.headless true"]
