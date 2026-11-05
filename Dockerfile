FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the project (see .dockerignore to keep the image small).
COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# FastAPI and Streamlit.
EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides it for the dashboard.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]