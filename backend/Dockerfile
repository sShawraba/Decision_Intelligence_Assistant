# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy pyproject.toml
COPY backend/pyproject.toml .

# Install Python dependencies using uv
RUN uv venv && uv pip install -r requirements.txt 2>/dev/null || uv pip install fastapi uvicorn pydantic chromadb openai scikit-learn

# Copy application code
COPY backend/app ./app

# Create necessary directories
RUN mkdir -p chroma_data logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
