FROM python:3.12-slim

# Copy uv from official image for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set UV environment variables
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy requirements file first (for layer caching)
COPY requirements_cpu.txt .

# Install CPU-only PyTorch first with specific index URL
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements_cpu.txt

# Copy application code
COPY ./app ./app
COPY run.py .

# Create necessary directories with proper permissions
RUN mkdir -p logs data/documents data/vectorstore && \
    chmod -R 777 logs data

# .env file will be provided via docker-compose volumes or environment variables
# Do not copy .env into image for security

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run.py"]
