FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install module dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && mkdir -p /outputs

# Copy source code
COPY src ./src

# Copy pre-downloaded models into the container
COPY models ./models

# Set environment variables
ENV HF_HOME=/models \
TRANSFORMERS_OFFLINE=1

# Set entrypoint
ENTRYPOINT ["python", "/workspace/src/run_inference.py"]
