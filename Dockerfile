# Use Python 3.9 base image
FROM arm64v8/python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables to work around hash issues
ENV DEBIAN_FRONTEND=noninteractive
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

RUN apt-get update --option Acquire::Retries=5 && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    ffmpeg \
    wget \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY preprocess.py .

# Create necessary directories
RUN mkdir -p /data/input/stroke_data /data/input/no_stroke_data /data/output

# Set environment variable for reproducible Python behavior
ENV PYTHONUNBUFFERED=1

# Define default command
ENTRYPOINT ["python", "preprocess.py"]
