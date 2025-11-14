# Base image with Python
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps for audio / ffmpeg
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy repo
COPY . /app

# Install python deps
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose MLflow port
EXPOSE 5000

# Default command will be to show help. Override when running container.
CMD ["python", "tools/orchestrate_training.py", "--help"]
