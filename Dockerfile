# Use Python 3.12 as base image for stability
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and cleanup in one layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# Adding common data science packages and utilities
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# Switch to non-root user
USER user

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Command to run when container starts
CMD ["bash"]
