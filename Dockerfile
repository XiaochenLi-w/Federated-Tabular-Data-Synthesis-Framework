# Use Python 3.11 base image
FROM python:3.11-slim

# Install system dependencies and zsh
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . .
