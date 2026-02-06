FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

RUN rm -f /etc/apt/sources.list.d/yarn.list

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt