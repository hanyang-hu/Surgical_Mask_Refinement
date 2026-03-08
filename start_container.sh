#!/bin/bash
# ==============================================================================
# Docker Container Launcher for ECE285 Training Environment
# ==============================================================================
# 
# USAGE:
#   1. Build the Docker image:
#      docker build -t ece285_train .
#
#   2. Start the container:
#      ./start_container.sh
#
#   3. To manually exec into a running container:
#      docker exec -it changwei_285 /bin/bash
#
# ==============================================================================

# Configuration variables (edit these as needed)
IMAGE_NAME="ece285_train"
CONTAINER_NAME="changwei_285"

# Host and container directory mappings
HOST_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_PROJECT_DIR="/workspace/ece285"

HOST_DATA_DIR="${HOST_PROJECT_DIR}/dataset"
CONTAINER_DATA_DIR="/workspace/ece285/dataset"

HOST_OUTPUT_DIR="${HOST_PROJECT_DIR}/outputs"
CONTAINER_OUTPUT_DIR="/workspace/ece285/outputs"

# TensorBoard port
TENSORBOARD_PORT=6006

# Shared memory size for DataLoader workers
SHM_SIZE="16g"

# ==============================================================================
# Container Management Logic
# ==============================================================================

# Check if container is already running
if docker ps -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "Container '${CONTAINER_NAME}' is already running."
    echo "Attaching to the existing container..."
    docker exec -it "${CONTAINER_NAME}" /bin/bash
    exit 0
fi

# Check if container exists but is stopped
if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    echo "Found stopped container '${CONTAINER_NAME}'. Removing it..."
    docker rm "${CONTAINER_NAME}"
fi

# Create output directory if it doesn't exist
mkdir -p "${HOST_OUTPUT_DIR}"

# Start the container
echo "Starting new container '${CONTAINER_NAME}'..."
docker run -it \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --shm-size="${SHM_SIZE}" \
    -v "${HOST_PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
    -v "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}" \
    -v "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
    "${IMAGE_NAME}" \
    /bin/bash
