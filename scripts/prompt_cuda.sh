#!/bin/bash
# Script to prompt user for CUDA support preference and directly build with the appropriate Dockerfile

echo "=== CUDA Support Selection ==="
echo ""
echo "Do you want to build with NVIDIA GPU (CUDA) support?"
echo "This requires an NVIDIA GPU and proper NVIDIA Docker runtime configuration."
echo ""
read -p "Build with CUDA support? (y/n): " choice

case "$choice" in
  y|Y|yes|YES|Yes )
    echo "Selected: Build WITH CUDA support"
    
    # Create or update .env file with the Dockerfile selection
    if [ -f .env ]; then
      # Update existing file
      if grep -q "DOCKER_BACKEND_DOCKERFILE" .env; then
        sed -i 's/^DOCKER_BACKEND_DOCKERFILE=.*/DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda/' .env
      else
        # Add a newline before appending new content
        echo "" >> .env
        echo "DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda" >> .env
      fi
    else
      # Create new file
      echo "DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend.cuda" > .env
    fi
    
    # Create a flag file to indicate GPU use
    echo "GPU" > .gpu_selected
    
    echo "Environment set to build with CUDA support"
    ;;
  * )
    echo "Selected: Build WITHOUT CUDA support (CPU only)"
    
    # Create or update .env file with the Dockerfile selection
    if [ -f .env ]; then
      # Update existing file
      if grep -q "DOCKER_BACKEND_DOCKERFILE" .env; then
        sed -i 's/^DOCKER_BACKEND_DOCKERFILE=.*/DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend/' .env
      else
        # Add a newline before appending new content
        echo "" >> .env
        echo "DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend" >> .env
      fi
    else
      # Create new file
      echo "DOCKER_BACKEND_DOCKERFILE=Dockerfile.backend" > .env
    fi
    
    # Remove any GPU flag file if it exists
    if [ -f .gpu_selected ]; then
      rm .gpu_selected
    fi
    
    echo "Environment set to build without CUDA support"
    ;;
esac

echo "=== CUDA Selection Complete ==="