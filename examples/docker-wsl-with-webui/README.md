# TripoSG Docker WebUI Setup Guide

This guide explains how to set up and run TripoSG with a web interface using Docker in WSL2 environment.

## Prerequisites
- Docker installed in WSL2
- WSL2 installed and configured 
  - **WSL distro is OFF**
- NVIDIA Container Toolkit installed
- Set up `nvidia-smi`
- CUDA-compatible GPU with at least 8GB VRAM

## Setup Instructions

1. Copy the necessary files to the project root directory:

```bash
rsync -aP examples/docker-wsl-with-webui/ ./
```

2. Build the Docker image in the project root directory:

```bash
docker-compose build
docker-compose up -d
```

The container will be accessible at `http://localhost:7860` once it's running.

## Configuration Details

### Dockerfile Configuration

The Dockerfile includes:
- CUDA 11.8.0 with cuDNN 8 support
- Python 3.10
- Required system libraries for OpenCV
- PyTorch with CUDA support
- All necessary Python dependencies

### Environment Variables

The following environment variables are configured:
- `CUDA_HOME`: CUDA installation path
- `TORCH_CUDA_ARCH_LIST`: CUDA architecture list
- `NVIDIA_VISIBLE_DEVICES`: GPU visibility settings
- `NVIDIA_DRIVER_CAPABILITIES`: GPU capabilities
- `GRADIO_SERVER_PORT`: WebUI port (7860)
- `GRADIO_SERVER_NAME`: Server binding address (0.0.0.0)

### Volume Mounts

The following directories are mounted:
- `./:/app`: Application code
- `./pretrained_weights:/app/pretrained_weights`: Model weights

## Usage

1. Access the WebUI at `http://localhost:7860` using Gradio
2. Upload an image or use the scribble interface
3. Configure generation parameters
4. Click "Generate" to create 3D models
