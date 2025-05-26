FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ninja-build \
    build-essential \
    gcc-11 \
    g++-11 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"
ENV FORCE_CUDA=1
ENV CUDA_ARCH="-arch=sm_75"

# Set the working directory
WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir -r requirements.txt


# Copy the application
COPY . .

# Set environment variable
ENV PYTHONPATH=/app

# Set Gradio server port
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860

CMD ["python3", "app.py"]
