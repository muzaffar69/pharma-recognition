FROM registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6

# Set working directory
WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/data/templates/packages \
    /app/data/templates/sheets \
    /app/data/roi_mappings/packages \
    /app/data/roi_mappings/sheets

# Environment variables for GPU optimization
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
ENV CUDA_VISIBLE_DEVICES=0
ENV FLAGS_fraction_of_gpu_memory_to_use=0.8
ENV FLAGS_cudnn_exhaustive_search=1
ENV FLAGS_cudnn_deterministic=0
ENV FLAGS_use_cudnn=1

# Set thread affinity for ARM CPU optimization
ENV OMP_NUM_THREADS=6
ENV OMP_WAIT_POLICY=ACTIVE
ENV KMP_AFFINITY=granularity=fine,compact,1,0

# Entry point
ENTRYPOINT ["python3", "src/main.py"]
