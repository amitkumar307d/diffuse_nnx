FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/diffuse_nnx

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Copy the rest of the codebase
COPY . .

# Install the diffuse_nnx package
RUN pip install --no-cache-dir -e .

# Copy Inception weights to the expected path inside the image
RUN mkdir -p /root/jmt/eval/
COPY eval/inception_v3_weights_fid.pickle /root/jmt/eval/inception_v3_weights_fid.pickle

# Copy VAE checkpoint to the expected path inside the image
RUN mkdir -p /root/jmt/networks/encoders/
# We expect to find vae_trial1.pkl in the build context (we will copy it there)
COPY vae_trial1.pkl /root/jmt/networks/encoders/vae_trial1.pkl

# Set environment variables for TPU/JAX
ENV JAX_PLATFORMS=tpu,cpu
ENV JAX_LOG_LEVEL=INFO
