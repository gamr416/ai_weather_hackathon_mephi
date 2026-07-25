# syntax=docker/dockerfile:1
# Residual-FSQ ERA5 codec — expert evaluation image (CUDA optional via nvidia-container-toolkit)

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# System deps for scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libeccodes0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
# torch already in base image — install the rest
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . /workspace

ENV PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1

# Default: show expert help
CMD ["bash", "scripts/docker_entrypoint.sh"]
