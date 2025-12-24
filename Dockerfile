FROM ubuntu:20.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        psmisc \
        git \
        wget \
        && rm -rf /var/lib/apt/lists/*

# Install Miniconda (as per README instructions)
RUN curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment from environment.yml (as per README instructions)
# This matches the README: conda env create -f environment.yml
WORKDIR /workspace
COPY environment.yml /workspace/

# Create the catpred environment using environment.yml
RUN conda env create -f environment.yml

# Activate the environment for subsequent commands
SHELL ["conda", "run", "-n", "catpred", "/bin/bash", "-c"]

# Copy CatPred source code (assuming we're building from the repo)
COPY . /workspace/catpred/

# Install CatPred package in editable mode (as per README: pip install -e .)
WORKDIR /workspace/catpred
RUN pip install -e .

# Set working directory
WORKDIR /workspace
