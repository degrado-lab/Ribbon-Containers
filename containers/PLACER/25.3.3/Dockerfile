# Use Ubuntu 20.04 as base image
FROM nicholasfreitas/cuda_miniconda_base:cuda12.2.0-devel-ubuntu22.04-Miniconda3

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages: wget, git, bzip2 (for conda installer), and clean up afterward
RUN apt-get update && \
    apt-get install -y wget git bzip2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory to /opt
WORKDIR /opt

# Clone the PLACER repository
RUN git clone https://github.com/baker-laboratory/PLACER.git

# Change working directory to the repository
WORKDIR /opt/PLACER

# Create the conda environment from the provided environment.yml file
RUN conda env create -f envs/placer_env.yml

# Add the new environment's bin directory to PATH so that its executables are used by default
ENV PATH="/opt/miniconda/envs/RaptorXSingle/bin:${PATH}"

# Set Conda environment to activate on container start
ENTRYPOINT ["/bin/bash", "-c", "source /usr/etc/profile.d/conda.sh && conda activate placer_env && exec \"$@\"", "--"]