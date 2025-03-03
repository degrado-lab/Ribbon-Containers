# Use the base image nvidia/cuda:12.5.0-devel-ubuntu22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install necessary packages
RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -u -p /usr && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set Miniconda path
ENV PATH="/usr/bin:/usr/condabin:/usr:$PATH"

# Install Python 3.11 via Conda
RUN conda install python==3.11 -y

# Cleanup conda cache to reduce image size
RUN conda clean -afy

# Install git
RUN apt-get update && \
    apt-get install -y git

# Default command to start the container
CMD ["/bin/bash"]
