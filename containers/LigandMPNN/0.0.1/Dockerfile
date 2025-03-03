# Use the existing base image
FROM nicholasfreitas/cuda_miniconda_base:cuda12.2.0-devel-ubuntu22.04-Miniconda3

# Set up environment variables
ENV CONDA_ENV=ligandmpnn_env
ENV WORKDIR=/workspace/LigandMPNN

# Set the working directory
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \ 
    && rm -rf /var/lib/apt/lists/*

# Clone the LigandMPNN repository
RUN git clone https://github.com/dauparas/LigandMPNN.git

# Change directory and download model parameters
WORKDIR $WORKDIR
RUN bash get_model_params.sh "./model_params"

# Create a Conda environment and install dependencies
RUN conda create -y -n $CONDA_ENV python=3.11 && \
    echo "source activate $CONDA_ENV" >> ~/.bashrc && \
    /bin/bash -c "source activate $CONDA_ENV && pip install -r requirements.txt"

# Set Conda environment to activate on container start
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source activate $CONDA_ENV && exec bash"]