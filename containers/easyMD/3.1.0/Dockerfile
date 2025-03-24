# Use the existing base image
FROM nicholasfreitas/cuda_miniconda_base:cuda12.2.0-devel-ubuntu22.04-Miniconda3

# Set up environment variables
ENV CONDA_ENV=easymd
ENV WORKDIR=/workspace/easyMD

# Set the working directory
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \ 
    && rm -rf /var/lib/apt/lists/*

# Clone the LigandMPNN repository
RUN git clone https://github.com/degrado-lab/easyMD.git

# Change directory and download model parameters
WORKDIR $WORKDIR

# Copy the environment.yml file into the container:
COPY environment.yml .

# Create a Conda environment and install dependencies
RUN conda env create -f environment.yml -y
RUN echo "source activate $CONDA_ENV" >> ~/.bashrc && \
    /bin/bash -c "source activate $CONDA_ENV && pip install ."

# Set Conda environment to activate on container start
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source activate $CONDA_ENV && exec bash"]