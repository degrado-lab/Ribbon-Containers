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

# Clone the RaptorX-Single repository
RUN git clone https://github.com/AndersJing/RaptorX-Single.git

# Change working directory to the repository
WORKDIR /opt/RaptorX-Single

# Create the conda environment from the provided environment.yml file
RUN conda env create -f environment.yml

# Add the new environment's bin directory to PATH so that its executables are used by default
ENV PATH="/opt/miniconda/envs/RaptorXSingle/bin:${PATH}"

# Download the parameter files into a new folder 'params'
RUN mkdir params && \
    wget -P params/ https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt && \
    wget -P params/ https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt && \
    wget -P params/ https://zenodo.org/record/4644188/files/prot_t5_xl_uniref50.zip && \
    wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1b.pt && \
    wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1v.pt && \
    wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ProtTrans.pt && \
    wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1b-ESM1v-ProtTrans.pt && \
    wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1b-Ab.pt && \
    wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1v-Ab.pt && \
    wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ProtTrans-Ab.pt && \
    wget -P params/ https://zenodo.org/records/7351378/files/RaptorX-Single-ESM1b-ESM1v-ProtTrans-Ab.pt

# Set Conda environment to activate on container start
#SHELL ["/bin/bash", "-c"]
#ENTRYPOINT ["/bin/bash", "-c", "source activate RaptorX-Single"]
ENTRYPOINT ["/bin/bash", "-c", "source /usr/etc/profile.d/conda.sh && conda activate RaptorX-Single && exec \"$@\"", "--"]
# Set default command to show the usage help message of pred.py.
# (You can override this CMD with a different command when running the container.)
#CMD ["python", "pred.py", "--help"] /usr/envs/RaptorX-Single