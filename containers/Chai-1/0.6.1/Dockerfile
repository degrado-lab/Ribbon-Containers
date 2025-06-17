# Start from your base image
FROM nicholasfreitas/cuda_miniconda_base:cuda12.2.0-devel-ubuntu22.04-Miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY download_weights.py /app/download_weights.py

# Ins
tall the most recent chai_lab package
ENV CHAI_DOWNLOADS_DIR=/app/.cache/chai-downloads
RUN mkdir -p $CHAI_DOWNLOADS_DIR
RUN pip install git+https://github.com/chaidiscovery/chai-lab.git

# Run example python script to download weights:
RUN python download_weights.py

# Set the default command to run when the container starts
CMD ["/bin/bash"]

