# Start from your base image
FROM nicholasfreitas/cuda_miniconda_base:cuda12.2.0-devel-ubuntu22.04-Miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY example_run.py /app/example_run.py

# Install the most recent chai_lab package
RUN pip install git+https://github.com/chaidiscovery/chai-lab.git

# Run example python script to download weights:
RUN python example_run.py

# Set the default command to run when the container starts
CMD ["/bin/bash"]

