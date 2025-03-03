# Start from your base image
FROM nicholasfreitas/cuda_miniconda_base:cuda12.2.0-devel-ubuntu22.04-Miniconda3

# Set the working directory inside the container
WORKDIR /app

# Install the most recent boltz-1 package
RUN git clone https://github.com/jwohlwend/boltz.git
RUN cd boltz; pip install -e .

# Set the default command to run when the container starts
CMD ["/bin/bash"]