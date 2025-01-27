# Start from the default Codespaces "universal" dev container image
FROM mcr.microsoft.com/devcontainers/universal:2

# Switch to root to install software
USER root

# Install Apptainer
RUN sudo add-apt-repository -y ppa:apptainer/ppa && \
    sudo apt update && \
    sudo apt install -y apptainer