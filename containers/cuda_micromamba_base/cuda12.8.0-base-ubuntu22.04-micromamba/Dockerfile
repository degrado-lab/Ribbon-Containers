FROM nvidia/cuda:12.8.0-base-ubuntu22.04

# Minimal helpers: ca-certs, curl, bzip2
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        ca-certificates curl bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Fetch micromamba (.tar.bz2) and drop the binary in /usr/bin
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/bin --strip-components=1 bin/micromamba

# Build the lean base env
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:/usr/bin:$PATH
RUN micromamba install -y -n base -c conda-forge python=3.11 && \
    micromamba clean -a -y && \
    rm -rf $MAMBA_ROOT_PREFIX/pkgs

# Load the shell hooks:
RUN echo 'eval "$(micromamba shell hook --shell bash)"' \
        > /etc/profile.d/mamba.sh && \
    echo 'micromamba activate base' >> /etc/profile.d/mamba.sh

# Make sure interactive containers start a login shell
ENTRYPOINT ["/bin/bash", "--login", "-i"]
WORKDIR /workspace
