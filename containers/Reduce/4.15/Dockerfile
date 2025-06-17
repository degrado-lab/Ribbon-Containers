# ───────────────────────────────
# Stage 1 – build the environment
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba AS builder

# 1. Install build dependencies
# Install dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#       build-essential \
#       ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

# 2.  Create env "app" and strip caches
# Create env “app”, then add pip packages inside it
RUN micromamba create -y -n app -c conda-forge -c bioconda python pip reduce=4.15
#RUN micromamba install -n app -c bioconda reduce=4.15
    # clean Conda & pip caches to keep the layer small
RUN micromamba clean -a -y
RUN rm -rf /opt/conda/pkgs /root/.cache/pip

# 3.  Optional: shave a few more MB by deleting static libs
RUN find /opt/conda/envs/app -name "*.a" -delete || true

# 4. Create a workspace, and store the weights:
WORKDIR /app

# ───────────────────────────────
# Stage 2 – final runtime image
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba

# 4.  Copy the ready-made env (≈ size of packages only)
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
COPY --from=builder /app /app

# 5.  Add any Debian runtime deps your software needs
#     (keep --no-install-recommends to stay lean)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#       build-essential \
#       ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

# 6.  Auto-activate "app" for every shell *and* every docker run CMD
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"'  > /etc/profile.d/mamba.sh && \
    echo 'micromamba activate app'                     >> /etc/profile.d/mamba.sh

# 7.  Make sure non-interactive commands also see the env’s executables first
ENV PATH=/opt/conda/envs/app/bin:$PATH

#CMD ["bash", "--login", "-i"] # This is in the base image.