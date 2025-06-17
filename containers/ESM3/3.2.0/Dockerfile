# ───────────────────────────────
# Stage 1 – build the environment
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba AS builder

# 1. Install build dependencies
# Install dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git curl wget build-essential python3-dev \
#     && rm -rf /var/lib/apt/lists/*
    
# 2. Clone the LigandMPNN repository
# WORKDIR /workspace
# RUN git clone https://github.com/dauparas/LigandMPNN.git

# 3. Download model parameters
# WORKDIR /workspace/LigandMPNN
# RUN bash get_model_params.sh "./model_params"
    
# 1.  Optional: bring in an environment spec
#     (If you don't have environment.yml, skip COPY and use RUN micromamba install …)
#COPY environment.yml /tmp/environment.yml

# 2.  Create env "app" and strip caches
# Create env “app”, then add pip packages inside it
RUN micromamba create -y -n app -c conda-forge python=3.11 pip
RUN micromamba run -n app pip install --no-cache-dir esm==3.2.0
    # clean Conda & pip caches to keep the layer small
RUN micromamba clean -a -y
RUN rm -rf /opt/conda/pkgs /root/.cache/pip

# 3.  Optional: shave a few more MB by deleting static libs
RUN find /opt/conda/envs/app -name "*.a" -delete || true

### We are NOT downloading model weights for ESM3. The Huggingface token system makes this intractable, 
# and it's quite easy for the user to downlaod the weights by running an example script.

# 4. Probably need to download weights here.
# COPY models--EvolutionaryScale--esm3-sm-open-v1.tar.gz /tmp/models--EvolutionaryScale--esm3-sm-open-v1.tar.gz
# RUN mkdir -p /app/huggingface/hub && \
#     tar -xzf /tmp/models--EvolutionaryScale--esm3-sm-open-v1.tar.gz -C /app/huggingface/hub && \
#     rm /tmp/models--EvolutionaryScale--esm3-sm-open-v1.tar.gz

# ───────────────────────────────
# Stage 2 – final runtime image
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba

# 4.  Copy the ready-made env (≈ size of packages only)
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
# Copy weights:
#COPY --from=builder /app/huggingface/hub /app/huggingface/hub

# 5.  Add any Debian runtime deps your software needs
#     (keep --no-install-recommends to stay lean)
# RUN apt-get update && \
#     apt-get install --no-install-recommends -y git && \
#     rm -rf /var/lib/apt/lists/*

# 6.  Auto-activate "app" for every shell *and* every docker run CMD
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"'  > /etc/profile.d/mamba.sh && \
    echo 'micromamba activate app'                     >> /etc/profile.d/mamba.sh

# 7.  Make sure non-interactive commands also see the env’s executables first
ENV PATH=/opt/conda/envs/app/bin:$PATH

WORKDIR /workspace

# Nope, we'll use the user-default huggingface cache.
# 8. Set the Huggingface cache directories:
#ENV HF_HOME=/app/huggingface/

#CMD ["bash", "--login", "-i"] # This is in the base image.