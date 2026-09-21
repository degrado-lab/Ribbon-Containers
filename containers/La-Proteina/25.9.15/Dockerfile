# ───────────────────────────────
# Stage 0 – borrow CUDA headers/libs
# ───────────────────────────────
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS cuda_devel

# ───────────────────────────────
# Stage 1 – build the environment
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba AS builder

# 1. Install build dependencies
# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Clone the LaProteina repository
WORKDIR /app
RUN git clone https://github.com/NVIDIA-Digital-Bio/la-proteina.git
    
# 1.  Optional: bring in an environment spec
#     (If you don't have environment.yml, skip COPY and use RUN micromamba install …)
#COPY environment.yml /tmp/environment.yml

# 2.  Create env "app" and strip caches
# Create env “app”, then add pip packages inside it
RUN micromamba env create -y -n app -c conda-forge -f ./la-proteina/environment.yaml
#RUN micromamba run -n app pip install --no-cache-dir -r ./requirements.txt
    # clean Conda & pip caches to keep the layer small
RUN micromamba run -n app pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118 &&
    micromamba run -n app pip install graphein==1.7.7 --no-deps && \
    micromamba run -n app pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
RUN micromamba clean -a -y
RUN rm -rf /opt/conda/pkgs /root/.cache/pip

# 3.  Optional: shave a few more MB by deleting static libs
RUN find /opt/conda/envs/app -name "*.a" -delete || true

# 4. Download model weights
COPY checkpoints/ /app/la_proteina/checkpoints_laproteina



# ───────────────────────────────
# Stage 2 – final runtime image
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba

# 4.  Copy the ready-made env (≈ size of packages only)
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
# Copy the LigandMPNN repository and model parameters
COPY --from=builder /app/la_proteina /app/la_proteina

# 5.  Add any Debian runtime deps your software needs
#     (keep --no-install-recommends to stay lean)
# Debian/Ubuntu base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

# 6) Copy only what KeOps needs from CUDA devel:
#    - headers: cuda.h, nvrtc.h (and friends)
#    - libraries: libnvrtc + builtins
# NOTE: copy from targets path to avoid symlink issues.
RUN mkdir -p /usr/local/cuda/include /usr/local/cuda/lib64
COPY --from=cuda_devel /usr/local/cuda/targets/x86_64-linux/include/ /usr/local/cuda/include/
COPY --from=cuda_devel /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so*           /usr/local/cuda/lib64/
COPY --from=cuda_devel /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc-builtins.so*  /usr/local/cuda/lib64/
# (Optional, sometimes useful)
COPY --from=cuda_devel /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so*          /usr/local/cuda/lib64/

# 4) CUDA env so KeOps can find them
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CPATH=/usr/local/cuda/include:$CPATH \
    LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PATH=/opt/conda/envs/app/bin:$PATH

# 6.  Auto-activate "app" for every shell *and* every docker run CMD
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"'  > /etc/profile.d/mamba.sh && \
    echo 'micromamba activate app'                     >> /etc/profile.d/mamba.sh

# 7.  Make sure non-interactive commands also see the env’s executables first
ENV PATH=/opt/conda/envs/app/bin:$PATH

# 8. Set up what happens when the container starts
WORKDIR /app
# Whatever command we pass in, run inside the "app" env
ENTRYPOINT ["micromamba", "run", "-n", "app"]
# If not command is given, start a shell
CMD ["bash", "--login", "-i"]