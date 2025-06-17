# ───────────────────────────────
# Stage 1 – build the environment
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba AS builder

# 1. Install build dependencies
# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Clone the RaptorX-Single repository
WORKDIR /app
RUN git clone https://github.com/AndersJing/RaptorX-Single.git

# 3 Download model weights:
WORKDIR /app/RaptorX-Single
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

# 1.  Optional: bring in an environment spec
#     (If you don't have environment.yml, skip COPY and use RUN micromamba install …)
#COPY environment.yml /tmp/environment.yml

# 2.  Create env "app" and strip caches
# Create env “app”, then add pip packages inside it
RUN micromamba create -y -n app -c conda-forge -f environment.yml
#RUN micromamba run -n app pip install --no-cache-dir -r ./requirements.txt
    # clean Conda & pip caches to keep the layer small
RUN micromamba clean -a -y
RUN rm -rf /opt/conda/pkgs /root/.cache/pip

# 3.  Optional: shave a few more MB by deleting static libs
RUN find /opt/conda/envs/app -name "*.a" -delete || true


# ───────────────────────────────
# Stage 2 – final runtime image
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba

# 4.  Copy the ready-made env (≈ size of packages only)
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
# Copy the RaptorX-Single repository and model parameters
COPY --from=builder /app/RaptorX-Single /app/RaptorX-Single

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
#CMD ["bash", "--login", "-i"] # This is in the base image.