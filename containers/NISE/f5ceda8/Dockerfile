# ─────────────────────────  Stage 0 – BUILD  ─────────────────────────
# Needs the *‑devel* image + compilers, headers, git, … but this stage
# will NOT be shipped to your users.
FROM rckormos/cuda_micromamba_base:cuda12.2.0-devel-ubuntu22.04-Micromamba2.1.0 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/usr/micromamba

# ── 1.  OS deps only for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git wget curl tar && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ── 2.  boltz  ──────────────────────────────────────────────────────
RUN git clone https://github.com/jwohlwend/boltz.git
RUN micromamba create -y -n env_boltz python=3.10 && \
    micromamba run -n env_boltz pip install -e ./boltz && \
    micromamba clean -afy

# ── 3.  NISE + LASErMPNN env  ───────────────────────────────────────
RUN git clone --recurse-submodules https://github.com/degrado-lab/NISE.git

# create the env but keep pkg cache out of the final image
RUN micromamba env create -f /opt/NISE/LASErMPNN/conda_env.yml -y && \
    micromamba clean -afy

# build ProDy inside the lasermpnn env
RUN git clone https://github.com/prody/ProDy.git
RUN micromamba run -n lasermpnn bash -c \
    "cd ProDy && python setup.py build_ext --inplace --force && pip install -Ue . && micromamba clean -afy"

# expand het dict
RUN tar -xf /opt/NISE/hetdict.tar.gz -C /opt/NISE

# ───────────────────────  Stage 1 – RUNTIME  ────────────────────────
# CUDA **runtime** image (~4 GB) + a single static micromamba binary.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

# Pull in the CUDA apt keyring & repository  
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg2 curl \
    && curl -fsSL \
       https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
       -o cuda-keyring.deb \
    && dpkg -i cuda-keyring.deb \
    && rm cuda-keyring.deb \
    && rm -rf /var/lib/apt/lists/*

# install curl to download micromamba + NVRTC dev bits so KeOps can JIT compile
# install curl + NVRTC dev bits so KeOps can JIT compile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-minimal-build-12-2 \
        cuda-nvrtc-12-2 \
        cuda-nvrtc-dev-12-2 \
        cuda-cudart-dev-12-2 \
    && rm -rf /var/lib/apt/lists/*

# set the CUDA_HOME and CUDA_PATH environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda

# tiny static micromamba (≈ 6 MB)
ENV MAMBA_ROOT_PREFIX=/usr/micromamba
RUN mkdir -p /usr/bin && \
    curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvjO bin/micromamba > /usr/bin/micromamba && \
    chmod 755 /usr/bin/micromamba

# ── copy only what’s needed from the build stage  ────────────────────
#   • the two environments under $MAMBA_ROOT_PREFIX
#   • the application code under /opt
COPY --from=build /usr/micromamba /usr/micromamba
COPY --from=build /opt /opt

# download Boltz cache
RUN mkdir -p /boltz_cache
RUN curl -fSL https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt \
      -o /boltz_cache/ccd.pkl \
    && curl -fSL https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt \
      -o /boltz_cache/boltz1_conf.ckpt \
    && curl -fSL https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar \
      -o /boltz_cache/mols.tar \
    && tar -xvf /boltz_cache/mols.tar \
    && rm /boltz_cache/mols.tar
ENV BOLTZ_CACHE=/boltz_cache

# strip pip/conda caches that snuck in
RUN find /usr/micromamba -name '*.pyc' -delete

# executables from envs on PATH
ENV PATH=/usr/micromamba/envs/lasermpnn/bin:/usr/micromamba/envs/env_boltz/bin:$PATH

WORKDIR /opt/NISE
ENTRYPOINT ["micromamba", "run", "-n", "lasermpnn", "--"]
CMD ["python", "run_nise_boltz.py"]
