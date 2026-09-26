# ───────────────────────────────
# Stage 1 – build the environment
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba AS builder

# Pinned commit of LDDM upstream repository
ARG LDDM_COMMIT=f254fb4f8525b3803e79eb95e9f1a163fe8b2459

# 1. Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      wget \
      curl \
      build-essential \
      ca-certificates \
      libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Create Python 3.11 environment
#    pyproject.toml specifies requires-python = "==3.11.*"
RUN micromamba create -y -n app -c conda-forge python=3.11 pip

# 3. Install PyTorch with CUDA 12.4 runtime wheels
RUN micromamba run -n app pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0+cu124

# 4. Install torch-scatter from PyG index
RUN micromamba run -n app pip install --no-cache-dir \
      --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html \
      torch-scatter==2.1.2+pt26cu124

# 5. Install all pinned core and scientific dependencies
RUN micromamba run -n app pip install --no-cache-dir \
      lightning==2.6.5 \
      torch-geometric==2.7.0 \
      rdkit==2023.9.6 \
      biopython==1.83 \
      prody==2.6.1 \
      prolif==2.0.3 \
      posebusters==0.3.1 \
      useful-rdkit-utils==0.65 \
      fcd==1.2.2 \
      biotite==1.6.0 \
      numpy==1.26.4 \
      scipy==1.13.0 \
      pandas==2.2.3 \
      pot==0.9.4 \
      einops==0.8.2 \
      tqdm==4.67.3 \
      setuptools==80.10.2 \
      pyyaml==6.0.3 \
      scikit-learn==1.8.0 \
      seaborn==0.13.2 \
      matplotlib==3.8.4

# 6. Build Reduce v4.15 from source
RUN wget https://github.com/rlabduke/reduce/archive/refs/tags/v4.15.tar.gz && \
    tar -xf v4.15.tar.gz && \
    cd reduce-4.15/ && \
    make && \
    mkdir -p /opt/conda/envs/app/bin && \
    cp reduce_src/reduce /opt/conda/envs/app/bin/ && \
    chmod +x /opt/conda/envs/app/bin/reduce && \
    cd .. && rm -rf reduce-4.15 v4.15.tar.gz

# 7. Download Gnina v1.1 binary
RUN wget https://github.com/gnina/gnina/releases/download/v1.1/gnina \
    -O /opt/conda/envs/app/bin/gnina && \
    chmod +x /opt/conda/envs/app/bin/gnina

# 8. Clone LDDM repo and install package in editable/module mode
RUN git clone https://github.com/LPDI-EPFL/lddm.git /app/lddm && \
    cd /app/lddm && git checkout ${LDDM_COMMIT} && \
    micromamba run -n app pip install --no-cache-dir --no-deps -e /app/lddm

# 9. Build-time gate to verify imports and version pins before baking weights
RUN micromamba run -n app python -c "\
import torch, torch_geometric, torch_scatter, lightning, rdkit, lddm, shutil; \
assert torch.__version__.startswith('2.6.0'), ('torch', torch.__version__); \
assert torch_scatter.__version__.startswith('2.1.2'), ('torch_scatter', torch_scatter.__version__); \
assert shutil.which('/opt/conda/envs/app/bin/reduce'), 'reduce missing'; \
assert shutil.which('/opt/conda/envs/app/bin/gnina'), 'gnina missing'; \
print('BUILD GATE PASSED: PyTorch', torch.__version__, 'PyG', torch_geometric.__version__, 'RDKit', rdkit.__version__)"

# 10. Download and bake model checkpoints, 3D validity geometry, and SynSpace data
WORKDIR /app
COPY download_weights.py /app/download_weights.py
RUN micromamba run -n app python /app/download_weights.py --base_dir /app && \
    rm -f /app/download_weights.py

# Clean micromamba & pip caches and strip static archives
RUN micromamba clean -a -y && \
    rm -rf /opt/conda/pkgs /root/.cache/pip /tmp/* && \
    find /opt/conda/envs/app -name "*.a" -delete || true


# ───────────────────────────────
# Stage 2 – final runtime image
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba

LABEL maintainer="Nicholas Freitas"
LABEL description="LDDM: A Unified 3D Generative Model for Synthesizable Structure-Based Drug Design (0.1.0)"
LABEL lddm.commit="f254fb4f8525b3803e79eb95e9f1a163fe8b2459"
LABEL lddm.torch="2.6.0+cu124"
LABEL lddm.python="3.11"
LABEL lddm.zenodo="22754501"

# 1. Copy the prepared environment and baked application files
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
COPY --from=builder /app/lddm /app/lddm
COPY --from=builder /app/models /app/models
COPY --from=builder /app/data /app/data

# 2. Install runtime debian dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libxrender1 \
      libxext6 \
      libsm6 \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Also ensure reduce and gnina are symlinked/available on standard system PATH
RUN cp /opt/conda/envs/app/bin/reduce /usr/local/bin/reduce && \
    cp /opt/conda/envs/app/bin/gnina /usr/local/bin/gnina

# 3. Auto-activate "app" for every shell and docker CMD
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >  /etc/profile.d/mamba.sh && \
    echo 'micromamba activate app'                      >> /etc/profile.d/mamba.sh

# 4. PATH and offline flags
ENV PATH=/opt/conda/envs/app/bin:$PATH
ENV PYTHONPATH=/app/lddm
ENV LDDM_MODELS=/app/models
ENV LDDM_DATA=/app/data
ENV LDDM_CHECKPOINT=/app/models/checkpoints/lddm.ckpt
ENV LDDM_CHECKPOINT_MIT=/app/models/checkpoints/lddm_CDBB.ckpt
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYSTOW_HOME=/tmp/pystow
ENV HF_HOME=/tmp/hf
ENV MPLCONFIGDIR=/tmp/mpl

# 5. Cache redirection for Apptainer read-only root compatibility
ENV HF_HOME=/tmp/hf
ENV XDG_CACHE_HOME=/tmp/cache
ENV TRITON_CACHE_DIR=/tmp/triton
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
ENV MPLCONFIGDIR=/tmp/mpl

# 6. Copy pre-configured templates and reference tests
COPY configs /app/configs
COPY test /app/test

# 7. Install CLI entry point shims
RUN printf '#!/bin/bash\nexec /opt/conda/envs/app/bin/python /app/lddm/scripts/sample.py "$@"\n' \
      > /opt/conda/envs/app/bin/lddm-sample && \
    chmod +x /opt/conda/envs/app/bin/lddm-sample && \
    printf '#!/bin/bash\nexec /opt/conda/envs/app/bin/python /app/lddm/scripts/generate_programmable_design.py "$@"\n' \
      > /opt/conda/envs/app/bin/lddm-programmable && \
    chmod +x /opt/conda/envs/app/bin/lddm-programmable && \
    printf '#!/bin/bash\nexec /opt/conda/envs/app/bin/python /app/lddm/scripts/prepare_chemical_space.py "$@"\n' \
      > /opt/conda/envs/app/bin/lddm-prep-chemspace && \
    chmod +x /opt/conda/envs/app/bin/lddm-prep-chemspace && \
    printf '#!/bin/bash\nexec /opt/conda/envs/app/bin/python /app/lddm/scripts/prepare_synspace.py "$@"\n' \
      > /opt/conda/envs/app/bin/lddm-prep-synspace && \
    chmod +x /opt/conda/envs/app/bin/lddm-prep-synspace

WORKDIR /workspace
ENTRYPOINT ["micromamba", "run", "-n", "app"]
CMD ["bash", "--login", "-i"]
