# ─────────────────────────────────────────────────────────────
# Stage 1 – build the environment and bake model parameters
# ─────────────────────────────────────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba AS builder

# Pinned commit of BindCraft2 v1.0.2
ARG BINDCRAFT_COMMIT=f5275212e48a379626c72a32b7f13129f3ac1417

# 1. Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      wget \
      build-essential \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2. Clone BindCraft2 repository at pinned commit
RUN git clone https://github.com/PacesaLab/BindCraft2.git /app/bindcraft && \
    cd /app/bindcraft && git checkout ${BINDCRAFT_COMMIT}

# 3. Create Python 3.12 micromamba environment
RUN micromamba create -y -n app -c conda-forge python=3.12 pip

# 4. Install BindCraft2 with CUDA 12 support
RUN micromamba run -n app pip install --no-cache-dir -e '/app/bindcraft[cuda12]'

# 5. Build-time gate to verify imports before baking weights
RUN micromamba run -n app python -c "\
import jax, haiku, optax, biotite, biotraj, matplotlib, numpy, scipy, ml_collections, absl; \
import bindcraft, cuequivariance, cuequivariance_jax; \
print('BUILD GATE PASSED: JAX', jax.__version__, 'BindCraft2 commit', '${BINDCRAFT_COMMIT}')"

# 6. Download and bake AlphaFold 2 parameters and generate WEIGHTS.json manifest
WORKDIR /app
COPY download_weights.py /app/download_weights.py
RUN micromamba run -n app python /app/download_weights.py --base_dir /app && \
    rm -f /app/download_weights.py

# 7. Clean caches and strip static libraries
RUN micromamba clean -a -y && \
    rm -rf /opt/conda/pkgs /root/.cache/pip /tmp/* && \
    find /opt/conda/envs/app -name "*.a" -delete || true


# ─────────────────────────────────────────────────────────────
# Stage 2 – final runtime image
# ─────────────────────────────────────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba

LABEL maintainer="Nicholas Freitas"
LABEL description="BindCraft2: Protein Binder Design Workflow (1.0.2)"
LABEL bindcraft.version="1.0.2"
LABEL bindcraft.commit="f5275212e48a379626c72a32b7f13129f3ac1417"
LABEL bindcraft.python="3.12"
LABEL bindcraft.accelerator="cuda12"
LABEL alphafold.params="alphafold_params_2022-12-06"

# 1. Copy prepared environment, code, and baked model weights
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
COPY --from=builder /app/bindcraft /app/bindcraft
COPY --from=builder /app/models /app/models

# 2. Symlink baked weights inside bindcraft package tree for default paths
RUN rm -rf /app/bindcraft/bindcraft/weights/alphafold && \
    ln -s /app/models/alphafold /app/bindcraft/bindcraft/weights/alphafold

# 3. Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 4. Configure dynamic linker for nvidia wheels
RUN /opt/conda/envs/app/bin/python -c "\
import nvidia, pathlib; \
print('\n'.join(sorted(str(path) for root in nvidia.__path__ for path in pathlib.Path(root).glob('*/lib'))))" \
    > /etc/ld.so.conf.d/bindcraft-cuda.conf && \
    ldconfig

# 5. Auto-activate "app" for every shell and docker CMD
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >  /etc/profile.d/mamba.sh && \
    echo 'micromamba activate app'                      >> /etc/profile.d/mamba.sh

# 6. Set environment variables, weights paths, and cache redirection for Apptainer read-only compatibility
ENV PATH=/opt/conda/envs/app/bin:$PATH
ENV PYTHONPATH=/app/bindcraft
ENV BINDCRAFT_AF2_PARAMS=/app/models/alphafold
ENV BINDCRAFT_WEIGHTS=/app/models
ENV BINDCRAFT_MPNN_WEIGHTS=/app/bindcraft/bindcraft/weights/proteinmpnn/weights_neutral
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/tmp/hf
ENV XDG_CACHE_HOME=/tmp/cache
ENV TRITON_CACHE_DIR=/tmp/triton
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
ENV MPLCONFIGDIR=/tmp/mpl
ENV JAX_COMPILATION_CACHE_DIR=/tmp/bindcraft_xla_cache

# 7. Install entry point CLI shims
RUN printf '#!/bin/bash\nexec /opt/conda/envs/app/bin/python -m bindcraft.cli "$@"\n' \
      > /opt/conda/envs/app/bin/bindcraft && \
    chmod +x /opt/conda/envs/app/bin/bindcraft && \
    cp /opt/conda/envs/app/bin/bindcraft /usr/local/bin/bindcraft

# 8. Copy validation tests
COPY test /app/test

WORKDIR /workspace
ENTRYPOINT ["micromamba", "run", "-n", "app"]
CMD ["bash", "--login", "-i"]
