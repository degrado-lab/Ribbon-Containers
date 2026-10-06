# ───────────────────────────────
# Stage 1 – build environment & apply Anthropic UBM patches
# ───────────────────────────────
# We use nicholasfreitas/esmfold2:docker-26.9.5-fa2.8.3 as base builder:
# it carries the pre-cast bf16 baked weights (/app/models, 13.97 GB)
# and pre-built flash-attn 2.8.3.post1 wheel on python 3.12 / cu126.
FROM nicholasfreitas/esmfold2:docker-26.9.5-fa2.8.3 AS builder

# 1. Install build tools for patching and wheel compiling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      patch && \
    rm -rf /var/lib/apt/lists/*

# 2. Stage packages and apply Anthropic UBM patches U1-U6
WORKDIR /tmp/build_pkg
COPY packages/transformers-ef32577f.tar.gz /tmp/build_pkg/
COPY packages/esm-26b0bc2b.tar.gz /tmp/build_pkg/
COPY patches /tmp/build_pkg/patches

RUN tar -xzf transformers-ef32577f.tar.gz && \
    cd transformers-ef32577f && \
    patch -p1 -i /tmp/build_pkg/patches/U1_swa_sdpa_mask_memoize.diff && \
    patch -p1 -i /tmp/build_pkg/patches/U2_sampler_schedule_host_syncs.diff && \
    patch -p1 -i /tmp/build_pkg/patches/U3_confidence_chain_pair_host_syncs.diff && \
    patch -p1 -i /tmp/build_pkg/patches/U4_trimul_stage3_row_pitch_alignment.diff && \
    patch -p1 -i /tmp/build_pkg/patches/U5_pair_bias_int64_offsets.diff && \
    cd /tmp/build_pkg && \
    tar -xzf esm-26b0bc2b.tar.gz && \
    cd esm-26b0bc2b && \
    patch -p1 -i /tmp/build_pkg/patches/U6_msa_features_vectorised.diff

# 3. Install patched transformers and esm into the app environment
RUN /opt/conda/envs/app/bin/pip install --no-cache-dir --no-deps --force-reinstall \
      /tmp/build_pkg/transformers-ef32577f \
      /tmp/build_pkg/esm-26b0bc2b && \
    rm -rf /tmp/build_pkg

# 4. Clean Conda and pip caches
RUN micromamba clean -a -y && \
    rm -rf /opt/conda/pkgs /root/.cache/pip

# 5. Build-time gate asserting the UBM patches and packages
RUN /opt/conda/envs/app/bin/python -c "\
import glob, json, os; \
sp = '/opt/conda/envs/app/lib/python3.12/site-packages'; \
common = open(f'{sp}/transformers/models/esmfold2/modeling_esmfold2_common.py').read(); \
assert '_swa_sdpa_mask' in common, 'U1 patch missing'; \
assert '_sched = [float(v) for v in schedule.tolist()]' in common, 'U2 patch missing'; \
model = open(f'{sp}/transformers/models/esmfold2/modeling_esmfold2.py').read(); \
assert 'forced one device->host sync per chain id' in model, 'U3 patch missing'; \
trimul = open(f'{sp}/transformers/models/esmfold2/kernels/trimul_with_residual.py').read(); \
assert '_einsum_row_pitch_aligned' in trimul, 'U4 patch missing'; \
bias = open(f'{sp}/transformers/models/esmfold2/kernels/fused_attention_pair_bias.py').read(); \
assert 'pid_b.to(tl.int64)' in bias, 'U5 patch missing'; \
msa = open(f'{sp}/esm/models/esmfold2/paired_msa.py').read(); \
assert '_INSERTION_BYTES' in msa, 'U6 patch missing'; \
print('OK: All UBM patches U1-U6 verified in installed packages')"

# ───────────────────────────────
# Stage 2 – final runtime image
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba

LABEL maintainer="Nicholas Freitas"
LABEL description="Biohub ESMFold2 + ESMFold2-Fast with Anthropic UBM patches U1-U6, weights baked in (stored bf16)"
LABEL esmfold2.version="26.9.5-UBM"
LABEL esmfold2.ubm_patches="U1,U2,U3,U4,U5,U6"
LABEL esmfold2.esm_commit="26b0bc2b771e3e419ea74f445a5f35cc094a1509"
LABEL esmfold2.transformers_commit="ef32577f55da19a4989cd7b22e004dc43a4998cb"
LABEL esmfold2.flash_attn="2.8.3.post1"
LABEL esmfold2.weights_dtype="bfloat16"

# Copy the ready-made env (with patched packages & flash-attn)
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
# Copy the model weights
COPY --from=builder /app/models /app/models

# Runtime dependencies (gcc for Triton JIT)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Auto-activate "app"
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"'  > /etc/profile.d/mamba.sh && \
    echo 'micromamba activate app'                     >> /etc/profile.d/mamba.sh

ENV PATH=/opt/conda/envs/app/bin:$PATH

ENV ESMFOLD2_MODELS=/app/models
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV ESMCFOLD_CCD_PATH=/app/models/ESMFold2/ccd.pkl

# Cache redirection for read-only Apptainer containers
ENV HF_HOME=/tmp/hf
ENV XDG_CACHE_HOME=/tmp/cache
ENV TRITON_CACHE_DIR=/tmp/triton
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
ENV MPLCONFIGDIR=/tmp/mpl
ENV TOKENIZERS_PARALLELISM=false

# Scripts and tests
COPY esmfold2_patches.py /app/esmfold2_patches.py
COPY esmfold2_fold.py /app/esmfold2_fold.py
COPY download_weights.py /app/download_weights.py
COPY test /app/test
ENV PYTHONPATH=/app

# CLI shim
RUN printf '#!/bin/bash\nexec /opt/conda/envs/app/bin/python /app/esmfold2_fold.py "$@"\n' \
      > /opt/conda/envs/app/bin/esmfold2-fold && \
    chmod +x /opt/conda/envs/app/bin/esmfold2-fold

WORKDIR /workspace
