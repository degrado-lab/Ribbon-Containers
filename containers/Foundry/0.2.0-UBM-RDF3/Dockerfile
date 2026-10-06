# ─────────────────────────────────────────────────────────────
# Ribbon Foundry Container - Version 0.2.0-UBM-RDF3
# Base: official Rosetta Commons Foundry 0.2.0 with baked weights
#       plus Anthropic Uplifting Biomolecular Modeling (UBM)
#       speed optimizations for RFdiffusion3 (RFD3)
# ─────────────────────────────────────────────────────────────
FROM nicholasfreitas/foundry:docker-0.2.0

LABEL maintainer="Nicholas Freitas"
LABEL description="Foundry 0.2.0 with Anthropic UBM acceleration for RFdiffusion3 (fused SwiGLU, CUDA graphs, Triton gather attention, torch.compile)"

# 1. Copy pre-built wheels, UBM rfdiffusion3 framework, and patches
COPY packages /app/packages
COPY ubm/rfdiffusion3 /app/rfdiffusion3
COPY patches /app/patches

# 2. Install UBM opt_core and rfdiffusion3_opt into Foundry venv, apply patches
RUN /app/foundry/.venv/bin/python -m ensurepip --default-pip && \
    /app/foundry/.venv/bin/pip install --no-deps /app/packages/opt_core-*.whl && \
    /app/foundry/.venv/bin/pip install --no-deps -e /app/rfdiffusion3/opt && \
    /app/foundry/.venv/bin/python /app/patches/patch_engine.py && \
    bash /app/rfdiffusion3/opt/forward/xattempt_addon/install.sh apply --force && \
    chmod -R a+rX /app && \
    rm -rf /root/.cache

# 3. Environment variables for offline inference and UBM optimization
ENV CC=/usr/bin/gcc \
    FOUNDRY_CHECKPOINT_DIRS=/weights \
    WEIGHTS=/weights \
    MODEL_OPT=/app/rfdiffusion3 \
    RFDIFFUSION3_OPT=fast \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/cache \
    TRITON_CACHE_DIR=/tmp/triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor \
    MPLCONFIGDIR=/tmp/mpl \
    PATH=/app/foundry/.venv/bin:$PATH

WORKDIR /workspace

CMD ["bash"]
