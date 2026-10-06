# ─────────────────────────────────────────────────────────────
# Ribbon Foundry Container - Version 0.2.0
# Base: official Rosetta Commons Foundry 0.2.0 image with weights
# ─────────────────────────────────────────────────────────────
FROM rosettacommons/foundry:0.2.0-weights

LABEL maintainer="Nicholas Freitas"
LABEL description="Foundry 0.2.0 (RFD3, RF3, ProteinMPNN/LigandMPNN/SolubleMPNN) with baked model weights and Triton JIT support"

# 1. Install build tools required for Triton/cuequivariance runtime compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Install SolubleMPNN checkpoint so the full MPNN suite is available offline
RUN /app/foundry/.venv/bin/foundry install solublempnn --checkpoint-dir /weights && \
    chmod a+r /weights/solublempnn_v_48_020.pt && \
    mkdir -p /workspace && chmod 777 /workspace

# 3. Set offline and cache environment variables for container execution
ENV CC=/usr/bin/gcc \
    FOUNDRY_CHECKPOINT_DIRS=/weights \
    WEIGHTS=/weights \
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
