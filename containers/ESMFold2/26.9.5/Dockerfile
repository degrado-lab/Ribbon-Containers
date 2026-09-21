# ───────────────────────────────
# Stage 1 – build the environment
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba AS builder

# Both halves of ESMFold2 live in Biohub forks with no PyPI release, so they are
# pinned by commit. Bump these (and the container version) together.
ARG ESM_COMMIT=f652b471d29da828b31e9b7a9cf7d0a7803240f5
ARG TRANSFORMERS_COMMIT=3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf

# 1. Build dependencies (git for the pinned forks, gcc for the Triton stubs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git \
      build-essential \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2.  Create env "app" and strip caches
#     python 3.12, NOT the template's 3.11: the esm fork declares
#     Requires-Python >=3.12,<3.13 and will not resolve on 3.11.
RUN micromamba create -y -n app -c conda-forge python=3.12 pip

#     torch first, from the cu126 index, so the forks resolve against it
#     instead of pulling a default-index build.
RUN micromamba run -n app pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu126 \
      torch==2.7.1
#     esm declares its transformers dependency as a direct URL on @main, so pip
#     rejects a second direct URL for the same package and the resolve fails
#     ("Cannot install esm==3.3.0 and transformers 4.57.6"). Install esm first,
#     letting it pull transformers@main plus every other dependency, then
#     overwrite transformers with the pinned commit. --no-deps stops pip from
#     re-resolving (and from pulling a non-cu126 torch) on the second call.
#     The pin is not cosmetic: main has already moved past the commit our
#     validated runs used, so a naive install gets a different transformers.
RUN micromamba run -n app pip install --no-cache-dir \
      "git+https://github.com/Biohub/esm.git@${ESM_COMMIT}"
RUN micromamba run -n app pip install --no-cache-dir --no-deps --force-reinstall \
      "git+https://github.com/Biohub/transformers.git@${TRANSFORMERS_COMMIT}"
RUN micromamba run -n app pip install --no-cache-dir \
      biotite==1.7.1 \
      rdkit==2026.3.6 \
      huggingface_hub \
      safetensors
    # clean Conda & pip caches to keep the layer small
RUN micromamba clean -a -y
RUN rm -rf /opt/conda/pkgs /root/.cache/pip

# 3.  Optional: shave a few more MB by deleting static libs
RUN find /opt/conda/envs/app -name "*.a" -delete || true

#     Fail the build here rather than at fold time if the resolver quietly
#     swapped torch for a CPU or non-cu126 wheel.
#     NB: this gate cannot import esm. `import esm` initialises Triton's driver,
#     which raises "0 active drivers" without a GPU, and docker build has no
#     GPU. So verify torch, the two commit pins and the module paths from
#     metadata and disk only; the import and a real fold are checked in the
#     GPU validation run (see README / container_validation.json).
ARG PY=/opt/conda/envs/app/lib/python3.12/site-packages
RUN micromamba run -n app python -c "\
import glob, json, torch; \
assert torch.__version__ == '2.7.1+cu126', torch.__version__; \
sha = lambda p: json.load(open(glob.glob('${PY}/' + p + '-*.dist-info/direct_url.json')[0]))['vcs_info']['commit_id']; \
assert sha('transformers') == '${TRANSFORMERS_COMMIT}', ('transformers', sha('transformers')); \
assert sha('esm') == '${ESM_COMMIT}', ('esm', sha('esm')); \
assert glob.glob('${PY}/transformers/models/esmfold2/modeling_esmfold2.py'), 'no transformers esmfold2'; \
assert glob.glob('${PY}/esm/models/esmfold2/__init__.py'), 'no esm esmfold2'; \
print('OK', torch.__version__, sha('transformers')[:8], sha('esm')[:8])"

# 4. Create a workspace, and store the weights.
#    Downloaded fp32 -> cast to bf16 -> originals deleted, all inside one RUN so
#    the 26 GB of fp32 safetensors never lands in a persisted layer. bf16 is what
#    load_esmc() casts to at load time anyway (precision="bf16" default), so this
#    is bit-identical at inference and halves the image.
WORKDIR /app
ENV HF_HOME=/tmp/hf
COPY download_weights.py /app/download_weights.py
RUN micromamba run -n app python /app/download_weights.py --out /app/models && \
    rm -rf /tmp/hf /root/.cache/huggingface

# ───────────────────────────────
# Stage 2 – final runtime image
# ───────────────────────────────
FROM nicholasfreitas/cuda_micromamba_base:cuda12.2.0-base-ubuntu22.04-micromamba

LABEL maintainer="Nicholas Freitas"
LABEL description="Biohub ESMFold2 + ESMFold2-Fast, all weights baked in (stored bf16)"
LABEL esmfold2.esm_commit="f652b471d29da828b31e9b7a9cf7d0a7803240f5"
LABEL esmfold2.transformers_commit="3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf"
LABEL esmfold2.weights_dtype="bfloat16"
LABEL esmfold2.weights_note="all three repos are published fp32 and were cast to bf16 at build time"

# 4.  Copy the ready-made env (≈ size of packages only)
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app
# Copy the model weights
COPY --from=builder /app/models /app/models

# 5.  Add any Debian runtime deps your software needs
#     (keep --no-install-recommends to stay lean)
#     ESMFold2's fused backend is Triton, which shells out to a C compiler when
#     it JIT-builds its launcher stubs on the first fold.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 6.  Auto-activate "app" for every shell *and* every docker run CMD
SHELL ["/bin/bash", "-c"]
RUN echo 'eval "$(micromamba shell hook --shell bash)"'  > /etc/profile.d/mamba.sh && \
    echo 'micromamba activate app'                     >> /etc/profile.d/mamba.sh

# 7.  Make sure non-interactive commands also see the env's executables first
ENV PATH=/opt/conda/envs/app/bin:$PATH

# 8.  Weights are local and complete: never let a code path reach the Hub.
ENV ESMFOLD2_MODELS=/app/models
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

#     ESMFold2InputBuilder() calls load_ccd(), which ignores ESMFOLD2_MODELS and
#     hf_hub_download's biohub/ESMFold2 ccd.pkl into the HF cache unless this var
#     (read at import time in esm/models/esmfold2/conformers.py) names a file.
#     Without it the container folds only where a populated HF cache happens to
#     be mounted -- which is exactly the non-stand-alone failure it hides.
#     Note the spelling: ESMCFOLD, not ESMFOLD2.
ENV ESMCFOLD_CCD_PATH=/app/models/ESMFold2/ccd.pkl

# 9.  Every cache Torch/Triton/HF might write goes to /tmp. Apptainer mounts the
#     image read-only, so a default cache under $HOME or in the image is the
#     failure mode that stopped weights being baked into the ESM3 container.
ENV HF_HOME=/tmp/hf
ENV XDG_CACHE_HOME=/tmp/cache
ENV TRITON_CACHE_DIR=/tmp/triton
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
ENV MPLCONFIGDIR=/tmp/mpl
ENV TOKENIZERS_PARALLELISM=false

# 10. The fold helper and its patches, plus example inputs.
COPY esmfold2_patches.py /app/esmfold2_patches.py
COPY esmfold2_fold.py /app/esmfold2_fold.py
COPY test /app/test
ENV PYTHONPATH=/app

#     Put the helper on PATH so `run ... esmfold2-fold --input ...` works
#     without anyone remembering where in the image it lives.
RUN printf '#!/bin/bash\nexec /opt/conda/envs/app/bin/python /app/esmfold2_fold.py "$@"\n' \
      > /opt/conda/envs/app/bin/esmfold2-fold && \
    chmod +x /opt/conda/envs/app/bin/esmfold2-fold

WORKDIR /workspace
