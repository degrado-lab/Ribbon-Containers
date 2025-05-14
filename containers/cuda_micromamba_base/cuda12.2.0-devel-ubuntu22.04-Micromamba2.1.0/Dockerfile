###############################################################################
# CUDA 12.2 (run‑time)  +  Micromamba in /usr/bin  +  Python 3.11
###############################################################################
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 1 ─────── Core utilities (no docs, no recommends → smaller image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl bzip2 git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2 ─────── Fetch the single‑file Micromamba build and drop in /usr/bin
#           The tarball layout is `bin/micromamba`, so:
#             • -O      → stream file to stdout
#             • > /usr/bin/micromamba  → write it directly
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvjO bin/micromamba > /usr/bin/micromamba && \
    chmod 755 /usr/bin/micromamba

# 3 ─────── (optional) pick a root prefix the binary can write to.
#           Here we keep everything under /usr/micromamba (could be /root/micromamba).
ENV MAMBA_ROOT_PREFIX=/usr/micromamba

# 4 ─────── Install Python 3.11 into the *base* env and clean caches
RUN micromamba install -y python=3.11 && \
    micromamba clean -afy

CMD ["/bin/bash"]
