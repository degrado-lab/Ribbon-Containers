#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export APPTAINER_TMPDIR=/scratch/freitas/.apptainer_tmp
export APPTAINER_CACHEDIR=/scratch/freitas/.apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

sed 's|^Bootstrap: docker$|Bootstrap: docker-daemon|' definition.def > definition.local.def

apptainer build --force --mksquashfs-args '-mem 512M -processors 2' \
    foundry_0.2.0-UBM-RDF3.sif definition.local.def

rm -f definition.local.def
