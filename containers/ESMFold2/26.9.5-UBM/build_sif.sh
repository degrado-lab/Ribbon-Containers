#!/bin/bash
set -e
cd /home/freitas/workspace/Ribbon-Containers/containers/ESMFold2/26.9.5-UBM
export APPTAINER_TMPDIR=/scratch/freitas/.apptainer_tmp
export APPTAINER_CACHEDIR=/scratch/freitas/.apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
sed 's|^Bootstrap: docker$|Bootstrap: docker-daemon|' definition.def > definition.local.def
echo "Starting apptainer build at $(date)"
apptainer build --force --mksquashfs-args '-mem 512M -processors 2' esmfold2_26.9.5-UBM.sif definition.local.def
rm -f definition.local.def
echo "Completed apptainer build at $(date)"
