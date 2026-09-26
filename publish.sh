#! /bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <container-name> <container-version>"
    exit 1
fi

CONTAINER_NAME=$1
CONTAINER_VERSION=$2

act -j publish-local -W .github/workflows/publish-container-local.yml --secret-file ../secrets/.secrets \
    --input container-name="$CONTAINER_NAME" \
    --input container-version="$CONTAINER_VERSION" \
    --input host-workspace="$PWD" \
    --env HOST_WORKSPACE="$PWD" \
    -P ubuntu-latest=-self-hosted
