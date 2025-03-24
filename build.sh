#! /bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <container-name> <container-version>"
    exit 1
fi

CONTAINER_NAME=$1
CONTAINER_VERSION=$2

act -j build-and-push-local --secret-file ../secrets/.secrets \
    --input container-name="$CONTAINER_NAME" \
    --input container-version="$CONTAINER_VERSION" \
    -P ubuntu-latest=-self-hosted
