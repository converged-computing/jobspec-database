#!/bin/bash
set -ex
REMOTE="$1"
DOCKER_IMAGE="$2"

# We assume that the REMOTE does not have ":" expect once
IFS=':' read -ra arr_remote <<< "$REMOTE"
# Get the path
MODEL_PATH=${arr_remote[1]}
THIS_DIR=$(dirname $(realpath "$0"))

# Copy model
scp -r "$REMOTE" "$THIS_DIR"/trained_model
# Fix file paths in moses.ini so they align with Dockerfile
sed -i "s|$MODEL_PATH|/work|g" "$THIS_DIR"/trained_model/moses.ini
docker build -t "$DOCKER_IMAGE" "$THIS_DIR"
# Clean-up on isle four
rm -rf "$THIS_DIR"/trained_model
docker push "$DOCKER_IMAGE"