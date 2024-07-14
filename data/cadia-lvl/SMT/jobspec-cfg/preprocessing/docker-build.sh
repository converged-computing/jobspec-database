#!/bin/bash
set -euxo
TAG=$1

THIS_DIR=$(dirname $(realpath "$0"))

docker build --no-cache -t haukurp/moses-lvl:$TAG "$THIS_DIR"
docker push haukurp/moses-lvl:$TAG