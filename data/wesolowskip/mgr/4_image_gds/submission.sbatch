#!/bin/bash
#SBATCH --gpus 1
#SBATCH --mem 32G
#SBATCH --cpus-per-gpu 8
#SBATCH --time 10:0:0

# Dockerfile @ https://github.com/NVIDIA/MagnumIO/tree/main/gds/benchmarks/pytorch/docker
CONTAINER="${HOME}/containers/pytorch-gds-benchmarks.sif"
CONTAINER_RC_FILE="${HOME}/containers/singularity_torch_benchmark_rc"
SCRIPT="./run_benchmark.sh"

cd "${HOME}/praca-mgr/deepCam-inference/scripts/"

set -x

for MODE in "io"; do
  # Both "inference" and "io" mode yielded the same results
  for ENABLE_GDS in 0 1; do
    MODE=$MODE ENABLE_GDS=$ENABLE_GDS singularity run --nv --no-mount proc/sys -B /scratch/shared/pwesolowski,/run/udev:/run/udev:ro "$CONTAINER" /bin/bash --rcfile "$CONTAINER_RC_FILE" -ci "$SCRIPT"
  done
done
