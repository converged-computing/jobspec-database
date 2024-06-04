#!/bin/bash
#SBATCH -J prepare-container
#SBATCH -t 2:00:00
#SBATCH --nodes 1
#SBATCH -C dgx
#SBATCH --gpus-per-task 0
#SBATCH --mem 60G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --container-image rapidsai/rapidsai-dev:23.02-cuda11.8-devel-ubuntu22.04-py3.10
#SBATCH --container-name mgr-pipeline
#SBATCH --container-writable
#SBATCH --container-mounts /home2/faculty/pwesolowski/mgr-pipeline:/mgr,/scratch/shared/pwesolowski/mgr-pipeline:/data

conda update -n base -c conda-forge conda

conda install -y -c conda-forge -c nvidia merlin-core merlin-models merlin-systems nvtabular transformers4rec tensorflow
pip install graphviz

conda env create -f /mgr/preprocess-datasets-environment.yml

sed -i '2i DISABLE_JUPYTER=true' /opt/docker/bin/entrypoint_source

mkdir -p /mgr/meta-json-parser/build
