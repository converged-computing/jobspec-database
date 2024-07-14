#!/bin/bash

# Usage: refine_clusters.sh h5ad_file_path initial_cluster_directory output_directory

h5ad_file_path=$1
initial_cluster_directory=$2
output_directory=$3

# Loop over each initial_200 file in initial_cluster_directory
for initial_file in "${initial_cluster_directory}"/initial_200_*.json; do
    echo "Processing file: ${initial_file}"
    python refine_clusters.py -f "${h5ad_file_path}" -i "${initial_file}" -o "${output_directory}"
done
