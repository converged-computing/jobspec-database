#!/bin/bash

# Usage: predict_clusters.sh h5ad_file_path refined_cluster_directory output_directory
# bash predict_clusters.sh runs/CRPEC_run_trial/full_dataset_1000var_pca_processed.h5ad runs/CRPEC_run_trial/refined_clusters runs/CRPEC_run_trial/predicted_clusters
#
h5ad_file_path=$1
refined_cluster_directory=$2
output_directory=$3

# Loop over each refined cluster file in refined_cluster_directory
for refined_file in "${refined_cluster_directory}"/refined_*.json; do
    echo "Processing file: ${refined_file}"
    python predict_clusters.py --full_dataset "${h5ad_file_path}" --refined_cluster "${refined_file}" --output_dir "${output_directory}"
done
