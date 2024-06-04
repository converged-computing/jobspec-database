#!/bin/bash
#SBATCH -N 4
#SBATCH -t 24:00:00
#SBATCH --mem 224G
#SBATCH -J simultaneous-jobsteps

# Define absolute paths for proper script execution
export MINICONDA_DIR=/home/"$USER"/miniconda3
export PROJ_DIR=/home/"$USER"/alphafold_non_docker
export DATA_DIR=/home/"$USER"/alphafold_databases
export OUTPUT_DIR=/home/"$USER"/alphafold_outputs

# Activate Conda environment for 'alphafold_non_docker' project
source "$MINICONDA_DIR"/bin/activate
conda activate alphafold_non_docker

# Establish hyperparameters for prediction
export MAX_TEMPLATE_DATE=2020-05-14
export USE_GPU=false
export REMOVE_MSAS_AFTER_USE=true

# Process each unprocessed FASTA file in the current working directory
for f in *.fasta;
  do srun -n1 -N1 -c32 --cpu-bind=cores --exclusive bash "$PROJ_DIR"/run_alphafold.sh -d "$DATA_DIR" -o "$OUTPUT_DIR" -f "$f" -t "$MAX_TEMPLATE_DATE" -g "$USE_GPU" -r "$REMOVE_MSAS_AFTER_USE" &
done;

# Prevent the first node to finish its predictions from ending all other nodes' running processes
wait
