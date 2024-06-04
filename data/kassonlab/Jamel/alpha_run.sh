#!/bin/bash
#SBATCH -p gpu          # partition
#SBATCH --gres=gpu:v100:2    # number of GPUs
#SBATCH -N 1            # number of nodes
#SBATCH -c 16            # number of cores
#SBATCH -t 72:00:00     # time

# IMPORTANT
# run this with input FASTA and output full paths as $1 and $2
export FASTA=$1
export ALPHA_OUT=$2
export MODEL_NAMES=model_1
# use preset = full_dbs or casp14
export MODE=full_dbs

export TEMPLATE_LIMIT=2022-12-12

module purge
module load singularity/3.7.1 alphafold/2.2.0

export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

singularity run -B $ALPHAFOLD_DATA_PATH:/data -B .:/etc --pwd /app/alphafold --nv $CONTAINERDIR/alphafold-2.2.0.sif \
    --fasta_paths=$FASTA \
    --output_dir=$ALPHA_OUT \
    --model_names=$MODEL_NAMES \
    --max_template_date=$TEMPLATE_LIMIT \
    --data_dir=/data \
    --use_gpu_relax=TRUE \
    --uniref90_database_path=/data/uniref90/uniref90.fasta \
    --mgnify_database_path=/data/mgnify/mgy_clusters.fa \
    --uniclust30_database_path=/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --pdb70_database_path=/data/pdb70/pdb70 \
    --template_mmcif_dir=/data/pdb_mmcif/mmcif_files \
    --obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat
