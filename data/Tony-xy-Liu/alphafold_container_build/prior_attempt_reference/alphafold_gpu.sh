#!/bin/bash
  
#SBATCH --time=24:00:00                  # Requested walltime
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (MPI processes)
#SBATCH --cpus-per-task=8               # Number of CPU cores per task
#SBATCH --gpus-per-node=1                        # Number of GPUs
#SBATCH --mem=64GB                      # Requested memory
#SBATCH --job-name=alphafold_ATPase               # Job name
#SBATCH --account=ad-arc-3-gpu          # Account allocation code
#SBATCH --mail-user=jiarui.li@ubc.ca    # Your email address for notifications
#SBATCH --mail-type=ALL                 #
#SBATCH --output=alphafold_gpu.ATPase.job%j.out    # Output file
#SBATCH --error=alphafold_gpu.ATPase.job%j.err      # Error file
  
################################################################################
 
cd $SLURM_SUBMIT_DIR
 
module load apptainer
DOWNLOAD_DIR="/arc/project/ad-arc-3/shared/Datasets/AlphaFold/2023_12"
apptainer exec --nv -B /arc/project -B /scratch --home=$PWD /home/jli106/project/consultation/alphafold/apptainer/alphafold.apptainer_from_def.sif \
    python /opt/alphafold/run_alphafold.py \
    --fasta_paths=input/ATPase.fasta \
    --output_dir=output \
    --data_dir=${DOWNLOAD_DIR} \
    --db_preset=full_dbs \
    --model_preset=multimer \
    --bfd_database_path=${DOWNLOAD_DIR}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --mgnify_database_path=${DOWNLOAD_DIR}/mgnify/mgy_clusters_2022_05.fa \
    --template_mmcif_dir=${DOWNLOAD_DIR}/pdb_mmcif/mmcif_files \
    --obsolete_pdbs_path=${DOWNLOAD_DIR}/pdb_mmcif/obsolete.dat \
    --pdb_seqres_database_path=${DOWNLOAD_DIR}/pdb_seqres/pdb_seqres.txt \
    --uniprot_database_path=${DOWNLOAD_DIR}/uniprot/uniprot.fasta \
    --uniref30_database_path=${DOWNLOAD_DIR}/uniref30/UniRef30_2021_03 \
    --uniref90_database_path=${DOWNLOAD_DIR}/uniref90/uniref90.fasta \
    --max_template_date=2021-12-31 \
    --use_gpu_relax='True'
