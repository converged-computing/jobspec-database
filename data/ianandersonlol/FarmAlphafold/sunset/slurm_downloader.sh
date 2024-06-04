#!/bin/bash -l
#SBATCH -D /home/icanders
#SBATCH -o /home/icanders/slurm-log/downloader.txt
#SBATCH -e /home/icanders/slurm-log/downloadererrors.txt
#SBATCH -J alphafoldDownloadMove
#SBATCH -t 150:00:00
#SBATCH -n 1
#SBATCH --mem 32G
#SBATCH --partition=bmh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=icanderson@ucdavis.edu

module load spack/aria2
bash /home/icanders/alphafold/scripts/download_all_data.sh /home/icanders/alphafoldDownload/