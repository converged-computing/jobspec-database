#! /bin/bash -l
#SBATCH -A snic2021-5-20
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 2-00:00:00
#SBATCH -J linkage_map_vanessa_lepmap_dtol
#SBATCH -e linkage_map_vanessa_lepmap_dtol_zlim
#SBATCH --mail-user karin.nasvall@ebc.uu.se
#SBATCH --mail-type=ALL

module load bioinfo-tools
module load conda
export CONDA_ENVS_PATH=/proj/uppstore2017185/b2014034_nobackup/Karin/envs/

module load R/4.0.0
module load R_packages/4.0.0

cd /proj/uppstore2017185/b2014034_nobackup/Karin/link_map_vanessa/output/07_LepMak3r_DTOL

#activate the environment
conda activate lepmap_snake

#start the snakemake workflow (have to make all changes to rules etc before launching batch)
#first do a dryrun to check the env
snakemake --unlock
snakemake --cores 4 
wait
conda deactivate

