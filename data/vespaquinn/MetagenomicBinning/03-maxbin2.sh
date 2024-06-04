#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=maxbin2
#SBATCH --output=/!!WORKDIR!!/errorOut/maxbin2out1.o  # < CHANGE
#SBATCH --error=/!!WORKDIR!!/errorOut/maxbin2error1.e # < CHANGE
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@unibe.ch                     # < CHANGE
#-------------------------------------------------------------
# STEP 3 -  - Maxbin2
# tools: maxbin2 2.2.4
# image source: https://hub.docker.com/r/pegi3s/maxbin2
# software source: https://sourceforge.net/projects/maxbin2/
#-------------------------------------------------------------
## !! BIT TO CHANGE !! 
workdir=/path/to/workdir                              # < CHANGE
##-------------------------------------------------------

# 0 --- setup 
datasets_array=($(<datasets.txt))


for dataset in "${datasets_array[@]}"; do

outdir="${workdir}/results/03-MaxBin2/${dataset}_bins"
contigs="${workdir}/data/assemblies/${dataset}.fa.gz"
bamfile="${workdir}/results/01-indexing/${dataset}_aligned.sorted.bam"
depthfile="${workdir}/results/02-MetaBAT2/${dataset}_bins/${dataset}_depth.txt"
maxbin2_sif="${workdir}/singularity/maxbin2.sif"
mkdir -p ${outdir}
cd ${outdir}

# Run maxbin2
singularity exec \
--bind ${outdir} \
--bind ${depthfile} \
--bind ${contigs} \
--bind ${workdir} \
${maxbin2_sif} run_MaxBin.pl -contig ${contigs} -abund ${depthfile} -out ${outdir}/${dataset}_maxbin2 -thread 16
done
