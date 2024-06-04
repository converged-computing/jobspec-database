#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --job-name=metabat2
#SBATCH --output=/!!WORKDIR!!/errorOut/metabat2_out.o  # < CHANGE
#SBATCH --error=/!!WORKDIR!!/errorOut/metabat2_error.e # < CHANGE
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@unibe.ch                      # < CHANGE
#-------------------------------------------------------------
# STEP 2 - MetaBat2
# tools: metabat2 v2.12.1
# image source: https://hub.docker.com/r/nanozoo/metabat2
# software source: https://bitbucket.org/berkeleylab/metabat/src/master/
#-------------------------------------------------------------
## !! BIT TO CHANGE !! 
workdir=/path/to/workdir                              # < CHANGE
##-------------------------------------------------------

# 0 --- setup 


# set variables
datasets_array=($(<datasets.txt))
metabat2_sif=${workdir}/singularity/metabat2.sif

# loop through datasets 
for dataset in "${datasets_array[@]}"; do
outdir="${workdir}/results/02-MetaBAT2/${dataset}_bins"
contigs="${workdir}/data/assemblies/${dataset}.fa.gz"
bamfile="${workdir}/results/01-indexing/${dataset}_indices/${dataset}_aligned.sorted.bam"
mkdir -p ${outdir}
cd ${outdir}



# 1 --- Generate a depth file 
singularity exec \
--bind ${workdir} \
${metabat2_sif} jgi_summarize_bam_contig_depths --outputDepth ${dataset}_depth.txt ${bamfile}

# Run metabat2
singularity exec \
--bind ${workdir} \
--bind ${outdir}
${metabat2_sif} metabat2 -t 16 -i ${contigs} -a ${dataset}_depth.txt -o ${outdir}/${dataset}_bin
done
