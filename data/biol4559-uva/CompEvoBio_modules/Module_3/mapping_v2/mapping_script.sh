#!/usr/bin/env bash
#
#SBATCH -J dockerMap # A single job name for the array
#SBATCH -c 10 ### 11 cores
#SBATCH -N 1 # on one node
#SBATCH -t 72:00:00
#SBATCH --mem 90G
#SBATCH -o /project/biol4559-aob2x/mapping_scripts/COMPUTEID/logs/RunDest.%A_%a.out # Standard output
#SBATCH -e /project/biol4559-aob2x/mapping_scripts/COMPUTEID/logs/RunDest.%A_%a.err # Standard error
#SBATCH -p instructional
#SBATCH --account biol4559-aob2x

### modules
  module load singularity

###################################
# Part  1. Get Sample information #
###################################
  #SLURM_ARRAY_TASK_ID=1

  sampleId=$( cat $4 | sed "${SLURM_ARRAY_TASK_ID}q;d" | cut -f3 -d',' )
  srr=$( cat $4 | sed "${SLURM_ARRAY_TASK_ID}q;d" | cut -f1 -d',' )
  numFlies=$( cat $4 | sed "${SLURM_ARRAY_TASK_ID}q;d" | cut -f2 -d',' )


  echo ${sampleId}
  echo ${srr}
  echo ${numFlies}


###################################
# Part  2. Run Docker             #
###################################

if [ -f /project/biol4559-aob2x/mapping_output/${sampleId}.sync.gz ]; then
  echo "Job already done"
  exit
fi

if [ ! -f /project/biol4559-aob2x/mapping_output/${sampleId}.sync.gz ]; then
  echo "Running job now"

  ### If your reads are Paired End use this version (and delete the other)
    singularity run \
    $1/dest_v2.6.1_latest.sif \
    $2/${srr}_1.fastq.gz \
    $2/${srr}_2.fastq.gz \
    ${sampleId} \
    $3 \
    --cores $SLURM_CPUS_PER_TASK \
    --max-cov 0.95 \
    --min-cov 4 \
    --base-quality-threshold 25 \
    --num-flies ${numFlies} \
    --do_poolsnp


  ### If your reads are Single End use this version (and delete the other)
    singularity run \
    $1/dest_v2.6.1_latest.sif \
    $2/${srr}.fastq.gz \
    ${sampleId} \
    $3 \
    --cores $SLURM_CPUS_PER_TASK \
    --max-cov 0.95 \
    --min-cov 4 \
    --base-quality-threshold 25 \
    --num-flies ${numFlies} \
    --do_poolsnp \
    --single_end

    ### do not delete anything after this line
fi
