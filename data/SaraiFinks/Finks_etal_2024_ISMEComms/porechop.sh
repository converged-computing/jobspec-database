#!/bin/bash

#SBATCH --job-name=porechop_array
#SBATCH -A              
#SBATCH -p standard                 
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=32          
#SBATCH -o myoutput_%j.out          
#SBATCH -e myerrors_%j.err          
#SBATCH --array=1-12
#SBATCH --mail-type=ALL              
#SBATCH --mail-user=email.com 

module load openmpi/4.0.3/gcc.8.4.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load anaconda/2020.07
source activate porechop 

WORKDIR=/path/to/read/files

THREAD=32

cd $WORKDIR

SAMPLE_NAME=`ls *_nanopore.fastq.gz | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`
PREFIX=`echo $SAMPLE_NAME | cut -f1 -d'_'` ;
genomeID=$(echo $SAMPLE_NAME)

	porechop -i ${PREFIX}_nanopore.fastq.gz -o ${PREFIX}_output.fastq.gz  --threads ${THREAD}
	
