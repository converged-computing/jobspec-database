#!/bin/bash


##########################################
#                                        #
#   Output some useful job information.  #
#                                        #
##########################################

#PBS -P your_project_code
#PBS -l ncpus=40
#PBS -l mem=10GB
#PBS -l walltime=24:00:0
#PBS -l storage=scratch/kw68+gdata/kw68
#PBS -N fgenesh

module load singularity
module load blast/2.11.0

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

export SINGULARITY_BINDPATH="/path/to/be/mounted"
export SINGULARITY_BIND="/data/folder"

OUTPUT_DIR=/g/data/kw68/genome

cpu=20
GENOME=GenomeOfInterest

cd ${SINGULARITY_BIND}/${GENOME}/fgenesh
mkdir gff3
mkdir results
mkdir contigs

#1)  make sequence list
singularity exec ${SINGULARITY_BINDPATH}/singularity/fgenesh.sif split_multi_fasta.pl ${SINGULARITY_BIND}/${GENOME}/repeatmasker/${GENOME}.genome.fa.masked \
       -name seq_id \
       -dir ${SINGULARITY_BIND}/${GENOME}/fgenesh/contigs \
       -ext fa \
       -col 60 \
       -mklist ${SINGULARITY_BIND}/${GENOME}/fgenesh/contigs.list

#2) make sequence list with softmasked sequence
singularity exec  ${SINGULARITY_BINDPATH}/singularity/fgenesh.sif split_multi_fasta.pl ${SINGULARITY_BIND}/${GENOME}/repeatmasker/${GENOME}.genome.fa.masked \
        -name seq_id \
        -dir ${SINGULARITY_BIND}/${GENOME}/fgenesh/contigs \
        -ext fa.masked \
        -col 60 \
        -mklist ${SINGULARITY_BIND}/${GENOME}/fgenesh/contigsN.list

#3) run fgenesh
singularity exec  ${SINGULARITY_BINDPATH}singularity/fgenesh.sif run_pipe.pl ${SINGULARITY_BIND}/${GENOME}/fgenesh/plant.cfg \
        -l ${SINGULARITY_BIND}/${GENOME}/fgenesh/contigs/contigs.list \
        -m ${SINGULARITY_BIND}/${GENOME}/fgenesh/contigs/contigsN.list \
        -d ${SINGULARITY_BIND}/${GENOME}/fgenesh/results \
        -w ${SINGULARITY_BIND}/${GENOME}/fgenesh/tmp 2> run_plant.log

#4 run convert resn3 to gff3 format
singularity exec  ${SINGULARITY_BINDPATH}/singularity/fgenesh.sif run_fgenesh_2_gff3.pl \
        ${SINGULARITY_BIND}/${GENOME}/fgenesh/results \
        ${SINGULARITY_BIND}/${GENOME}/fgenesh/gff3 -sort -print_exons

