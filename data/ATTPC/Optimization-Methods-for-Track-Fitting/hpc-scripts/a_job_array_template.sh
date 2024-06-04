#!/bin/bash
#PBS -N MonteCarloTest120-125
#PBS -j oe
#PBS -o /mnt/home/mkuchera/ATTPC/ar40_jobs/MonteCarlo/MC_output.o
#PBS -l walltime=10:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=5gb
#PBS -m a
#PBS -M chchen@davidson.edu 
#PBS -t 120-122

#This script is used to submit a job array.
#The system only runs 5 jobs at a time, so to submit up to 15 jobs individually please use submit_jobs.sh

echo ${PBS_ARRAYID}

RUN_NUM=$PBS_ARRAYID
RUN_NUM_PADDED=`printf "%04d" ${PBS_ARRAYID}`

CONFIG_FILE="/mnt/home/mkuchera/ATTPC/ar40_reqfiles/config_e15503a_runs_105-137.yml"

DATA_ROOT="/mnt/research/attpc/data/e15503a/hdf5_cleaned"
OUTPUT_DIR="/mnt/research/attpc/data/e15503a/mc_test"

INPUT_FILE=${DATA_ROOT}/clean_run_${RUN_NUM_PADDED}.h5
OUTPUT_FILE=${OUTPUT_DIR}/run_${RUN_NUM_PADDED}.h5

TEMP_DIR=/tmp/${USER}
INPUT_FILE_TEMP=${TEMP_DIR}/clean_run_${RUN_NUM_PADDED}.h5
OUTPUT_FILE_TEMP=${TEMP_DIR}/run_${RUN_NUM_PADDED}.h5


if [ -e $TEMP_DIR ]; then
   echo "temp dir exists"
else
   mkdir ${TEMP_DIR}
fi


if [ -e $INPUT_FILE ]; then
   cd $OUTPUT_DIR
   cp ${INPUT_FILE} ${TEMP_DIR}/.

   /mnt/home/mkuchera/external/Python-3.6.1/python ${HOME}/ATTPC/pytpc/bin/Monte_carlo.py ${CONFIG_FILE} ${INPUT_FILE_TEMP} ${OUTPUT_FILE_TEMP}

   cp ${OUTPUT_FILE_TEMP} ${OUTPUT_FILE}
   rm ${INPUT_FILE_TEMP}
   rm ${OUTPUT_FILE_TEMP}

   echo "Data fitted successfully"
else
   echo "File does not exist"
fi
