#!/bin/bash --login

#PBS -l select=16
#PBS -l walltime=0:20:0
#PBS -N jenkins_ior_16
#PBS -A z17

module swap PrgEnv-cray PrgEnv-intel
module swap intel intel/15.0.2.164

cd $PBS_O_WORKDIR

# System and file system identifier
system=ARCHER
fs=fs3

# Base directories
BASE_DIR=/home/z17/z17/jenkins/ci-monitoring/monitoring/io/ior
OUTPUT_DIR=/work/z17/z17/jenkins/ci-monitoring/io/ior/shared_file
IOR=/home/z17/z17/jenkins/ci-monitoring/monitoring/io/ior/bin/ior

RESULT_DIR=${OUTPUT_DIR}/${system}/${fs}
CONFIG_NAME=shf_config.ior
CONFIG_FILE=${BASE_DIR}/shared_file/${CONFIG_NAME}
echo $CONFIG_FILE

# Make sure the results direcotry exists
if [ ! -d "${RESULT_DIR}" ];
then
   mkdir -p ${RESULT_DIR}
fi

# Basic test parameters
striping=-1
nodes=16
block=8

timestamp=$(date '+%Y%m%d%H%M%S')
TARGET="${RESULT_DIR}/data"
echo $TARGET

outfile=${RESULT_DIR}/ior_res_s${striping}_c${nodes}_b${block}_${timestamp}.dat

if [ ! -d "${RESULT_DIR}" ]; then
   mkdir -p ${RESULT_DIR}
fi

cd ${RESULT_DIR}

if [ ! -d "${TARGET}" ]; then
   mkdir -p ${TARGET}
fi
cp $CONFIG_FILE .
lfs setstripe -c ${striping} ${TARGET}
echo "${timestamp} Running: Test=${iortest}, Stripe=${striping}, nodes=${nodes}, blocksize=${block}"
echo "JobID = ${PBS_JOBID}" > ${outfile}
echo "Striping = ${striping}" >> ${outfile}
aprun -n ${nodes} -N 1 $IOR -vvv  -b ${block}g -f ${CONFIG_NAME} >> ${outfile}
rm ${CONFIG_NAME}
rm -r data

