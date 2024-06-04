#!/bin/bash

# Set the number of processing elements (PEs) or cores
# Set the number of PEs per node
# xe is for normal nodes, xk is for GPU nodes
#PBS -l nodes=1:ppn=16:xe

# Set the wallclock time
#PBS -l walltime=0:45:00

# Use shifter queue
#PBS -l gres=shifter

# Set the PBS_JOBNAME
#PBS -N delphes

# Set the job stdout and stderr
#PBS -e "${PBS_JOBNAME}.${PBS_JOBID}.err"
#PBS -o "${PBS_JOBNAME}.${PBS_JOBID}.out"

# Set email notification on termination (e) or abort (a)
#PBS -m a
#PBS -M matthew.feickert@cern.ch

# Set allocation to charge
#PBS -A bbdz

# Ensure modern bash
module load bash
# Ensure shifter enabled
module load shifter

FINAL_STATE="ll"
PHYSICS_PROCESS="drell-yan_${FINAL_STATE}"

USER_SCRATCH="/mnt/c/scratch/sciteam/${USER}"
OUTPUT_BASE_PATH="${USER_SCRATCH}/${PHYSICS_PROCESS}/${PBS_JOBNAME}"
OUTPUT_PATH="${OUTPUT_BASE_PATH}/${PBS_JOBID}"
mkdir -p "${OUTPUT_PATH}"

# INPUT_FILE_PATH passed through by qsub -v in run_delphes.sh
if [ -z "${INPUT_FILE_PATH}" ]; then
  echo "# ERROR: Variable INPUT_FILE_PATH is required to be set"
  exit 1
fi

# $HOME is /u/sciteam/${USER}
SHIFTER_IMAGE="scailfin/delphes-python-centos:3.5.0"
shifterimg pull "${SHIFTER_IMAGE}"

# The need to edit the contents of LD_LIBRARY_PATH is to remove NVIDIA libraries
# that place versions of libOpenGL in LD_LIBRARY_PATH that conflict with the
# Delphes Docker image and give a symbol lookup error.
# c.f. https://bluewaters.ncsa.illinois.edu/shifter#remarks-on-running-apps
# c.f. https://jira.ncsa.illinois.edu/browse/BWAPPS-7234
# Note also that INPUT_FILE_PATH is going to be either a .hepmc.gz or a .hepmc
# and the file extension is determined by what is available when run_delphes.sh
# is run
aprun \
  --bypass-app-transfer \
  --pes-per-node 1 \
  --cpu-binding none \
  -- shifter \
    --clearenv \
    --image="${SHIFTER_IMAGE}" \
    --volume="${OUTPUT_PATH}":/root/data \
    --volume=/mnt/a/"${HOME}":/mnt/a/"${HOME}" \
    --env INPUT_FILE_PATH="${INPUT_FILE_PATH}" \
    --workdir=/root/data \
    -- /bin/bash -c 'export LD_LIBRARY_PATH=$(echo -e "${LD_LIBRARY_PATH//\:/\\n}" | grep -v /opt/cray/nvidia/390.46-1_1.0502.2481.1.1.gem/lib64 | tr "\n" ":") && \
        export PATH="/usr/local/venv/bin:${PATH}" && \
        printf "\n# printenv:\n" && printenv && printf "\n\n" && \
        printf "# Unzip HEPMC file if needed\n\n" && \
        if [ -f "${INPUT_FILE_PATH::-3}.gz" ]; then export INPUT_FILE_PATH="${INPUT_FILE_PATH::-3}"; gunzip "${INPUT_FILE_PATH}.gz"; fi && \
        time DelphesHepMC2 \
        /usr/local/venv/cards/delphes_card_ATLAS.tcl \
        delphes_output.root \
        "${INPUT_FILE_PATH}"'
