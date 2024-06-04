#!/bin/bash
#SBATCH --account=proj16
#SBATCH --ntasks=16
#SBATCH --partition=prod
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --constraint=volta
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=2
#SBATCH --exclude=ldir01u01
module_suffix="$1"
acc_sync="$2"
num_gpus="$3"
netpyne_m1_prefix="$(pwd)/.."
cnrn_input_data="${netpyne_m1_prefix}/generate_coreneuron_input/coredat_${SLURM_NTASKS}/coredat"
module purge
# Get modules from the local Spack install.
spack_prefix=/gpfs/bbp.cscs.ch/project/proj16/NEURONFrontiers2021/hippocampus
module use ${spack_prefix}/spack/opt/spack/modules/tcl/linux-rhel7-x86_64
netpyne_m1_version=netpyne-m-one/0.1-20211206
module load unstable ${netpyne_m1_version}-gpu${module_suffix}
working_dir_prefix="$(pwd)"
working_dir="${working_dir_prefix}/${SLURM_JOBID}-gpu${module_suffix}-${acc_sync}"
mkdir -p "${working_dir}"
cd "${working_dir}"
cp ${netpyne_m1_prefix}/run_coreneuron_simulation/{special-core-with-mps,setup_caliper}.sh .
module list
. setup_caliper.sh
echo Launching CoreNEURON on GPU using input data ${cnrn_input_data}
if [[ ${acc_sync} == "synchronous" ]]
then
  export NVCOMPILER_ACC_SYNCHRONOUS=1
elif [[ ${acc_sync} == "asynchronous" ]]
then
  export NVCOMPILER_ACC_SYNCHRONOUS=0
else
  echo "Unknown value for 3rd argument: ${acc_sync}"
fi
srun ./special-core-with-mps.sh --datpath "${cnrn_input_data}" --tstop 1000 --mpi --gpu --cell-permute=2 --nwarp 2048 
augment_caliper_json ${netpyne_m1_version} gpu${module_suffix}-${acc_sync} "${working_dir}" "${cnrn_input_data}" "$num_gpus"
mv ../slurm-${SLURM_JOBID}.out .
