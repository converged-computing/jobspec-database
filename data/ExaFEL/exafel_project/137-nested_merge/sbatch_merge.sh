#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=merge_phase1_<tag_template>
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --constraint=haswell
#SBATCH --image=docker:mlxd/xfel:latest
#SBATCH -A lcls
#SBATCH -o 1node_<tag_template>.log
#SBATCH -e 1node_<tag_template>.err
#SBATCH --mail-user=loriordan@lbl.gov
#SBATCH --mail-type=ALL
#DW jobdw type=scratch capacity=60GB access_mode=striped
#DW stage_in source=/global/cscratch1/sd/mlxd/sept_sprint/merge_multi/TAR_95-114 destination=$DW_JOB_STRIPED/TAR_95-114 type=directory
#DW stage_out source=$DW_JOB_STRIPED/merge_multi destination=/global/cscratch1/sd/mlxd/sept_sprint/merge_multi/bb type=directory

#cd into the directory with the 4ngz* files
cd $1

#Set the path to the data on the burst buffer. Staging location can be changed by modifying DW stage_in above, where data must be based on scratch
mkdir $DW_JOB_STRIPED/subsel
#Python glob modules does not do brace expansion, so ccreate temporary soft links on BB and then using these to point to files with the given glob
for ii in $(ls $DW_JOB_STRIPED/TAR_95-114/<glob_template>);
do
  ln -sf ${ii} $DW_JOB_STRIPED/subsel/
done
export TARDATA=$DW_JOB_STRIPED/subsel/*.tar
mkdir $DW_JOB_STRIPED/merge_multi

#Copy the necessary model files to the burst buffer directory. Modify the path here to your directory with 4ngz*
cp /global/cscratch1/sd/mlxd/sept_sprint/merge_multi/4ngz* $DW_JOB_STRIPED/merge_multi
export MERGE_ROOT=$DW_JOB_STRIPED/merge_multi
export TAG=1n_merge_<tag_template>
export MULTINODE=True
mkdir -p ${MERGE_ROOT}/${TAG}
echo "SRUN START RANK=0 TIME=" $(date +%s)
srun -n 32 -c 2 --cpu_bind=cores shifter ${1}/merge_cori.sh
echo "SRUN STOP RANK=0 TIME=" $(date +%s)

#Stage 2
export MULTINODE=False

#Source the Phenix ENV
source /global/project/projectdirs/lcls/mlxd/cctbx_merge/setup_env.sh
./merge_cori.sh
