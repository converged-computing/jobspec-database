#!/usr/bin/env bash
#
# Authors: Anne Fouilloux and Jean Iaquinta
# Copyright University of Oslo 2021
# 
usage()
{
    echo "usage: run-noresm.bash -m|--machine betzy -c|--compset NF2000climo"
    echo "                       -r|--res f19_f19_mg17 -p|--project nn1000k"
    echo "                       --name case_name_prefix -i|--prefix /path/to/dir"
    echo "                       --singularity version [-t|--task-per-node 128]"
    echo "                       [-n|--number-ensembles 10]"
    echo "                       -d|--dry -h|--help"

}


# Default number of task per node
TASK_PER_NODE=128 

while [[ $# -gt 0 ]]; do
    case $1 in
        -m | --machine )          shift
                                  TARGET_MACHINE="$1"
                                  ;;
        -r | --res )              shift
                                  RES="$1"
                                  ;;
        -p | --project )          shift
                                  PROJECT="$1"
                                  ;;
        -t | --task-per-node )    shift
                                  TASK_PER_NODE="$1"
                                  ;;
        -i | --prefix )           shift
                                  PREFIX="$1"
                                  ;;
        -c | --compset )          shift
                                  COMPSET="$1"
                                  ;;
        -n | --number-ensembles ) shift
                                  NENSEMBLES="$1"
                                  ;;
        --name )                  shift
                                  CASE_PREFIX="$1"
                                  ;;
        --singularity )           shift
                                  SINGULARITY_VERSION="$1"
                                  ;;
        -d | --dry )              DRY_RUN=echo
                                  ;;
        -h | --help )             usage
                                  exit
                                  ;;
        * )                       usage
                                  exit 1
    esac
    shift
done


if [[ "$TARGET_MACHINE" == '' ]] || [[ "$SINGULARITY_VERSION" == "" ]] || [[ "$CASE_PREFIX" == "" ]] || [[ "$COMPSET" == '' ]] || [[ "$RES" == '' ]] || [[ "$PROJECT" == '' ]] || [[ "$PREFIX" == '' ]] ; then
  usage
  exit 1 
fi

echo "Number of ensembles: $NENSEMBLES"

# Pull container from quay.io

if [[ $SINGULARITY_VERSION = *gnu* ]]; then

  $DRY_RUN singularity pull docker://quay.io/nordicesmhub/container-noresm:${SINGULARITY_VERSION}
fi

for node in {1..8}; do

  member=1
  while [ $member -le $NENSEMBLES ]; do

    echo "Node $node and Member $member"
# Generate SLURM batch script for a given machine

    cat > noresm-singularity_${SINGULARITY_VERSION}_${node}_${member}.job <<EOF
#!/bin/bash
#
#SBATCH --account=$PROJECT
#SBATCH --job-name=noresm-container_${SINGULARITY_VERSION}_${node}_${member}
#SBATCH --time=01:00:00
#SBATCH --nodes=$node
#SBATCH --tasks-per-node=$TASK_PER_NODE
#SBATCH --export=ALL
#SBATCH --switches=1
#SBATCH --exclusive
#
module purge
module load intel/2020b
#
export KMP_STACKSIZE=64M
#
export COMPSET='$COMPSET'
export RES='$RES'
export CASENAME='$CASE_PREFIX-'\$SLURM_JOB_NUM_NODES'x128p-$COMPSET-$RES-$member'
echo $CASENAME
#
mkdir -p $PREFIX/work
mkdir -p $PREFIX/archive
mkdir -p $HOME/.cime

singularity exec --bind $PREFIX/work:/opt/esm/work,/cluster/shared/noresm/inputdata:/opt/esm/inputdata,$PREFIX/archive:/opt/esm/archive container-noresm_${SINGULARITY_VERSION}.sif /opt/esm/prepare

mpirun -np \$SLURM_NTASKS singularity exec --bind $PREFIX/work:/opt/esm/work,/cluster/shared/noresm/inputdata:/opt/esm/inputdata,$PREFIX/archive:/opt/esm/archive container-noresm_${SINGULARITY_VERSION}.sif /opt/esm/execute

EOF

    $DRY_RUN sbatch noresm-singularity_${SINGULARITY_VERSION}_${node}_${member}.job 

    member=$(($member+1))
  done
done
