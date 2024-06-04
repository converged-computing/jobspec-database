#!/bin/bash
#SBATCH --account=XXXXXXX --ntasks-per-node 1 --cpus-per-task=1 --mem 5gb -t 24:00:00 --array 1-40
set -e -x
FWDIR="$(cd "`dirname $0`"/..; pwd)"
BASEDIR=${BASEDIR:-${FWDIR}}

#
# You need to change the sbatch line mostly the array flag, but time and memory may be worth checking as well
# You can only have running one array per temp directory
#

# START HYPER-PARAMETERS
# If you comment out a parameter, the default will be used

if [ "$#" -eq  "0" ]
  then
    echo "No arguments supplied"
    minRep=2
    maxOrder=4
    nTrees=1000
    mtry=100
    maxDepth=0
    minNode=5
    workingDir="/scratch2/IDENT/tst/tmp39_nTrees${nTrees}/"
else
    echo "Arguments suppliedd"
    minRep=$1
    maxOrder=$2
    nTrees=$3
    mtry=$4
    maxDepth=$5
    minNode=$6
#workingDir="/scratch1/projects/HB_TB_Share/roc/procan/grid_search/tmp_nTrees${nTrees}_minRep${minRep}_maxOrder${maxOrder}_mtry${mtry}_maxDepth${maxDepth}_minNode${minNode}/"
    workingDir="/scratch2/IDENT/random_search/tmp311_nTrees${nTrees}_minRep${minRep}_maxOrder${maxOrder}_mtry${mtry}_maxDepth${maxDepth}_minNode${minNode}/"
fi

# END HYPER-PARAMETERS


TASK_NUM="${SLURM_ARRAY_TASK_ID:-0}"
N_TASKS="${SLURM_ARRAY_TASK_COUNT:-0}"
TASK_TAG="$(printf '%04d' ${TASK_NUM})"


echo "Running batch task: ${TASK_NUM}  tag: $TASK_TAG"

mkdir -p "${workingDir}"


module load R/4.1.3
#module load python/3.9.4
module load python/3.11.0


#Setting the script for the parameters in the bash
finalScript="python data_prep-parallel.py --split $TASK_NUM --nSplits $N_TASKS"

if [ -n "$minRep" ]; then
	finalScript+=" --minRep $minRep"
fi

if [ -n "$maxOrder" ]; then
        finalScript+=" --maxOrder $maxOrder"
fi

if [ -n "$nTrees" ]; then
        finalScript+=" --nTrees $nTrees"
fi

if [ -n "$mtry" ]; then
        finalScript+=" --mtry $mtry"
fi

if [ -n "$maxDepth" ]; then
        finalScript+=" --maxDepth $maxDepth"
fi

if [ -n "$minNode" ]; then
        finalScript+=" --minNode $minNode"
fi

if [ -n "$workingDir" ]; then
        finalScript+=" --workingDir $workingDir"
fi

echo $finalScript
eval $finalScript


#python3.9 data_prep-parallel.py --split $TASK_NUM --nSplits $N_TASKS --minRep $minRep --maxOrder $maxOrder --nTrees $nTrees --mtry $mtry --maxDepth $maxDepth --minNode $minNode --workingDir $workingDir
