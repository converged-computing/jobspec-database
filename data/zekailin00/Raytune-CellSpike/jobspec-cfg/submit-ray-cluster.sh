#!/bin/bash
#SBATCH -C gpu -J RayTune
#SBATCH --time=00:30:00

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=1

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=80


##########################
# training configuration #
##########################

#restoreID=1967117 # comment out restoreID if starting a new training

numHparams=4
numGPU=1
globalSamples=12000
epochs=10

design=a2f791f3a_ontra3
cellName=bbp012
probeType=8inhib157c_3prB8kHz
dataPath=/global/cfs/cdirs/m2043/balewski/neuronBBP-pack8kHzRam/probe_3prB8kHz/ontra3/etype_8inhib_v1
outPath=out # if more than one submission will be running at the same time, outPaths cannot be the same

maxConcurrent=4


# adapted from https://github.com/NERSC/slurm-ray-cluster
### no training configuration below this line
####################################################################


# Load modules or your own conda environment here
module load cgpu
module load tensorflow/gpu-2.1.0-py37
module list
pip list

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]} 
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 bash start-head.sh $ip $redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i bash start-worker.sh $ip_head $redis_password &
  sleep 5
done
##############################################################################################

localSamples=$((globalSamples / numGPU))
rm -r ./$outPath

restorePath=$SCRATCH/ray_results_$cellName/$restoreID
wrkDir=$SCRATCH/ray_results_$cellName/$SLURM_JOBID

echo "Current Submission ID: $SLURM_JOBID"
echo "Restore Path: $restorePath"
echo Work Directory is $wrkDir

bash -c "nvidia-smi -l 10 >&gpu_utilization-$SLURM_JOBID.smi &"

if [ -z "$restoreID" ]
then

    echo "CMD: python ./train_RayTune.py --localSamples $localSamples --dataPath $dataPath --probeType $probeType --design $design --cellName $cellName --rayResult $wrkDir --numHparams $numHparams --nodes GPU --numGPU $numGPU --epochs $epochs --outPath $outPath"

    python ./train_RayTune.py --localSamples $localSamples --dataPath $dataPath --probeType $probeType --design $design --cellName $cellName --rayResult $wrkDir --numHparams $numHparams --nodes GPU --numGPU $numGPU --epochs $epochs --outPath $outPath --maxConcurrent $maxConcurrent 

    echo "RestoreID is empty. Log files will be moved to the current submission folder"
    cd $wrkDir
    mkdir submission-$SLURM_JOBID
    cd $HOME/master
    mv ./gpu_utilization-$SLURM_JOBID.smi ./slurm-$SLURM_JOBID.out $wrkDir/submission-$SLURM_JOBID
    cp -r ./$outPath $wrkDir
    rm -r ./$outPath
    cp submit-ray-cluster.sh train_RayTune.py $wrkDir/submission-$SLURM_JOBID
    cp tune_summary_writer.py $wrkDir/experiment
    cd $wrkDir/experiment
    python ./tune_summary_writer.py --jobID $SLURM_JOBID --cellName $cellName --epochSize $globalSamples --maxConcurrent $maxConcurrent --numGPU $numGPU
    
else

    echo "CMD: python ./train_RayTune.py  --localSamples $localSamples --dataPath $dataPath --probeType $probeType --design $design --cellName $cellName --rayResult $wrkDir --numHparams $numHparams --nodes GPU --numGPU $numGPU --epochs $epochs --restorePath $restorePath --outPath $outPath "

    python ./train_RayTune.py  --localSamples $localSamples --dataPath $dataPath --probeType $probeType --design $design --cellName $cellName --rayResult $wrkDir --numHparams $numHparams --nodes GPU --numGPU $numGPU --epochs $epochs --restorePath $restorePath --outPath $outPath --maxConcurrent $maxConcurrent


    echo "Log files will be moved to $restoreID submission folder"
    cd $restorePath
    mkdir submission-$SLURM_JOBID
    cd $HOME/master
    mv ./gpu_utilization-$SLURM_JOBID.smi ./slurm-$SLURM_JOBID.out $restorePath/submission-$SLURM_JOBID
    cp submit-ray-cluster.sh train_RayTune.py $restorePath/submission-$SLURM_JOBID
    cp -r ./$outPath $restorePath
    rm -r ./$outPath
    cp submit-ray-cluster.sh train_RayTune.py $restorePath/submission-$SLURM_JOBID
    
    cp tune_summary_writer.py $restorePath/experiment
    cd $restorePath/experiment
    python ./tune_summary_writer.py --jobID $SLURM_JOBID --cellName $cellName --epochSize $globalSamples --maxConcurrent $maxConcurrent --numGPU $numGPU
    
fi



exit
