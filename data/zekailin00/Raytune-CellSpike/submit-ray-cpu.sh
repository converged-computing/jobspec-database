#!/bin/bash
#SBATCH -C haswell -J Ray-CPU --account m2043
#SBATCH -q debug -t 00:30:00 
#SBATCH --nodes=5 --tasks-per-node=1
#-SBATCH --cpus-per-task=20

trainTime=200
useDataFrac=0.05
steps=10
numHparams=10
numCPU=20 # number of CPUs used by each ray tune trial

### no training configuration below this line
####################################################################

module load tensorflow/gpu-2.1.0-py37
export HDF5_USE_FILE_LOCKING=FALSE


###
### copied from https://docs.ray.io/en/master/cluster/slurm.html
###


let "worker_num=(${SLURM_NTASKS} - 1)"

suffix='6379'
ip_head=`hostname`:$suffix
export ip_head # Exporting for latter access by trainer.py

# Start the ray head node on the node that executes this script by specifying --nodes=1 and --nodelist=`hostname`
srun --nodes=1 --ntasks=1 --nodelist=`hostname` ray start --head --block --port=6379  &
sleep 30
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

# Now we execute worker_num worker nodes on all nodes in the allocation except hostname by
# specifying --nodes=${worker_num} and --exclude=`hostname`. Use 1 task per node, so worker_num tasks in total (--ntasks=${worker_num}) 
srun --nodes=${worker_num} --ntasks=${worker_num} --exclude=`hostname` ray start --address $ip_head --block &
sleep 5


python ./train_RayTune.py --dataPath /global/homes/b/balewski/prjn/neuronBBP-pack40kHzDisc/probe_quad/bbp153 --probeType quad -t $trainTime --useDataFrac $useDataFrac --steps $steps --rayResult $SCRATCH/ray_results --numHparams $numHparams --nodes CPU --numCPU $numCPU

# Pass the total number of allocated CPUs