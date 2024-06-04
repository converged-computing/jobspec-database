#!/bin/bash
#SBATCH -J SY/GI/Damp
#SBATCH -A MLMI-MOJB2-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:
module load parallel
module load python/3.8 cuda/11.2 cudnn/8.1_cuda-11.2  
source /home/mojb2/GI-PVI/venv/bin/activate

#! Full path to application executable: 
application="python experiments/classification.py"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

mkdir -p slurm_logs/

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):


#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

JOBID=$SLURM_JOB_ID
LOG=slurm_logs/train-log.$JOBID
ERR=slurm_logs/train-err.$JOBID

echo "Initialising..." > $LOG

cd $workdir
echo -e "Changed directory to `pwd`.\n" >> $LOG

echo -e "JobID: $JOBID\n======" >> $LOG
echo "Time: `date`" >> $LOG
echo "Running on master node: `hostname`" >> $LOG
echo "Current directory: `pwd`" >> $LOG

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================" >> $LOG
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'` >> $LOG
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)" >> $LOG

echo -e "\nExecuting command:\n==================\n$CMD\n" >> $LOG

# Alloc one node per task; uses n_tasks to determine alloc
# parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask.log --resume"
srun="srun -N1 -n1"

#### BANK 
$srun $application --q GI -d A --prior neal --server SYNC --split A --lr 0.001 -l 10000 -g 10 --batch 256 --M=100 --damp=0.2 &

# $srun $application --q MFVI -d B --prior neal --server SYNC --split A --lr 0.001 -l 10000 -g 10 --batch 256 --damp=0.2 &
wait

echo "Time: `date`" >> $LOG
