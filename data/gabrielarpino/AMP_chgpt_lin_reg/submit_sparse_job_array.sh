#!/bin/bash
#! sbatch directives begin here ###############################
#SBATCH -J sparse_amp_only_p200_L3_Lmax4_sigma01_pl_none_sparsity_05_Delta_003_15trials
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*32)
#! The skylake/skylake-himem nodes have 32 CPUs (cores) each.
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! %A means slurm job ID and %a means array index
#SBATCH -o hpc_results/sparse_amp_only_p200_L3_Lmax4_sigma01_pl_none_sparsity_05_Delta_003_15trials_%A_%a.out
#SBATCH -e hpc_results/sparse_amp_only_p200_L3_Lmax4_sigma01_pl_none_sparsity_05_Delta_003_15trials_%A_%a.err

#! Submit a job array with index values between 0 and some integer number
#! NOTE: This must be a range, not a single number (i.e. specifying 
#! '32' here would only run one job, with index 32. Specifying '0-32'
#! would run 33 jobs, with indices 0 through 32 inclusive)
#SBATCH --array=0-9

#! For 6GB per CPU, set "-p skylake"; for 12GB per CPU, set "-p skylake-himem": 
#SBATCH -p icelake
##SBATCH -p icelake-himem
##SBATCH -p skylake

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by core number*walltime.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*32 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.
#! Each task is allocated 1 core by default, and each core is allocated 5980MB (skylake)
#! and 12030MB (skylake-himem). If this is insufficient, also specify
#! --cpus-per-task and/or --mem (the latter specifies MB per node).

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
##module load rhel7/default-peta4            # REQUIRED - loads the basic environment
module load rhel8/default-icl
# module load python/3.8
module load python/3.8.11/gcc/pqdmnzmw
## module load gcc-7.2.0-gcc-4.8.5-pqn7o2k
# module load R

# activate environment
source ~/.bashrc # this is needed for conda activate, deactivate to work
# source activate chgpts_venv_v1
source ~/rds/hpc-work/chgpts_venv_v1/bin/activate

#! Full path to application executable: 
application="~/rds/hpc-work/chgpts_venv_v1/bin/python3.8"
# application="~/.conda/envs/chgpts_venv_v1/bin/python3.7"

#! Run options for the application:
save_path=/home/xl394/rds/hpc-work/AMP/hpc_results/$SLURM_ARRAY_JOB_ID/
mkdir -p "$save_path"

# In interactive mode, the line below will only create one task on the 
# logged-in compute node. Hard code $SLURM_ARRAY_TASK_ID instead.
options="hpc_comparison_sparse.py --save_path "$save_path" \
    --num_delta 10 --delta_idx $SLURM_ARRAY_TASK_ID \
    --p 200 --sigma 0.1 --alpha 0.5 \
    --L 3 Lmax 4 --frac_Delta 0.03 --num_trials 15"
# options="hpc_comparison_sparse_diff.py --save_path "$save_path" \
#    --delta_idx 0 --sigma_w 3 --num_trials 2"

#! Work directory (i.e. where the job will run):
workdir=~/rds/hpc-work/AMP  

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 32:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
#! Notes:
#! 1. These variables influence Intel MPI only.
#! 2. Domains are non-overlapping sets of cores which map 1-1 to MPI tasks.
#! 3. I_MPI_PIN_PROCESSOR_LIST is ignored if I_MPI_PIN_DOMAIN is set.
#! 4. If MPI tasks perform better when sharing caches/sockets, try I_MPI_PIN_ORDER=compact.


#! Uncomment one choice for CMD below (add mpirun/mpiexec options if necessary):

#! Choose this for a MPI code (possibly using OpenMP) using Intel MPI.
# CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code (possibly using OpenMP) using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

# if [ "$SLURM_JOB_NODELIST" ]; then
#         #! Create a machine file:
#         export NODEFILE=`generate_pbs_nodefile`
#         cat $NODEFILE | uniq > machine.file.$JOBID
#         echo -e "\nNodes allocated:\n================"
#         echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
# fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 
