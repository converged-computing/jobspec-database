#!/bin/bash
#! Name of the job:
#SBATCH -J optim_102
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=112


#SBATCH --time=24:00:00
##SBATCH --mail-type=NONE
##SBATCH --no-requeue
 
#SBATCH --error ../diag/%x.err
#SBATCH --output ../diag/%x.out

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
 
. /etc/profile.d/modules.sh

module purge
module load rhel8/default-icl
module load intel-oneapi-mkl
module load fftw

# module load miniconda/3

# conda init
conda activate pylammps

export LD_LIBRARY_PATH=$HOME/lammps/src:$LD_LIBRARY_PATH 
export LD_LIBRARY_PATH=$HOME/.conda/envs/pylammps/lib:$LD_LIBRARY_PATH 
export PATH=$HOME/lammps/src/:$PATH


#! Full path to application executable:
application="python Python_Scripts/python_gmm_optim.py"
 
#! Run options for the application:
options="optim_param.json"

#! Work directory (i.e. where the job will run):
workdir=/home/ir-tiru1/Samanyu/WHHe_Fitting/git_folder
 
#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 228:
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
CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
#CMD="$application $options"
 
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

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi
 
echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
 
echo -e "\nExecuting command:\n==================\n$CMD\n"
 
eval $CMD
 heart 1