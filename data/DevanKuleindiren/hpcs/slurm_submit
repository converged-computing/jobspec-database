#!/bin/bash
#
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J language_modelling
#! Which project should be charged (NB Wilkes projects end in '-GPU'):
##SBATCH -A COMPUTERLAB-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<=nodes*12)
##SBATCH --ntasks=2
#! How much wallclock time will be required?
##SBATCH --time=00:10:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
##SBATCH -p tesla
##SBATCH --gpu_compute_mode=3

cluster_type=$1;
script_path=$2;
shift;
shift;


#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded

if [ $cluster_type == "cpu" ]; then
	module load default-impi                 # REQUIRED - loads the basic environment
	#module load default-impi-LATEST
elif [ $cluster_type == "gpu" ]; then
	module load default-wilkes
else
	echo "Need to choose cluster type";
        exit;
fi;


module load gcc/4.8.1
module load git/2.0.0
module swap cuda cuda/8.0
module load cudnn/5.1_cuda-8.0
module load python/2.7.10

if [ $cluster_type == "cpu" ]; then
	source /home/dk503/.venv/tensorflow_cpu/bin/activate;
elif [ $cluster_type == "gpu" ]; then
	source /home/dk503/.venv/tensorflow/bin/activate;
fi;

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.
cd $workdir


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:$CUDA_INSTALL_PATH/lib64;

i=0;
for configfile in "$@"; do
	configprefix="${configfile%.*}"
	configargs=$(cat $configfile)

	echo "srun --exclusive -N1 -n1 python -u $script_path $configargs > $configprefix.log 2> $configprefix.err &"
	srun --exclusive -N1 -n1 python -u $script_path $configargs > $configprefix.log 2> $configprefix.err &
	i=$(($i + 1));
done;
wait;
