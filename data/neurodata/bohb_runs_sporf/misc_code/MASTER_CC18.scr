#!/bin/bash -l

#SBATCH
#SBATCH --job-name=BOHB
#SBATCH --time=0:30:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=shared,parallel,lrgmem
#SBATCH --mail-type=end
#SBATCH --mail-user=jpatsol1@jhu.edu

##
## This is used for setup between MARCC and dev on my local box.
## 

export SHARE_DIR=test_runs

if [[ "$USER" == "jpatsol1@jhu.edu" ]]; then
	export NIC=eth4
	export NCORES=$SLURM_NTASKS
	export nJOBS=24
	export nWORKERS=1
	module load python/3.7
	source ~/env/bin/activate
	ml intel/18.0
	dataID=$(awk '{print $1}' tasksCC18_R.dat | sed "$SLURM_ARRAY_TASK_ID q;d")
elif [[ "$USER" == "JLP" ]]; then
	export NIC=en0
	export nJOBS=1
	export nWORKERS=1
	#export dataID=$1
	export dataID=$(awk '{print $1}' tasksCC18_R.dat | sed "1 q;d")

	export SHARE_DIR=test_runs
fi


python bohb_run.py --nic_name $NIC --n_jobs $nJOBS --n_workers $nWORKERS --openml_dataid $dataID --run_id $dataID --shared_directory $SHARE_DIR

##parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
##parser.add_argument('--min_budget',   type=int, help='Minimum budget used during the optimization.',    default=5)
##parser.add_argument('--max_budget',   type=int, help='Maximum budget used during the optimization.',    default=500)
##parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=10)
##parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
##parser.add_argument('--n_jobs', type=int,   help='Number of threads given to training.', default=1)
##parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
##parser.add_argument('--openml_dataid', type=int, help='OpenML-DataID.', default = 6)
##parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.')
##parser.add_argument('--run_id', type=int,   help='run ID.', default=1)
##parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.')
##
##args=parser.parse_args()
##
