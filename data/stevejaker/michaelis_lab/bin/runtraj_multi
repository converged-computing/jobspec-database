#!/bin/bash

#SBATCH --time=00-01:00         # time (DD-HH:MM)
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH --mail-user=sjaker12@byu.edu   # email address
#SBATCH --mail-type=FAIL
#SBATCH --requeue
#SBATCH --array=1-43%1
# #SBATCH --qos=test

. /etc/profile

# Because this function is bad-A :D
loadamber 

echo "Running cpptraj.MPI for Catalyst atom ${SLURM_ARRAY_TASK_ID}"

mpirun -np 16 cpptraj.MPI -i cpptraj_C${SLURM_ARRAY_TASK_ID}_S.in

echo "Finished cpptraj.mpi"

echo "
Running location.py
"
case $1 in
	-o | -O )
	case $2 in
		"" | " " )
			outfile=all_contacts.dat
			;;
		* )
			outfile=$2
			;;
	esac
		python location.py -i distance_C_${SLURM_ARRAY_TASK_ID}_S_all.dat -o ${outfile}.dat
		;;
	* )
		python location.py -i distance_C_${SLURM_ARRAY_TASK_ID}_S_all.dat
		;;
esac

echo "
DONE!
"

