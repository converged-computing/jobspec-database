#!/bin/bash
#SBATCH -p small-g
#SBATCH -A project_465000434
#SBATCH --time=0:29:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=400GB
#SBATCH --job-name=convergence
#SBATCH --output=%x.%j.out

source paths/gbar/clang.sh
export COMPILER=clang
export GPU=V100
export OMP_NUM_THREADS=31 #$SLURM_JOB_CPUS_PER_NODE
export OMP_WAIT_POLICY=ACTIVE
export OMP_TARGET_OFFLOAD=MANDATORY
echo $SLURM_JOB_CPUS_PER_NODE

export CUDA_VISIBLE_DEVICES=0,1,2,3 #,2,3 #,1 #2,3#,4,5,6,7


for PROB in 1 2 3;
do
	make realclean; make APP=mixed PROBLEM=$PROB

	FILENAME=results/$GPU/problem_$PROB.txt
	rm -rf $FILENAME

	LEVELS=2
	for N in 9 17 33 65 129 257;# 513;# 1025;
	do	
 		./bin/mixed -x $N -y $((($N/2)+1)) -z $((($N/2)+1)) -maxiter 40 -levels $LEVELS -stats $FILENAME -length 0.125 -tol 1e-8
		LEVELS=$(($LEVELS + 1))
	done
done
matlab -nosplash -nodesktop -r "run('./results/plot_convergence.m');exit;"
