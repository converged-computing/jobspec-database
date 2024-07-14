#!/bin/bash
#SBATCH --account=p32089  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=short  ### PARTITION (buyin, short, normal, etc)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=25 ## how many cpus or processors do you need on each computer
#SBATCH --time=4:00:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=1G ## how much RAM do you need per CPU (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=151075  ## When you run squeue -u NETID this is how you can identify the job
#SBATCH --exclude=qnode0565,qnode0626,qnode0637


script_name="FDTR_input_theta_75_freq_10e6_x0_15_v16.i"

#moose_exec.sh ../purple-opt -i ${script_name} --mesh-only
#moose_exec.sh ../purple-opt -i ${script_name}

mpiexec -np ${SLURM_NTASKS} singularity exec -B /projects:/projects -B /scratch:/scratch -B /projects/p32089/singularity/moose/moose:/opt/moose /projects/p32089/singularity/moose_latest.sif /projects/p32089/MOOSE_Applications/purple/purple-opt -i ${script_name}
