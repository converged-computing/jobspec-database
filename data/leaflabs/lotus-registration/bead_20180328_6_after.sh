#!/bin/bash
#
#SBATCH -p om_all_nodes           # partition (queue)
#SBATCH -N 1                      # number of nodes
#SBATCH -n 20                     # number of cores
#SBATCH --mem 32000              # memory pool for all cores
#SBATCH -t 2-00:00                # time (D-HH:MM)
#SBATCH -o %x.slurm.%N.%j.out        # STDOUT
#SBATCH -e %x.slurm.%N.%j.err        # STDERR
#SBATCH --mail-type=END,FAIL      # notifications for job done & fail
#SBATCH --mail-user=jkinney@mit.edu # send-to address
module add mit/matlab/2016b
cd /home/jkinney/lotus-registration
export TZ=America/New_York
matlab -nodisplay -nodesktop -nosplash -r "run('master_reg_bead_20180328_6_after.m');disp('FINISHED');exit;"
