#!/bin/bash

#SBATCH -t 6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=20
#SBATCH -J gp-noisy-PS                                    # job name
#SBATCH --mem=32G                                # memory in GB

# sends mail when process begins, and
# when it ends. Make sure you define your email
# #SBATCH --mail-type=begin
# #SBATCH --mail-type=end
# #SBATCH --mail-user=zq@princeton.edu

module load anaconda3 intel intel-mkl
export OMP_NUM_THREADS=20
cd /home/zequnl/jia/gp_lens/
python -u run_gp.py -o /tigress/zequnl/gp_chains/Peaks_9_med_${SLURM_JOB_ID}.dat \
  -d Peaks -binmin 1 -binmax 3 -cn KN -s 2.00 --bin_center_row 1
