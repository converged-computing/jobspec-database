#!/bin/bash -l
#SBATCH --account=proj16
#SBATCH -t 01:00:00 
#SBATCH --qos=bigjob
#SBATCH --exclusiv
#SBATCH --mem-per-cpu=128

module purge

echo "Running with nix/mvapich2-rdma" 
module load nix/mvapich2-rdma
module load nix/bench/ior

export BGLOCKLESSMPIO_F_TYPE=0x47504653
echo "SLURM_NTASKS: $SLURM_NTASKS" 
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_NODELIST: $SLURM_NODELIST"

outdir=$(mktemp -p . -d iorout_XXXXXX)

srun ior -a MPIIO -b 1g -w -r -F -t 64m -i 3  \
    -o $outdir/

rm -rf $outdir
