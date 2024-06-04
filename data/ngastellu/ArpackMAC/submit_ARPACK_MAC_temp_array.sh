#!/bin/bash
#SBATCH --account=def-simine
#SBATCH --time=0-10:00
#SBATCH --mem=249G
#SBATCH --array=0-4
#SBATCH --output=slurm-%a.out
#SBATCH --error=slurm-%a.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export KMP_BLOCKTIME=0

module load julia

frame1=10000
nframes=1000
step=10

Ts=(40 100 200 300 400)

T=${Ts[SLURM_ARRAY_TASK_ID]}

tempdir="${T}K_initplanar_norotate"

if [[ ! -d  $tempdir ]]; then
	mkdir "$tempdir"
fi

cd $tempdir


cp ../run_files/* .

echo "Starting at: $(date)"

julia run_QuickArpackBigMAC_MD_multiframes.jl $T $frame1 $nframes $step

echo "Ending at: $(date)"
