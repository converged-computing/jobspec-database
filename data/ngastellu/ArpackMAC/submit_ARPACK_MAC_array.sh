#!/bin/bash
#SBATCH --account=def-simine
#SBATCH --time=0-01:00
#SBATCH --array=1-300
#SBATCH --output=slurm_%A-%a.out
#SBATCH --error=slurm_%A-%a.err

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export KMP_BLOCKTIME=0

nn=$SLURM_ARRAY_TASK_ID

if [[ ! -d sample-${nn} ]]; then
	mkdir sample-${nn}

else
	rm sample-${nn}/*
fi

cp run_files/* sample-${nn}
cd sample-${nn}


julia runARPACK_MAC.jl ~/scratch/clean_bigMAC/10x10/relax/relaxed_structures/bigMAC-${nn}_relaxed.xsf
