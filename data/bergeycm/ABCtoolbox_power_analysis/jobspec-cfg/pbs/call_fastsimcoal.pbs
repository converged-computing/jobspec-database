#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=0:30:00
#PBS -N fsc
#PBS -m a
#PBS -M cxb585@psu.edu
#PBS -e localhost:${PBS_O_WORKDIR}/${PBS_JOBNAME}.e${PBS_JOBID}.${PBS_ARRAYID}
#PBS -o localhost:${PBS_O_WORKDIR}/${PBS_JOBNAME}.o${PBS_JOBID}.${PBS_ARRAYID}

# Call with qsub -t 1-[number of par files] pbs/call_fastsimcoal.pbs

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

working_dir=$PBS_O_WORKDIR

module load fastsimcoal/25221

# ----------------------------------------------------------------------------------------
# --- Call snakemake
# ----------------------------------------------------------------------------------------

cd $working_dir

PAR=`ls results/par_files/*par | head -n $PBS_ARRAYID | tail -n1`

echo "Input par file is $PAR.";

# Move into results directory so output is written there
FSC_OUTDIR=results/fastsimcoal_output
mkdir -p $FSC_OUTDIR
cd $FSC_OUTDIR

fsc25 -i ../../$PAR -n 1000

exit
