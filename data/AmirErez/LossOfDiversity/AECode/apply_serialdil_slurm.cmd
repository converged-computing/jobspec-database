#!/bin/bash
#SBATCH --array=1-1260
##SBATCH --array=1-151
##SBATCH --array=89-90
#SBATCH -o logs/scan_%A_%a.out
#SBATCH -N 1 # node count
#SBATCH -c 1
##SBATCH -t 03:30:00
#SBATCH -t 11:59:00
##SBATCH -t 119:59:00
#SBATCH --mem=8000

if [ $# -ne 1 ]; then
   echo "Error! Need 1 argument as rundir name. Aborting."
   exit 1
fi
rundir=$1

OFFSET=0
LINE_NUM=$(echo "$SLURM_ARRAY_TASK_ID + $OFFSET" | bc)
FILE_LINE_NUM=$(echo "$SLURM_ARRAY_TASK_ID + $OFFSET + 1" | bc)
outdir="../AEData/Raw/${rundir}"
runfile=$outdir/to_run.csv

# Check output file is not there already
line=$(sed -n "$FILE_LINE_NUM"p $runfile)
outfilestr=$(echo "$line" | cut -d "," -f 5)
outfile="$outdir/$outfilestr.mat"
echo "Offset $OFFSET ; Line $LINE_NUM ; outfile $outfile"
# if [ -e $outfile ]; then
if [ -e $outdir/copied/$outfilestr.mat ]; then   # Temp patch, replace with above
   echo "File $outfile exists. Not running."
   exit 0
fi

module load matlab/2021a mcc
export MCR_CACHE_ROOT=/tmp/$SLURM_JOB_ID
echo "Running MATLAB"

# Below, exec matlab compiled
if ! ./run_exec_SLURM_line.sh $MATLABROOT $outdir $LINE_NUM ; then
   echo "Failed MATLAB"
   exit 1
fi

# Below, exec matlab uncompiled
#if ! matlab -batch "exec_SLURM_line $outdir $LINE_NUM"; then
#   echo "Failed MATLAB"
#   exit 1
#fi

echo "-----------------------------------------------"
echo "Success. Finished."
# Below depricated because requires a new matlab license each time. Instead, use compiled version"
# Compilation instructions in https://wiki.cs.huji.ac.il/hurcs/software/matlab
#matlab -nojvm -r "load('$outdir/params.mat'); \
#           params.log10c0 = $log10c0; \
#           params.P = eval('$p_str'); \
#           params.P = params.P'; \
#           params.alpha = eval('$alpha_str');\
#           disp('A serial dilution simulation is starting...'); \
#           output = exec_serialdil(params); \
#           save('$outfile', 'params', 'output'); \
#           disp('Finished! Saved to $outdir/$outfile'); \
#           quit();"

