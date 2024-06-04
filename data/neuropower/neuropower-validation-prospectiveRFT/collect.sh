#!/bin/bash

#SBATCH --job-name=SIM.coll
#SBATCH --output=error/out.SIM.coll
#SBATCH --error=error/err.SIM.coll
#SBATCH --time=2:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --qos=russpold
#SBATCH -p russpold

#SBATCH --mail-user=joke.durnez@gmail.com
#SBATCH --mail-type=begin   # email me when the job starts
#SBATCH --mail-type=end     # email me when the job finishes

. ./config_tacc.sh
# Load modules
module use /scratch/PI/russpold/modules
source /share/PI/russpold/software/setup_all.sh
module load R/3.2.0

export PILOT=15
export FINAL=61
export EXC="2.3"
#export SEED=$SLURM_ARRAY_TASK_ID
export ADAPTIVE="predictive"
export SIMS=30
export MODALITY='SIM'
export ADAPTIVE='predictive'
export MODEL='RFT'


export OUTDIR=$(echo $RESDIR$MODALITY\_$ADAPTIVE\_$PILOT\_$EXC\_$MODEL)


# for i in $(seq 1 $SIMS)
# do
#   rm $OUTDIR/estimation_$MODALITY\_$i.csv
# done
#
# for i in $(seq 1 $SIMS)
# do
#   echo $i
#   prt=($(ls $OUTDIR/estimation_$MODALITY\_$i\_*.csv | sort -n -t _ -k 3))
#   for k in "${prt[@]}"
#   do
#     more $k >> $OUTDIR/estimation_$MODALITY\_$i.csv
#   done
# done

python -i $SCRIPTDIR/aggregate_estimation.py $PILOT $FINAL $SIMS $MODALITY $ADAPTIVE $EXC $MODEL

#Rscript $HOMEDIR\Figures/SIM_figures_NIMG.R $TABDIR $HOMEDIR $FIGDIR $EXC
Rscript $HOMEDIR\Figures/HCP_figures_NIMG.R $TABDIR $HOMEDIR $FIGDIR


#TABDIR="/scratch/users/jdurnez/power_revision/tables/"
