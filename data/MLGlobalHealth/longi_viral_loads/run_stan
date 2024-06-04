#!/bin/bash

ABSOLUTE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ABSOLUTE_PATH="$(dirname $ABSOLUTE_PATH)"

# help, usage
if [[ "$1" == "-h" || "$1" == "-help" || "$1" == "--help" ]]; then
    echo "Usage: ./run_stan OPTION1=value1 OPTION2=value2 ..."
    echo "OPTIONs:"
    echo "    VL            : viremic viral load threshold                                    [default: 1000]"
    echo "    FTP           : subset to first-time participants or not                        [default: FALSE]"
    echo "    STANWAY       : either (r)stan or cmdstan                                       [default: cmdstan]"
    echo "    ALPHA         : sd for alpha prior in GP                                        [default: 1.00]"
    echo "    SHARE         : share hyperpars for FTP and ALL?                                [default: TRUE]"
    echo "    LOCAL         : run the analysis locally?                                       [default: TRUE]"
    echo "    MODELS        : models to be run hyperpars (run-gp-{prevl,supp-pop,supp-hiv})   [default: ALL]"
    echo "    ROUND         : round                                                           [default: 19]"
    echo "    CONFIDENTIAL  : whether you have access to confidential files                   [default: FALSE]"
    echo "    INDIR         : directory where github directory is located"
    echo "    OUTDIR        : directory where output will be saved"
    echo "    REFIT         : if FALSE, will use previously fitted model, good for debugging  [default: FALSE]"
    exit 0
fi

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# default options
VL="${VL:-1000}" 
FTP="${FTP:-FALSE}"
ALPHA="${ALPHA:-1.00}"
SHARE="${SHARE:-TRUE}"
REFIT="${REFIT:-FALSE}"
STANWAY="${STANWAY:-cmdstan}"
CONFIDENTIAL="${CONFIDENTIAL:-FALSE}"
INDIR="${INDIR:-$ABSOLUTE_PATH}"
OUTDIR="${OUTDIR:-/rds/general/user/ab1820/home/projects/2022/longvl}"
LOCAL="${LOCAL:-TRUE}"
MODELS="${MODELS:-run-gp-prevl run-gp-supp-hiv run-gp-supp-pop}"
ROUND="${ROUND:-19}"

#################
# OPTION CHECKS #
#################

# stop if STANWAY is not stan or cmdstan
if [[ ! "$STANWAY" =~ ^(stan|cmdstan)$ ]]; then
    echo "STANWAY must be either stan (rstan) or cmdstan (cmdstanr)."
    exit 1
fi
# stop if FTP is not True, TRUE, T, False, FALSE, F
if [[ ! "$FTP" =~ ^(TRUE|FALSE)$ ]]; then
    echo "FTP must be either TRUE or FALSE."
    exit 1
fi
# stop if REFIT is not True, TRUE, T, False, FALSE, F
if [[ ! "$REFIT" =~ ^(TRUE|FALSE)$ ]]; then
    echo "REFIT must be either TRUE or FALSE."
    exit 1
fi
# stop if SHARE is not True, TRUE, T, False, FALSE, F
if [[ ! "$SHARE" =~ ^(TRUE|FALSE)$ ]]; then
    echo "SHARE must be either TRUE or FALSE." 
    exit 1
fi

echo "Selected options:"
echo "  VL = $VL"
echo "  FTP = $FTP"
echo "  ALPHA = $ALPHA"
echo "  SHARE = $SHARE"
echo "  STANWAY = $STANWAY"
echo "  INDIR = $INDIR"
echo "  OUTDIR = $OUTDIR"
echo "  REFIT = $REFIT"
echo "  LOCAL = $LOCAL"
echo "  MODELS = $MODELS"
echo "  ROUND = $ROUND"

# "build" jobname and envname
ENVNAME="longivl"
ALPHA2=$(echo "$ALPHA" | sed 's/\.//')
if [[ "$SHARE" == "TRUE" ]]; then
    ALPHA2="${ALPHA2}sharedhyper"
fi
JOBNAME="alpha${ALPHA2}_vl_$VL"
if [[ "$STANWAY" == "cmdstan" ]]; then
	JOBNAME="${STANWAY}_${JOBNAME}"
	ENVNAME="${ENVNAME}_cmdstan"
fi
if [[ "$FTP" == "TRUE" ]]; then
    JOBNAME="$JOBNAME"_firstpart
fi
echo "  JOBNAME = $JOBNAME"

########
# MAIN #
########

SUBMIT_CMD="qsub"
mkdir $OUTDIR/$JOBNAME

for MODEL in $MODELS
do

	HEAD="#!/bin/sh
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:ompthreads=1:mem=99gb
#PBS -J 16-19
#PBS -N l-$JOBNAME-$MODEL
#PBS -j oe
module load anaconda3/personal
source activate $ENVNAME

#  job will fail as soon as error occurs in your job script and that you don’t reference unset environment variables.
set -euo pipefail

JOB_TEMP=\${EPHEMERAL}/\${PBS_JOBID}
mkdir -p \$JOB_TEMP
cd \$JOB_TEMP"

    HEAD2="#!/bin/sh
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:ompthreads=1:mem=99gb
#PBS -j oe
#PBS -N l-$JOBNAME-$MODEL-postprocessing
module load anaconda3/personal
source activate $ENVNAME

#  job will fail as soon as error occurs in your job script and that you don’t reference unset environment variables.
set -euo pipefail"

    ROUND_CMD="ROUND=\$PBS_ARRAY_INDEX"

	if [[ "$LOCAL" == "TRUE" ]]; then
	    HEAD="#!/bin/sh
conda activate $ENVNAME"
	    HEAD2=$HEAD
	    ROUND_CMD="ROUND=$ROUND"
	    SUBMIT_CMD="sh"
	fi

    cat > $OUTDIR/bash-$JOBNAME-$MODEL.pbs <<EOF
$HEAD
PWD=\$(pwd)

INDIR=$INDIR
OUTDIR=$OUTDIR
JOBNAME=$JOBNAME
$ROUND_CMD

# mkdir -p \$CWD

echo "-------------"
echo "  RUN STAN   "
echo "-------------"

Rscript \$INDIR/scripts/VL_run_$STANWAY.R \\
    --viremic-viral-load $VL \\
    --outdir \$OUTDIR/\$JOBNAME/$MODEL \\
    --$MODEL TRUE \\
    --stan-alpha $ALPHA \\
    --round \$ROUND \\
    --firstpart $FTP \\
    --refit $REFIT \\
    --shared-hyper $SHARE \\
    --confidential $CONFIDENTIAL

echo "-------------"
echo "ASSESS MIXING"
echo "-------------"

Rscript \$INDIR/scripts/VL_postprocessing_assess_mixing.R \\
    --outdir-prefix $OUTDIR \\
    --jobname $JOBNAME  \\
    --$MODEL TRUE \\
    --round \$ROUND

# cp -R --no-preserve=mode,ownership \$PWD/\$JOBNAME/. \$OUTDIR/\$JOBNAME

# submit the postprocessing once done.
cd \$OUTDIR

if [ \$(find \$(pwd) -mindepth 1 -maxdepth 1 -type f -name "*rda" | wc -l) -eq 4 ]; then  
    echo "All models have finished running. Submitting postprocessing."
    $SUBMIT_CMD bash-$JOBNAME-$MODEL-postprocessing.pbs
else
    echo "Not all models have finished running. Not submitting postprocessing."
fi

EOF

    cat > $OUTDIR/bash-$JOBNAME-$MODEL-postprocessing.pbs <<EOF
$HEAD2

INDIR=$INDIR
OUTDIR=$OUTDIR
JOBNAME=$JOBNAME
  
# main directory
# CWD=\$PWD\$JOBNAME

# write code to check there are 4 files for  ending by rda in $CWD
if [ find \$OUTDIR/\$JOBNAME -mindepth 1 -maxdepth 1 -type f -name "*rda" | wc -l == 4 ]
then
    echo "All models have finished running. Submitting postprocessing."
else
    echo "Not all models have finished running. Not submitting postprocessing."
    exit 1
fi

# would need to check that all the models are done running before VL_postprocessing.R
Rscript \$INDIR/scripts/VL_postprocessing.R \\
    --viremic-viral-load $VL \\
    --outdir \$OUTDIR/$JOBNAME \\
    --indir \$OUTDIR 

# ideally this should oly be running once. (check it is)
Rscript \$INDIR/scripts/VL_jointpostprocessing.R \\
    --viremic-viral-load $VL \\
    --outdir \$OUTDIR/$JOBNAME \\
    --indir \$OUTDIR


EOF

done

cd $OUTDIR

for MODEL in $MODELS
do
    $SUBMIT_CMD bash-${JOBNAME}-${MODEL}.pbs
done
