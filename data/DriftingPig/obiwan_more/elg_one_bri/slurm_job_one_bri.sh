#!/bin/bash -l

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH --account=cosmo
#SBATCH --image=driftingpig/obiwan_composit:v3
#SBATCH -J obiwan
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
############SBATCH --volume=/global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker/obiwan_data:/legacysurvey-0359m042
#SBATCH --mail-user=kong.291@osu.edu  
#SBATCH --mail-type=ALL

export name_for_run=elg_one_brick
export randoms_db=None #run from a fits file
export dataset=dr3
export brick=0359m042
export rowstart=0
export do_skipids=no
export do_more=no
export minid=1
export object=elg
export nobj=200

export usecores=4
export threads=$usecores
#threads=1
export CSCRATCH_OBIWAN=$CSCRATCH/obiwan_Aug/repos_for_docker
#obiwan paths
export PYTHONPATH=$CSCRATCH_OBIWAN/obiwan_code/py:$CSCRATCH_OBIWAN/legacypipe/py:$PYTHONPATH 
export obiwan_data=$CSCRATCH_OBIWAN/obiwan_data 
export obiwan_code=$CSCRATCH_OBIWAN/obiwan_code 
export obiwan_out=$CSCRATCH_OBIWAN/obiwan_out   

# Load production env
#source $CSCRATCH/obiwan_code/obiwan/bin/run_atnersc/bashrc_obiwan
export RAW_LEGACY_SURVEY_DIR=$obiwan_data/



# Redirect logs
export bri=$(echo $brick | head -c 3)
export outdir=${obiwan_out}/${name_for_run}
if [ ${do_skipids} == "no" ]; then
  if [ ${do_more} == "no" ]; then
    export rsdir=rs${rowstart}
  else
    export rsdir=more_rs${rowstart}
  fi
else
  if [ ${do_more} == "no" ]; then
    export rsdir=skip_rs${rowstart}
  else
    export rsdir=more_skip_rs${rowstart}
  fi
fi
export log=${outdir}/logs/${bri}/${brick}/${rsdir}/log.${brick}
mkdir -p $(dirname $log)
echo Logging to: $log

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Protect against astropy configs
export XDG_CONFIG_HOME=/dev/shm
srun -n $SLURM_JOB_NUM_NODES mkdir -p $XDG_CONFIG_HOME/astropy

#cd $obiwan_code/obiwan/py
#export dataset=`echo $dataset | tr '[a-z]' '[A-Z]'`
srun -n 1 -c $usecores shifter ./slurm_job_one_bri_init.sh 

echo HUI-TEST:FINISHED EVERYTHING HERE
