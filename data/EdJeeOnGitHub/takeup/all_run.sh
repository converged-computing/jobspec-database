#!/bin/bash
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --cpus-per-task=8        # number of cores per task (4 is default)
#SBATCH --time=0-10:00:00        # maximum time needed (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=edjee96@gmail.com
#SBATCH --output=temp/log/takeup-%A_%a.log
#SBATCH --error=temp/log/takeup-%A_%a.log
#SBATCH --export=IN_SLURM=1
#SBATCH --array=0-2          # create 10 task

LATEST_VERSION=101
VERSION=${1:-$LATEST_VERSION} # Get version from command line if provided
CMDSTAN_ARGS="--cmdstanr"
SLURM_INOUT_DIR="/project/akaring/takeup-data/data/stan_analysis_data"
ITER=800

echo "Version: $VERSION"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

if [[ -v IN_SLURM ]]; then
  echo "Running in SLURM..."

  module load -f midway2 gdal/2.4.1 udunits/2.2 proj/6.1 cmake R/4.2.0

  OUTPUT_ARGS=" --output-path=${SLURM_INOUT_DIR}"
  POSTPROCESS_INOUT_ARGS=" --input-path=${SLURM_INOUT_DIR} --output-path=${SLURM_INOUT_DIR}"
  CORES=$SLURM_CPUS_PER_TASK

  echo "Running with ${CORES} cores."
  echo "INOUT ARGS: ${POSTPROCESS_INOUT_ARGS}."
else
  OUTPUT_ARGS="--output-path=data/stan_analysis_data"
  POSTPROCESS_INOUT_ARGS=
  CORES=8
fi

STAN_THREADS=$((SLURM_CPUS_PER_TASK / 4))

fit_model () {
  Rscript --no-save \
    --no-restore \
    --verbose \
    run_takeup.R takeup fit \
    --models=${1} \
    ${CMDSTAN_ARGS} \
    ${OUTPUT_ARGS} \
    --threads=${STAN_THREADS} \
    --outputname=dist_fit${VERSION} \
    --num-mix-groups=1 \
    --chains=4 \
    --iter=${ITER} \
    --sequential > temp/log/output-${1}-fit${VERSION}.txt 2>&1
}

source quick_postprocess.sh

models=(
#  "REDUCED_FORM_NO_RESTRICT"
#  "REDUCED_FORM_NO_RESTRICT_NO_GP"
   "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP"
#  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_DIFFUSE_BETA"
#  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_DIFFUSE_BETA_DIFFUSE_CLUSTER"
#  "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_DIFFUSE_CLUSTER"
   "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_HIER_FOB"
   "STRUCTURAL_LINEAR_U_SHOCKS_PHAT_MU_REP_HIER_FIXED_FOB"
)

model=${models[${SLURM_ARRAY_TASK_ID}]}
rf_model=()
struct_model=()

fit_model ${model}

#for model in "${models[@]}"; do
#  if [[ $model == *"REDUCED"* ]]; then
#    rf_model+=("$model")
#  else
#    struct_model+=("$model")
#  fi
#done
#
#for rf_model in "${rf_model[@]}"; do
#  postprocess_rf_models "$rf_model" ${VERSION} ${POSTPROCESS_INOUT_ARGS} &
#done
#
#for s_model in "${struct_model[@]}"; do
#  postprocess_struct_models "$s_model" ${VERSION} ${POSTPROCESS_INOUT_ARGS} &
#done
#
### Wait for all background jobs to finish
#wait
