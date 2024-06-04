
#PBS -N "optim" 
#PBS -j oe
#PBS -V
#PBS -e $PBS_O_WORKDIR/logs/qsub_log.txt
#PBS -l procs=16,mem=40gb
cd $PBS_O_WORKDIR


module load gurobi/952

LATEST_VERSION=71
VERSION=${1:-$LATEST_VERSION} # Get version from command line if provided

NUM_CORES=16

# Setting arguments
PRED_DISTANCE="" # --pred-distance
MODEL="STRUCTURAL_LINEAR_U_SHOCKS_LOG_MU_REP"
NUM_POST_DRAWS=200
POSTERIOR_MEDIAN="--posterior-median" # --posterior-median
SKIP_PREDICTION=0 # 1
SKIP_OA=0 # 1 or 0
SKIP_PP=0 # 1 or 0
RUN_TARGET_CREATION=0
RUN_ESTIMATION="--run-estimation"
WELFARE_FUNCTION="log"
CONSTRAINT_TYPE="agg"
COUNTY="siaya"
OUTPUT_PATH="${PBS_O_WORKDIR}/data/${CONSTRAINT_TYPE}-${WELFARE_FUNCTION}-${COUNTY}"
PLOT_OUTPUT_PATH="${PBS_O_WORKDIR}/plots/${CONSTRAINT_TYPE}-${WELFARE_FUNCTION}-${COUNTY}"
DATA_INPUT_NAME="${COUNTY}-experiment.rds"
CUTOFF="no-" # either no- or empty string
SOLVER="gurobi"

mkdir -p ${OUTPUT_PATH}
mkdir -p ${PLOT_OUTPUT_PATH}

set -e

if [ ${COUNTY} == "full" ]
then
    COUNTY_VAR=""
else 
    COUNTY_VAR=${COUNTY}
fi


Rscript ./optim/create-distance-data.R \
    --output-name=${DATA_INPUT_NAME} \
    --num-extra-pots=4 \
    --county=${COUNTY_VAR}

if [ ${POSTERIOR_MEDIAN} == "--posterior-median" ] 
then 
    POSTVAR="median"
else
    POSTVAR="post-draws"
fi


run_optim () {

    if [ ${CUTOFF} == "no-" ]
    then
        CUTOFF_DIST=10000
    else
        CUTOFF_DIST=2500
    fi

    # Gurobi doesn't play nice with renv atm

    if [ $SKIP_PREDICTION != 1 ]
    then
        Rscript ./optim/predict-takeup-for-optim.R \
                                    ${VERSION} \
                                    $1 \
                                    $2 \
                                    --output-name=${CUTOFF}cutoff-b-$1-mu-$2-${MODEL} \
                                    --from-csv \
                                    --num-post-draws=${NUM_POST_DRAWS} \
                                    --rep-cutoff=Inf \
                                    --dist-cutoff=${CUTOFF_DIST} \
									--num-cores=${NUM_CORES} \
                                    --type-lb=-Inf \
                                    --type-ub=Inf \
                                    --data-input-name=$DATA_INPUT_NAME \
                                    --output-path=${OUTPUT_PATH} \
                                    --model=${MODEL} \
                                    ${PRED_DISTANCE} \
                                    ${RUN_ESTIMATION} 
    fi

    if [ ${RUN_TARGET_CREATION} == 1 ]
    then

        Rscript ./optim/create-village-target.R \
            pred-demand-dist-fit${VERSION}-${CUTOFF}cutoff-b-control-mu-control-${MODEL}.csv \
            --input-path=${OUTPUT_PATH} \
            --output-path=${OUTPUT_PATH} \
            --output-basename=target-${CUTOFF}cutoff-b-control-mu-control-${MODEL} 
    fi

    if [ $SKIP_OA != 1 ]
    then
        Rscript ./optim/optimal_allocation.R  \
                                    ${POSTERIOR_MEDIAN} \
                                    --num-cores=12 \
                                    --min-cost  \
                                    --constraint-type=${CONSTRAINT_TYPE} \
                                    --target-constraint=target-${CUTOFF}cutoff-b-control-mu-control-${MODEL}.csv \
                                    --output-path=${OUTPUT_PATH} \
                                    --output-filename=${CUTOFF}cutoff-b-$1-mu-$2-${MODEL} \
                                    --input-path=${OUTPUT_PATH}  \
                                    --data-input-path=${PBS_O_WORKDIR}/data \
                                    --data-input-name=$DATA_INPUT_NAME \
                                    --time-limit=10000 \
                                    --demand-input-filename=pred-demand-dist-fit${VERSION}-${CUTOFF}cutoff-b-$1-mu-$2-${MODEL}.csv \
                                    --welfare-function=${WELFARE_FUNCTION} \
                                    --solver=${SOLVER}
    fi

    if [ $SKIP_PP != 1 ]
    then
        Rscript ./optim/postprocess_allocation.R  \
                                    --min-cost \
                                    ${POSTERIOR_MEDIAN} \
                                    --constraint-type=${CONSTRAINT_TYPE} \
                                    --welfare-function=${WELFARE_FUNCTION} \
                                    --optim-input-path=${OUTPUT_PATH} \
                                    --demand-input-path=${OUTPUT_PATH} \
                                    --optim-input-a-filename=${CUTOFF}cutoff-b-$1-mu-$2-${MODEL}-${POSTVAR}-optimal-allocation.rds \
                                    --data-input-name=$DATA_INPUT_NAME \
                                    --output-path=${PLOT_OUTPUT_PATH} \
                                    --output-basename=${CONSTRAINT_TYPE}-${WELFARE_FUNCTION}-${CUTOFF}cutoff-b-$1-mu-$2-${MODEL}-${POSTVAR} \
                                    --cutoff-type=${CUTOFF}cutoff
    fi

}


CUTOFF=""
## Cutoff
run_optim "control" "control"
run_optim "control" "bracelet"

run_optim "control" "calendar"

# run_optim "bracelet" "bracelet"


CUTOFF="no-"
# ## No Cutoff
run_optim "control" "control"
run_optim "control" "bracelet"

run_optim "control" "calendar"

# run_optim "bracelet" "bracelet"
