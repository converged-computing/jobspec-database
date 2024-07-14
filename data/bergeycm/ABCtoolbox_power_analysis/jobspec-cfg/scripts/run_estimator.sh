#!/bin/bash

# ----------------------------------------------------------------------------------------
# --- Run ABCestimator
# ----------------------------------------------------------------------------------------

PARAM_d=$1
PARAM_s=$2
PARAM_i=$3

ITER_NUM=$4

module load abctoolbox

OBS_LINE=$((ITER_NUM + 1))

DSI_STRING=DNA${PARAM_d}_STR${PARAM_s}_IND${PARAM_i}
ESTIMATOR_INPUT=ABCestimator.$DSI_STRING.Obs${ITER_NUM}.input

SUM_STATS=results/simulated_data/ABCsampler_output
SUM_STATS=${SUM_STATS}_DNA${PARAM_d}_STR${PARAM_s}_IND${PARAM_i}.sumstats.combined.txt

echo "//inputfile for the program ABCestimator" > $ESTIMATOR_INPUT
echo "estimationType standard" >> $ESTIMATOR_INPUT
echo "simName $SUM_STATS" >> $ESTIMATOR_INPUT
echo "obsName pseudoObservedData.$DSI_STRING.Obs${ITER_NUM}.obs" >> $ESTIMATOR_INPUT
echo "params 2-12" >> $ESTIMATOR_INPUT
echo "//rejection" >> $ESTIMATOR_INPUT
#echo "numRetained 1000000" >> $ESTIMATOR_INPUT
echo "tolerance 0.001" >> $ESTIMATOR_INPUT
echo "maxReadSims 100000000" >> $ESTIMATOR_INPUT
echo "//parameters for posterior estimation" >> $ESTIMATOR_INPUT
echo "diracPeakWidth 0.01" >> $ESTIMATOR_INPUT
echo "posteriorDensityPoints 200" >> $ESTIMATOR_INPUT
echo "standardizeStats 1" >> $ESTIMATOR_INPUT
echo "pruneCorrelatedStats 1" >> $ESTIMATOR_INPUT
echo "writeRetained 1" >> $ESTIMATOR_INPUT
echo "task estimate" >> $ESTIMATOR_INPUT
echo "verbose" >> $ESTIMATOR_INPUT

# All the summary stats
sed -n -e '1p' -e "${OBS_LINE}p" $SUM_STATS \
    | cut -f 13-40 > pseudoObservedData.$DSI_STRING.Obs${ITER_NUM}.obs

# Set output prefix to inclue observation number
echo "outputPrefix ABC_GLM.$DSI_STRING.Obs${ITER_NUM}_" >> $ESTIMATOR_INPUT

#ABCtoolbox $ESTIMATOR_INPUT &> /dev/null
ABCtoolbox $ESTIMATOR_INPUT &> tmp_est_output.$DSI_STRING.Obs${ITER_NUM}.txt

# Creates:
#  ABC_GLM.$DSI_STRING.Obs8_model0_BestSimsParamStats_Obs0.txt
#  ABC_GLM.$DSI_STRING.Obs8_model0_MarginalPosteriorDensities_Obs0.txt
#  ABC_GLM.$DSI_STRING.Obs8_model0_MarginalPosteriorCharacteristics.txt
#  ABC_GLM.$DSI_STRING.Obs8_modelFit.txt

#module load R
#R --vanilla $ESTIMATOR_INPUT < ~/bin/ABCtoolbox/scripts/plotPosteriorsGLM.r

# ----------------------------------------------------------------------------------------
# --- Gather important results
# ----------------------------------------------------------------------------------------

POD_INPUT=`sed -n -e "${OBS_LINE}p" $SUM_STATS`

BOUNDS=`tail -n1 ABC_GLM.$DSI_STRING.Obs${ITER_NUM}_model0_MarginalPosteriorCharacteristics.txt`

NUM_SIM_PTS=`grep "retained with" tmp_est_output.$DSI_STRING.Obs${ITER_NUM}.txt | \
    sed -e "s/.*-> //" -e "s/ simulations.*//"`

echo -e "$ITER_NUM\t$POD_INPUT\t$BOUNDS\t$NUM_SIM_PTS"

# ----------------------------------------------------------------------------------------
# --- Clean everything up
# ----------------------------------------------------------------------------------------

rm pseudoObservedData.$DSI_STRING.Obs${ITER_NUM}.obs
rm ABCestimator.$DSI_STRING.Obs${ITER_NUM}.input
rm ABC_GLM.$DSI_STRING.Obs${ITER_NUM}_model0_BestSimsParamStats_Obs0.txt
rm ABC_GLM.$DSI_STRING.Obs${ITER_NUM}_model0_MarginalPosteriorDensities_Obs0.txt
rm ABC_GLM.$DSI_STRING.Obs${ITER_NUM}_model0_MarginalPosteriorCharacteristics.txt
rm ABC_GLM.$DSI_STRING.Obs${ITER_NUM}_modelFit.txt
