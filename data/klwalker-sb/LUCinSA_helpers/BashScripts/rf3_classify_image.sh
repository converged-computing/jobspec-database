#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o RFclass.%N.%a.%j.out # STDOUT
#SBATCH -e RFclass.%N.%a.%j.err # STDERR
#SBATCH --job-name="RSclass"
#SBATCH --array=772,773,876,879,908,909,910,942,943,977,981,737,773,774,806,807,808,809,810,841,842,843,844,845,877,878,880,911,912,913,914,944,945,946,947,948,976,978,979,980%4

################################################################

## If a running smallish number of cells, can enter in array above and use these values as the cell input:
#CELLS="${SLURM_ARRAY_TASK_ID}"
CELLS="$(($SLURM_ARRAY_TASK_ID + 3000))"

## If running a lot of cells, can use a list. To parallelize, can split list into multiple lists in a directory:
#CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/AtaCells2.csv"
#CELLS="/home/downspout-cel/paraguay_lc/vector/sampleData/TrainingCells/${SLURM_ARRAY_TASK_ID}.csv"
#CELLS="/raida-cel/r/downspout-cel/paraguay_lc/vector/sampleData/ReadyMar4/${SLURM_ARRAY_TASK_ID}.csv"

# Settables:
FEATMOD='base4NoPoly'
SAMPMOD='base1000'
YR=2021
STARTMO=11
MODNAME="${FEATMOD}_${SAMPMOD}_${YR}"
VARDF="/home/downspout-cel/paraguay_lc/classification/RF/pixdf_${MODNAME}.csv"
#################################################################
# Probably do not change:
INDIR="/home/downspout-cel/paraguay_lc/stac/grids"
MODPATH="/home/downspout-cel/paraguay_lc/classification/RF/${MODNAME}_RFmod.joblib"
#MODPATH="/home/downspout-cel/paraguay_lc/classification/RF/base4NoPoly_base1000_2021_RFmod.joblilb"
SINGDICT='/home/downspout-cel/paraguay_lc/singleton_var_dict.json'
MODDICT='/home/downspout-cel/paraguay_lc/Feature_Models.json'
#OUTIMG="${INDIR}/00${GRID_ID}_${MODNAME}.tif"
LUT="/home/klwalker/Jupyter/LUCinSA_helpers/LUCinSA_helpers/Class_LUT.csv"

#############################################
# Turn off NumPy parallelism and rely on dask
#############################################
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# This should be sufficient for OpenBlas and MKL
export OMP_NUM_THREADS=1
################################################

# activate the virtual environment
conda activate venv.lucinsa38_pipe

# if running from installed module:
LUCinSA_helpers rf_classification --in_dir $INDIR --cell_list $CELLS --df_in $VARDF --feature_model $FEATMOD --start_yr $YR --start_mo $STARTMO --samp_model_name $SAMPMOD --feature_mod_dict $MODDICT --singleton_var_dict $SINGDICT --rf_mod $MODPATH  --img_out None --spec_indices None --si_vars None --spec_indices_pheno None  --pheno_vars None --singleton_vars None --poly_vars None --poly_var_path None --combo_bands None --lc_mod None --lut $LUT --importance_method None --ran_hold 0 --out_dir None --scratch_dir None

conda deactivate
