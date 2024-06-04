#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o stacpipe_ts.%N.%a.%j.out # STDOUT
#SBATCH -e stacpipe_ts.%N.%a.%j.err # STDERR
#SBATCH --job-name="stpipe_ts"
#SBATCH --array=524,543
GRIDS="$(($SLURM_ARRAY_TASK_ID + 3000))"

#VIs=("evi2")
VIs=("gcvi" "wi")
#VIs=("kndvi" "wi") 
#VIs=("nbr" "ndmi" "kndvi")
#VIs=("wi" "gcvi" "kndvi" "nbr" "ndmi")

#############################################
# Turn off NumPy parallelism and rely on dask
#############################################
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# This should be sufficient for OpenBlas and MKL
export OMP_NUM_THREADS=1
################################################

METHOD='STAC'
STEP="reconstruct"
SAT_SENSORS=S2,S2cp,LT05,LE07,LC08,LC09
NCHUNKS=512
REWRITE_WIN='True'
WIN_BATCH=1

#############
# RECONSTRUCT
#############
## REC_VI = evi2 | gcvi | wi
REC_VI="evi2"

RINPUT="ms"

START="2010-01-01"
START_PAD="2010-06-01"
END_PAD="2022-12-01"
END="2022-09-01"
SKIP_INTERVAL=7
SKIP_YEARS=1
ROVERWRITE="False"
SM_CHUNKS=512
PREFILL_GAPS='False'
DTS_MAX_WIN=61
DTS_MIN_WIN=15
PREFILL_YEARS=2
DTS_t=5
PREFILL_MAX_DAYS=50
PREFILL_WMAX=75
PREFILL_WMIN=21
RMOUT="True"
SMOOTH_METH="wh"


###############################
# DO NOT MODIFY BELOW THIS LINE
###############################

# activate the virtual environment
conda activate venv.lucinsa38_pipe
#conda activate .lucinla38_pipe
# Do for each index:
for VI in "${VIs[@]}"
do

    WIN=0
    END_WIN=16
    while [ $WIN -ne $END_WIN ]
    do

        CONFIG_UPDATES="grids:[${GRIDS}] res:${REF_RES} crs:${REF_CRS} 
        cloud_mask:sat_sensors:${SAT_SENSORS}
        main_path:/home/sandbox-cel/paraguay_lc/stac/grid
        backup_path:/home/downspout-cel/paraguay_lc/stac/grids
        dlMehod:${METHOD}
        num_workers:${SLURM_CPUS_ON_NODE} 
        io:n_chunks:${NCHUNKS} cloud_mask:reset_db:${RESET_CLOUD_DB} 
        reconstruct:rewrite_win:${REWRITE_WIN}
        reconstruct:win_batchsize:${WIN_BATCH}
        reconstruct:start_win:${WIN}
        reconstruct:input:${RINPUT} reconstruct:start_pad:${START_PAD} 
        reconstruct:end_pad:${END_PAD} reconstruct:start:${START} reconstruct:end:${END} 
        reconstruct:skip_interval:${SKIP_INTERVAL} reconstruct:skip_years:${SKIP_YEARS}
        reconstruct:vi:${VI} reconstruct:overwrite:${ROVERWRITE} reconstruct:chunks:${SM_CHUNKS}
        reconstruct:smooth_kwargs:max_window:${DTS_MAX_WIN}
        reconstruct:smooth_kwargs:min_window:${DTS_MIN_WIN}
        reconstruct:smooth_kwargs:prefill_max_years:${PREFILL_YEARS}
        reconstruct:smooth_kwargs:prefill_max_years:${PREFILL_MAX_DAYS}
        reconstruct:smooth_kwargs:prefill_wmax:${PREFILL_WMAX}
        reconstruct:smooth_kwargs:prefill_wmax:${PREFILL_WMIN}
        reconstruct:smooth_kwargs:prefill_gaps:${PREFILL_GAPS} reconstruct:smooth_kwargs:t:${DTS_t}
        reconstruct:smooth_kwargs:prefill_max_years:${PREFILL_MAX_DAYS}
        reconstruct:smooth_kwargs:prefill_wmax:${PREFILL_WMAX}
        reconstruct:smooth_kwargs:prefill_wmax:${PREFILL_WMIN}
        reconstruct:smooth_kwargs:remove_outliers:${RMOUT}
        reconstruct:smooth_kwargs:smooth_method:${SMOOTH_METH}
        clean:remove_items:${REMOVE_ITEMS}"

        tuyau $STEP --config-updates $CONFIG_UPDATES

        WIN=$(($WIN+$WIN_BATCH))
    done
done

conda deactivate
