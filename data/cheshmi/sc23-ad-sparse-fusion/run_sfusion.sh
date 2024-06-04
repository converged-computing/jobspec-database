#!/bin/bash
#SBATCH --job-name="fusion"
#SBATCH --output="fusion_gs.%j.%N.out"
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --export=ALL
#SBATCH -t 11:00:05

# this is for stampede2, should change for other servers
#module load tacc-singularity/3.7.2

BINPATH="singularity exec --env LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64 artifact.sif /source/fusion/build/demo/"
LOGS=./logs/ 
SCRIPTPATH=./
UFDB=./mm/
LOGS=./logs/
MATLIST=./spd_list.txt

THRD=20
NUM_THREAD=20
export OMP_NUM_THREADS=20

mkdir $LOGS


bash $SCRIPTPATH/run_exp.sh "${BINPATH}/flop_counter_demo" $UFDB 8 $THRD $MATLIST > $LOGS/flop_counts.csv

bash $SCRIPTPATH/run_exp.sh "${BINPATH}/sptrsv_sptrsv_demo" $UFDB 1 $THRD $MATLIST > $LOGS/sptrsv_sptrsv.csv
bash ${SCRIPTPATH}/run_exp.sh "${BINPATH}/sptrsv_spmv_demo" $UFDB 1 $THRD $MATLIST > $LOGS/sptrsv_spmv.csv
bash $SCRIPTPATH/run_exp.sh "${BINPATH}/spmv_sptrsv_demo" $UFDB 1 $THRD $MATLIST > $LOGS/spmv_sptrsv.csv
bash $SCRIPTPATH/run_exp.sh "${BINPATH}/scal_spilu_demo" $UFDB 1 $THRD $MATLIST > $LOGS/scal_spilu0.csv
bash $SCRIPTPATH/run_exp.sh "${BINPATH}/spilu_sptrsv_demo" $UFDB 1 $THRD $MATLIST > $LOGS/spilu0_sptrsv.csv
bash $SCRIPTPATH/run_exp.sh "${BINPATH}/scal_spic0_demo" $UFDB 1 $THRD $MATLIST > $LOGS/scal_spic0.csv
bash $SCRIPTPATH/run_exp.sh "${BINPATH}/spic0_sptrsv_demo" $UFDB 1 $THRD $MATLIST > $LOGS/spic0_sptrsv.csv
bash $SCRIPTPATH/run_exp.sh "${BINPATH}/spmv_spmv_demo" $UFDB 1 $THRD $MATLIST > $LOGS/spmv_spmv_static.csv







