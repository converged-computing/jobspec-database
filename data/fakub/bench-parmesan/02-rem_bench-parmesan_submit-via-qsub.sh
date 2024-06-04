#!/bin/bash
#
# with hyperthreading (only possible with place=exclhost; ncpus means CPU cores, this makes 128 threads, however, it might be slower):
#PBS -l select=1:ncpus=64:hyperthreading=True:mem=16gb:scratch_local=1gb:cluster=halmir -l place=exclhost
#
# no hyperthreading (not possible to turn HT OFF, workaround to select 64 threads without exclhost /runs as fast as with HT, slower for long mul/):
# #PBS -l select=1:ncpus=64:mem=16gb:scratch_local=1gb:cluster=halmir
#
#PBS -l walltime=00:10:00
#
#   Name        CPU's                           Queue                           Threads                     Rust CPU family         Clock
#
#   samson      4x Intel Xeon Platinum 8280     cerit-pbs.cerit-sc.cz           4x56 threads (224)          cascadelake             2.70 - 4.00 GHz
#   eltu        4x Intel Xeon Platinum 8260     elixir-pbs.elixir-czech.cz      4x48 threads (192)          cascadelake             2.40 - 3.90 GHz
#   elwe        2x AMD EPYC 7532                elixir-pbs.elixir-czech.cz      2x64 threads (128)          znver2                  2.40 - 3.30 GHz
#   kirke       2x AMD EPYC 7532                meta-pbs.metacentrum.cz         dtto
#   TODO        they seem to have the same number of processors, which is..?
#   halmir      1x AMD EPYC 7543                meta-pbs.metacentrum.cz         64 threads                  znver2                  2.80 - 3.70 GHz
#
#PBS -N parmesan-bench_halmir
#PBS -j oe
#PBS -m ae
#PBS -M fakubo@gmail.com

# describtion from 'man qsub' (also see https://wiki.metacentrum.cz/wiki/About_scheduling_system):
# -N ... declares a name for the job. The name specified may be up to and including 15 characters in length. It
#        must consist of printable, non white space characters with the first character alphabetic.
# -q ... defines the destination of the job (queue)
# -l ... defines the resources that are required by the job
# -j oe ... standard error stream of the job will be merged with the standard output stream
# -m ae ...  mail is sent when the job aborts or terminates
# job array: $ qsub -J 2-7:2 script.sh


# ------------------------------------------------------------------------------
#
#   Setup Variables
#

# *BEN .. compiled without any measurement feature (measurement & log only at simple_duration! -- which is not inside the lib)
BIN_C4_BEN="bench-parmesan_C4_BEN_znver2-AMD"
BIN_C8_BEN="bench-parmesan_C8_BEN_znver2-AMD"
BIN_C16_BEN="bench-parmesan_C16_BEN_znver2-AMD"
BIN_C32_BEN="bench-parmesan_C32_BEN_znver2-AMD"
# *LOG .. compiled with "log_ops" feature (measurement & log at every call of measure_duration! -- which is at every operation)
BIN_C4_LOG="bench-parmesan_C4_LOG_znver2-AMD"
BIN_C8_LOG="bench-parmesan_C8_LOG_znver2-AMD"
BIN_C16_LOG="bench-parmesan_C16_LOG_znver2-AMD"
BIN_C32_LOG="bench-parmesan_C32_LOG_znver2-AMD"
    # for Halmir, Kirke, Elwe and other AMD-based:
    #   bench-parmesan_ALL_BEN_znver2-AMD
    #   bench-parmesan_PBS_znver2-AMD
    #   bench-parmesan_ADD_znver2-AMD
    #   bench-parmesan_SGN_znver2-AMD
    #   bench-parmesan_MAX_znver2-AMD
    #   bench-parmesan_MUL_znver2-AMD
    #   bench-parmesan_SCM_znver2-AMD
    #   bench-parmesan_NN_znver2-AMD
        # for Samson, Eltu and other Intel-based:
        #   bench-parmesan_ALL_BEN_cascadelake-XEON
        #   bench-parmesan_PBS_cascadelake-XEON
        #   bench-parmesan_ADD_cascadelake-XEON
        #   bench-parmesan_SGN_cascadelake-XEON
        #   bench-parmesan_MAX_cascadelake-XEON
        #   bench-parmesan_MUL_cascadelake-XEON
        #   bench-parmesan_SCM_cascadelake-XEON
        #   bench-parmesan_NN_cascadelake-XEON

CLUSTER_NAME="halmir"   # elwe   samson   eltu   halmir

MEASURE_METHOD="dstat"   # dstat   top

MEASURE_SCRIPT="measure-$MEASURE_METHOD.sh"
CPU_STATS_SRC_LOG="raw-cpu-stats-$MEASURE_METHOD.log"
CPU_STATS_C4_LOG="raw-cpu-stats-${MEASURE_METHOD}_C4.log"
CPU_STATS_C8_LOG="raw-cpu-stats-${MEASURE_METHOD}_C8.log"
CPU_STATS_C16_LOG="raw-cpu-stats-${MEASURE_METHOD}_C16.log"
CPU_STATS_C32_LOG="raw-cpu-stats-${MEASURE_METHOD}_C32.log"

# ------------------------------------------------------------------------------


# initialize required modules (if any)
# --- not needed anymore: module add fftw/fftw-3.3.8-intel-19.0.4-532p634
# --- module add fftw/fftw-3.3.8-intel-20.0.0-au2vxr2   # does not compile with this one

# clean the SCRATCH when job finishes (and data are successfully copied out) or is killed
trap 'clean_scratch' TERM EXIT

# go to the right place
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }
cd $SCRATCHDIR

# copy files: keys, pre-compiled binary, measurement scripts
DATA_DIR="/storage/brno2/home/fakub/parallel-arithmetics-benchmark"
# copy keys
rm -rf keys
mkdir -p keys
cp \
    $DATA_DIR/keys/parm__tfhe_rs_v0_2-keys__n-742_N-2048_gamma-23_l-1_kappa-3_t-5.key \
    $DATA_DIR/keys/tfhe-rs-keys__4-8-16-32.key \
    keys/ || { echo >&2 "Error while copying input file(s)!"; exit 2; }

# copy binaries & scripts
cp \
    $DATA_DIR/bin/$BIN_C4_BEN  \
    $DATA_DIR/bin/$BIN_C8_BEN  \
    $DATA_DIR/bin/$BIN_C16_BEN \
    $DATA_DIR/bin/$BIN_C32_BEN \
    $DATA_DIR/bin/$BIN_C4_LOG  \
    $DATA_DIR/bin/$BIN_C8_LOG  \
    $DATA_DIR/bin/$BIN_C16_LOG \
    $DATA_DIR/bin/$BIN_C32_LOG \
    $DATA_DIR/dstat-with-short-intervals/dstat \
    $DATA_DIR/dstat-with-short-intervals/measure-dstat.sh \
    $DATA_DIR/dstat-with-short-intervals/measure-top.sh \
    . || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cp -r \
    $DATA_DIR/dstat-with-short-intervals/plugins \
    . || { echo >&2 "Error while copying input folder(s)!"; exit 3; }
# copy ASC's
rm -rf assets
mkdir -p assets
cp \
    $DATA_DIR/bench-parmesan/assets/asc-12.yaml \
    assets/ || { echo >&2 "Error while copying ASC's!"; exit 4; }

# add exec rights
chmod a+x $MEASURE_SCRIPT

# --------------------------------------
# run main command(s):

# benchmark without extra measurements (log goes to operations.log)
echo -e "\n>>> Running main command: benchmarking maximum performance\n"
./$BIN_C4_BEN
mv operations.log operations-bench_C4.log
./$BIN_C8_BEN
mv operations.log operations-bench_C8.log
./$BIN_C16_BEN
mv operations.log operations-bench_C16.log
./$BIN_C32_BEN
mv operations.log operations-bench_C32.log

# processor load measurements (CPU log goes to raw-cpu-stats-dstat.log, besides operations.log)
echo -e "\n>>> Running main command: CPU load & detailed measurements\n"
./$MEASURE_SCRIPT ./$BIN_C4_LOG
mv operations.log operations-${MEASURE_METHOD}_C4.log
mv $CPU_STATS_SRC_LOG $CPU_STATS_C4_LOG
./$MEASURE_SCRIPT ./$BIN_C8_LOG
mv operations.log operations-${MEASURE_METHOD}_C8.log
mv $CPU_STATS_SRC_LOG $CPU_STATS_C8_LOG
./$MEASURE_SCRIPT ./$BIN_C16_LOG
mv operations.log operations-${MEASURE_METHOD}_C16.log
mv $CPU_STATS_SRC_LOG $CPU_STATS_C16_LOG
./$MEASURE_SCRIPT ./$BIN_C32_LOG
mv operations.log operations-${MEASURE_METHOD}_C32.log
mv $CPU_STATS_SRC_LOG $CPU_STATS_C32_LOG

# --------------------------------------

# copy output log files
ts=$(date +"%y-%m-%d_%H-%M")
logpath=$DATA_DIR/logs/$CLUSTER_NAME/$ts
mkdir -p $logpath


cp \
    $CPU_STATS_C4_LOG \
    $CPU_STATS_C8_LOG \
    $CPU_STATS_C16_LOG \
    $CPU_STATS_C32_LOG \
    operations-${MEASURE_METHOD}_C4.log \
    operations-${MEASURE_METHOD}_C8.log \
    operations-${MEASURE_METHOD}_C16.log \
    operations-${MEASURE_METHOD}_C32.log \
    operations-bench_C4.log \
    operations-bench_C8.log \
    operations-bench_C16.log \
    operations-bench_C32.log \
    $logpath || { echo >&2 "Error while copying result file(s)!"; exit 6; }
    #~ $DATA_DIR || { export CLEAN_SCRATCH=false; echo >&2 "Error while copying result file(s)! Try to copy them manually."; exit 6; }
