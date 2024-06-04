#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -l filesystems=home
#PBS -q debug
#PBS -A radix-io

export OUTPUT_FILE=margo-gpu-regression.output
echo "" > $OUTPUT_FILE
echo "################################" >> $OUTPUT_FILE
echo "## MARGO-GPU-REGRESSION JOB RESULTS:" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

if [ -n "$CI_PROJECT_DIR" ]; then
    # we appear to be running a gitlab-ci action
    . $CI_PROJECT_DIR/sandbox/spack/share/spack/setup-env.sh
    export EXE_PATH="$CI_PROJECT_DIR/sandbox/install/bin/"
elif [ -n "$SANDBOX" ]; then
    # we appear to be running from the run-regression.sh script
    . $SANDBOX/spack/share/spack/setup-env.sh
    # Change to working directory; jobs are launched in place
    cd ${PBS_O_WORKDIR}
    export EXE_PATH="./"
else
    # can't tell what's going on
    echo "ERROR: neither the CI_PROJECT_DIR (for gitlab-ci execution) or the SANDBOX"
    echo "       (for manual command line execution) variable are set."
    exit 1
fi

spack env activate mochi-regression >> $OUTPUT_FILE
spack find -vN >> $OUTPUT_FILE

echo "=== Testing transfer from GPU memory of Host A to GPU memory of Host B ===" >> $OUTPUT_FILE
mpiexec -n 2 -ppn 1 $EXE_PATH/gpu-margo-p2p-bw -X 524288000 -x 1048576 -n "cxi://" -c 1 -D 10 >> $OUTPUT_FILE

echo "=== Testing transfer from GPU memory of Host A to main memory of Host B ===" >> $OUTPUT_FILE
mpiexec -n 2 -ppn 1 $EXE_PATH/gpu-margo-p2p-bw -X 524288000 -x 1048576 -n "cxi://" -c 1 -D 10 -j >> $OUTPUT_FILE

echo "=== Testing transfer from main memory of Host A to GPU memory of Host B ===" >> $OUTPUT_FILE
mpiexec -n 2 -ppn 1 $EXE_PATH/gpu-margo-p2p-bw -X 524288000 -x 1048576 -n "cxi://" -c 1 -D 10 -k >> $OUTPUT_FILE
