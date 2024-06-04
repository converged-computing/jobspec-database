#PBS -A IT4I-6-9
#PBS -q qnvidia
#PBS -l select=1:ncpus=16
#PBS -l walltime=48:00:00

RUN_TIME="47h"

TRAIN_DB="/scratch/ITS/binary/ilsvrc12-256_train_lmdb"
VAL_DB="/scratch/ITS/binary/ilsvrc12-256_val_lmdb"
MEAN_FILE="/scratch/ITS/binary/ilsvrc12-256_train_mean_lmdb"

source /home_lustre/psvoboda/ITS/PBS/utils.sh || echo "Can not load /home_lustre/psvoboda/ITS/PBS/utils.sh exiting."

# load  modules
prolog

# PBS Enviroment variable: Directory where the qsub command was executed.
run cd $PBS_O_WORKDIR

# get individual task from tasklist with index ARRAY_INDEX
TASK=$(sed -n "${TASK_ID}p" $PBS_O_WORKDIR/tasklist)

run cd ${TASK}

# get last snapshot
LAST_SNAPSHOT=$(find . -name "*.solverstate" | sed "s/\.\///" | sort -n -t "_" -k 4 | tail -n 1)

echo "Last snapshot: ${LAST_SNAPSHOT}"

SNAPSHOT_PARAM=""
#Does the snapshot exist? Then use it...
if [ -n "$LAST_SNAPSHOT" ]; then
	SNAPSHOT_PARAM="-snapshot=$LAST_SNAPSHOT"
fi

#run it
(
	run timeout $RUN_TIME caffe train -solver=net_solver.prototxt ${SNAPSHOT_PARAM} \> log_${ITERATION}.txt 2\>\&1
)

if (( $ITERATION < $ITERATION_SIZE ));
then
	run cd ..
	qsub -N "c$((ITERATION+1))_${RUN_TIME}_${TASK_ID}" -v ITERATION=$(($ITERATION+1)),TASK_ID=${TASK_ID}  -k oe -j oe run.sh
fi

