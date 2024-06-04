#PBS -A IT4I-3-9
#PBS -q qnvidia
#PBS -l select=1:ncpus=16

TRAIN_DB=/scratch/ITS/binary/ILSVRC2012-256_train
VAL_DB=/scratch/ITS/binary/ILSVRC2012-256_val
MEAN_FILE=/scratch/ITS/binary/imagenet_mean.binaryproto

module load cuda
module load mkl
module load hdf5

cd $PBS_O_WORKDIR


# get individual task from tasklist with index from PBS JOB ARRAY
TASK=$(sed -n "${PBS_ARRAY_INDEX}p" $PBS_O_WORKDIR/tasklist)  

echo CONFIGURATION $TASK >&2
cd $TASK

# get last snapshot
LAST_SNAPSHOT=`ls | grep solverstate | gawk 'BEGIN{FS="[_.]"}{print $4, $0}' | sort -n  |tail -1 |cut -f 2 -d \ `

echo LAST_SNAPSHOT $LAST_SNAPSHOT  >&2


if [ ! -e train_db ]
then
    mkdir train_db
    ln -s -t ./train_db $TRAIN_DB/*
    rm ./train_db/LOCK
    touch ./train_db/LOCK
fi

if [ ! -e val_db ]
then
    mkdir val_db
    ln -s -t ./val_db $VAL_DB/*
    rm ./val_db/LOCK
    touch ./val_db/LOCK
fi

if [ ! -e ./mean.binaryproto ]
then
    ln -s $MEAN_FILE ./mean.binaryproto
fi


#run it
GLOG_logtostderr=1 train_net.bin net_solver.prototxt $LAST_SNAPSHOT >log.txt 2>&2

echo ALL_DONE $TASK  >&2

