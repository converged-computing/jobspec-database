#!/bin/bash

EPOCHS=20
LOGFILE=small_model.txt
MODEL_MODULE=ex04.small_model
TRAIN_FN_MODULE=ex04.exercise5

# run jobs locally one after another
shell() # (opti, lr, bs)
{
    CMD="python3 util.py -o $1 -l $2 -b\
        $3 -e $EPOCHS -f $LOGFILE  -m $MODEL_MODULE -t\
        $TRAIN_FN_MODULE"
    # run the shit
    ${CMD}
}

# run jobs on ikw grid
sge() # (opti, lr, bs, name)
{

    NAME=$4
    CMD="python3 util.py -o $1 -l $2 -b $3 -e 20 -f\
        /home/student/r/rdiederichse/ann-tensorflow/$LOGFILE -m $MODEL_MODULE -t\
        $TRAIN_FN_MODULE"

    SGE="#!/bin/bash\n\
        cd /home/student/r/rdiederichse/ann-tensorflow\n\
        export PATH=\$PATH:/net/store/cv/projects/software/conda/bin\n\
        source activate rdiederichse-env\n\
        ${CMD}"

    echo -e ${SGE} | qsub -l mem=4G -l cuda=1 -N ${NAME}
}

pbs()
{
    NAME=`echo $4 | cut -c -14`
    CMD="python3 util.py -o $1 -l $2 -b $3 -e 20 -f\
        /home/student/r/rdiederichse/ann-tensorflow/$LOGFILE -m $MODEL_MODULE -t\
        $TRAIN_FN_MODULE"

    PBS="#!/bin/bash\n\
        #PBS -N ${NAME}\n\
        #PBS -l select=1:ncpus=1:mem=8000:ngpus=1,place=vscatter:excl\n\
        #PBS -l walltime=1:00:00\n\
        echo \$PBS_JOBID\n\
        cd /dev/cpuset/PBSPro/\$PBS_JOBID\n\
        CPUNODES=\`cat cpuset.mems\`\n\
        CPULIST=\`cat cpuset.cpus\`\n\
        cd \$PBS_O_WORKDIR\n\
        NUMACMD=\"numactl -i \$CPUNODES -N \$CPUNODES ${CMD}\"\n\
        echo \"Command:\"\n\
        echo \"\$NUMACMD\"\n\
        \$NUMACMD"

    echo -e ${PBS} | qsub
}


for OPTIMIZER in Adadelta Adagrad Adam RMSProp
do
    for LEARN_RATE in 0.0001 0.001 0.01
    do
        for BATCH_SIZE in 32 64 128 256
        do


            NAME=iannwtf_${OPTIMIZER}_${LEARN_RATE}_${BATCH_SIZE}
            case $1 in
                shell) shell $OPTIMIZER $LEARN_RATE $BATCH_SIZE;;
                pbs) pbs $OPTIMIZER $LEARN_RATE $BATCH_SIZE $NAME;;
                sge) sge $OPTIMIZER $LEARN_RATE $BATCH_SIZE $NAME;;
                *) echo "Unknown command '$1'."; echo "Usage: $0 {shell, sge, pbs}"; exit 1
            esac
            sleep 1
        done
    done
done
