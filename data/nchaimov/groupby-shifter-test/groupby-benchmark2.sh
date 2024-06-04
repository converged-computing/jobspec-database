#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 16
#SBATCH -n 16
#SBATCH -c 32
#SBATCH -t 00:15:00
#SBATCH -J GroupBy
#SBATCH -e GroupBy-%j.err
#SBATCH -o GroupBy-%j.out
#SBATCH -L SCRATCH   #Job requires $SCRATCH file system
#SBATCH -C haswell   #Use Haswell nodes
#SBATCH --image=docker:nchaimov/spark202-shifter:v8
#SBATCH --volume=/scratch1/scratchdirs/nchaimov/backingFile2:/mnt:perNodeCache=size=25G


#module load shifter 

export JAVA_HOME="/usr/lib/jvm/java-8-oracle"
export SPARK_HOME=/opt/spark
export SPARK_CONF_DIR="/global/homes/n/nchaimov/groupby-shifter-test/conf"
export MASTER=`hostname`

NUM_NODES=16
CORES_PER_NODE=32   

KVP_PER_PARTITION=1000
PARTITIONS_PER_CORE=4
VALUE_BYTES=8
SEED=50

TOTAL_EXECUTOR_CORES=$(( $NUM_NODES * $CORES_PER_NODE ))
NUM_PARTITIONS=$(( $TOTAL_EXECUTOR_CORES * $PARTITIONS_PER_CORE ))

                                                               
echo ">>>>> Starting servers"
srun -n $NUM_NODES -N $NUM_NODES -c $CORES_PER_NODE shifter $SPARK_HOME/sbin/start-cluster.sh $MASTER &

sleep 10 # wait for master to start up

echo ">>>>> Running GroupByTest"
echo ">>>>> Max nodes: $NUM_NODES Cores Per Node: $CORES_PER_NODE"
echo ">>>>> KVP Per Partition: $KVP_PER_PARTITION Partitions Per Core: $PARTITIONS_PER_CORE"
echo ">>>>> Value Bytes: $VALUE_BYTES Seed: $SEED"

CUR_NUM_NODES=1
while [ $CUR_NUM_NODES -le $NUM_NODES ] ; do
    TOTAL_EXECUTOR_CORES=$(( $CUR_NUM_NODES * $CORES_PER_NODE ))
    NUM_PARTITIONS=$(( $TOTAL_EXECUTOR_CORES * $PARTITIONS_PER_CORE ))
    echo ">>>>> Nodes: $CUR_NUM_NODES Cores: $TOTAL_EXECUTOR_CORES Partitions: $NUM_PARTITIONS"
    shifter $SPARK_HOME/bin/spark-submit --master spark://$MASTER:7077 --total-executor-cores $TOTAL_EXECUTOR_CORES --conf spark.scheduler.minRegisteredResourcesRatio=1.0 --conf spark.executor.cores=$CORES_PER_NODE $SPARK_HOME/instrumentation-benchmarks/GroupBy/target/scala-2.11/groupbytest_2.11-1.0.jar $NUM_PARTITIONS $KVP_PER_PARTITION $VALUE_BYTES $NUM_PARTITIONS $SEED
    CUR_NUM_NODES=$(( $CUR_NUM_NODES * 2 ))
    echo ">>>>> Iteration done"
done

mkdir -p $SCRATCH/spark-event-logs/$SLURM_JOB_ID
shifter cp -r /mnt/spark-event-logs $SCRATCH/spark-event-logs/$SLURM_JOB_ID

echo ">>>>> Done"
