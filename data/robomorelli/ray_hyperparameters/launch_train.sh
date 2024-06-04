#!/bin/bash

#PBS -N ray_hpo
#PBS -o ray_hpo.txt
#PBS -q gpu
#PBS -e ray_hpo_error.txt
#PBS -k oe
#PBS -m e
#PBS -M roberto.morelli.ext@leonardocompany.com
#PBS -l select=3:ngpus=4:ncpus=48

##PBS_O_WORKDIR = "${PBS_O_WORKDIR}/artificial_intelligence/repos/fdir"
##echo "PBS WORKDIR: {$PBS_O_WORKDIR}"
module load openmpi
export nodes=$(cat $PBS_NODEFILE)
echo "Execution nodes: {$nodes}"
N_NODES=`cat $PBS_NODEFILE | wc -l`
#pbsnodes -a [:queue] >>> scontrol show nodes
#cat $PBS_NODEFILE >>> Slurm_JOB_NODELIST
nodes_array=( $nodes )

node_1=${nodes_array[0]}
#################################################################
echo "Nodelist: {$PBS_NODELIST}"
echo "Nodefile: {$PBS_NODEFILE}"

cat $PBS_NODEFILE
#We want names of master and slave nodes
MASTER=`/bin/hostname -s`
echo "MASTER: {$MASTER}"

WORKER=`cat $PBS_NODEFILE | grep -v $MASTER`
echo "WORKER: {$WORKER}"
#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER $WORKER"
######################################################################

#export CUDA_VISIBLE_DECIVES=0,1,2,3
export TUNE_MAX_PENDING_TRIALS_PG=12
export prefix=10.141.1.
export port=:6379
last=$(hostname | cut -b8)
plast=$(hostname | cut -b7)
if [ $plast -eq 0 ] ; then ip_head=$prefix$last$port; else ip_head=$prefix$plast$last$port; fi
#ip_head=10.141.1.1:6379
export ip_head
redis_password=5241590000000000
export redis_password
echo "IP Head: $ip_head"

echo "STARTING HEAD at $MASTER"
ssh -q $MASTER "conda activate ray ; ray start --head"
#ray start --head
sleep 5

echo "STARTING Worker right now"
N_NODES=`echo $HOSTLIST | wc -w`
for node in $WORKER; do
        echo "now on node: {$node}"
        ssh -q $node
        export redis_password=$redis_password && export ip_head=$ip_head && echo "address to connect: {$ip_head}"
        ssh -q $node\
        conda "activate ray ; ray start --address $ip_head --redis-password $redis_password";

done
wait

##############################################################################################

#### call your code below

ssh -q $MASTER "conda activate ray ; python /davinci-1/home/morellir/artificial_intelligence/repos/fdir/main.py \
 --address $ip_head --password $redis_password --config_file $model_name"