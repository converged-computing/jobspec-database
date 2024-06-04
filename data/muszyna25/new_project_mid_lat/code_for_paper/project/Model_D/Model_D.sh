#!/bin/bash 

#SBATCH --qos premium
#-SBATCH --qos regular 
#-SBATCH --qos interactive

### Specify number of nodes that is equal to number of model's replicas ###
#SBATCH -N 5

#SBATCH -t 15:00:00
#SBATCH -L SCRATCH

#-SBATCH -C knl
#SBATCH -C haswell

#SBATCH --mail-user=gmuszynski@lbl.gov
#SBATCH --mail-type=BEGIN,END

### Bash options ###
set -u

BATCH_START_TIME=$(date)
echo "[+] ------START TIME (ST): $BATCH_START_TIME------"

echo "Start cluster..."
#module load tensorflow/intel-1.13.1-py36
module load tensorflow/intel-1.15.0-py37

./startCluster.sh & sleep 60 && module load tensorflow/intel-1.15.0-py37 && python test_ipyparallel.py ${1} ${2} ${3} ${4} & wait

######### Sequential code ###########
BATCH_END_TIME=$(date)
echo "[+] ------END TIME (ET) $BATCH_END_TIME------"

