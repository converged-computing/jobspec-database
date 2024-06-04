#!/bin/bash -l
#SBATCH --account=m2859
#SBATCH --job-name=ps2cctbx
#SBATCH --nodes=5000
#SBATCH --constraint=knl,quad,cache
#SBATCH --time=00:20:00
#SBATCH --image=docker:monarin/ps2cctbx:latest
#SBATCH --exclusive
#SBATCH --qos=premium

t_start=`date +%s`

export PMI_MMAP_SYNC_WAIT_TIME=600
sbcast -p ./input/process_batch.phil /tmp/process_batch.phil
sbcast -p ./xtc_process.py /tmp/xtc_process.py
sbcast -p ./input/geom_ld91.json /tmp/geom_ld91.json
sbcast -p ./test_chk_nodes.py /tmp/test_chk_nodes.py
t_end_sbcast=`date +%s`

NODES=5000
ALL_CORES=$((NODES*68))
LIMIT=0

srun -n $ALL_CORES -c 4 --cpu_bind=cores shifter ./chk_nodes.sh 

t_end_check=`date +%s`

# Determine failed nodes from chk_nodes slurm output
failed_nodes=`./show_failed_nodes.sh`
IFS=',' read -r -a array <<< $failed_nodes
let "n_failed_nodes=${#array[@]}"
exclude_flag=""
if [ "$n_failed_nodes" -gt 0 ]; then
  exclude_flag="-x $failed_nodes "
fi
ACTIVE_NODES=$((NODES-n_failed_nodes))
ACTIVE_CORES=$((ACTIVE_NODES*68))

echo "srun -N $ACTIVE_NODES -n $ACTIVE_CORES -c 4 --cpu_bind=cores ${exclude_flag}shifter ./index_single.sh cxid9114 2 0 debug $LIMIT /tmp"

srun -N $ACTIVE_NODES -n $ACTIVE_CORES -c 4 --cpu_bind=cores ${exclude_flag}shifter ./index_single.sh cxid9114 2 0 debug $LIMIT /tmp

t_end=`date +%s`

echo PSJobCompleted sbcast $((t_end_sbcast-t_start)) chk $((t_end_check-t_end_sbcast)) index $((t_end-t_end_check)) total $((t_end-t_start)) $t_start $t_end_sbcast $t_end_check $t_end
