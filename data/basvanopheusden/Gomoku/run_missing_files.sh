#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=200-308
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=6GB
#SBATCH --job-name=Gomoku
#SBATCH --mail-type=ALL
#SBATCH --mail-user=svo213@nyu.edu
#SBATCH --output=Gomoku_%j.out

declare -A allN
allN["hvh"]=40
allN["gen"]=40
allN["eye"]=10
allN["learn1"]=50
allN["learn2"]=50
allN["learn3"]=50
allN["tai1"]=50
allN["tai2"]=40

declare -A allmodel2
allmodel2["final"]=final
allmodel2["final_noprune"]=final
allmodel2["final_nodelta"]=final
allmodel2["final_noact"]=final
allmodel2["final_notree"]=final
allmodel2["final_nocenter"]=final
allmodel2["final_no2conn"]=final
allmodel2["final_no2unc"]=final
allmodel2["final_no3"]=final
allmodel2["final_no4"]=final
allmodel2["final_nonoise"]=final
allmodel2["final_weight_hvd"]=final
allmodel2["final_triangle"]=final
allmodel2["final_drop_hvd"]=final
allmodel2["final_drop_type"]=final
allmodel2["final_opp"]=final
allmodel2["final_fixed_iters"]=final_fixed_iters
allmodel2["final_fixed_branch"]=final_fixed_branch
allmodel2["final_fixed_depth"]=final_fixed_depth
allmodel2["final_mcts_myopic"]=mcts
allmodel2["final_mcts_rand"]=mcts
allmodel2["final_drop_tile"]=final_drop_tile
allmodel2["final_optweights"]=final

for i in $(seq ${SLURM_ARRAY_TASK_ID}); do read -r line; done <missing_files.txt
arr=($line)

data=${arr[0]}
model=${arr[1]}
player=${arr[2]}
group=${arr[3]}
Nplayers=${allN[$data]}
model2=${allmodel2[$model]}

direc=$SCRATCH/Gomoku/${data}/${model}

module purge
module load matlab/2016b
export MATLABPATH=$MATLABPATH:$SCRATCH/Gomoku/Code:$SCRATCH/Gomoku/Code/mcs:$SCRATCH/Gomoku/Code/mcs/gls:$SCRATCH/Gomoku/Code/mcs/minq5:$direc

cd $direc
mkdir subject_${player}_group_${group}
cd subject_${player}_group_${group}
cp $direc/times.txt .
mkdir Output
rm Output/out*

echo $data $model $model2 $Nplayers $player $group

echo "Gomoku_optim_mcs($player,$group); getbestx($player,$group,'${data}','${model2}'); exit;" | matlab -nodisplay

echo "Done"

