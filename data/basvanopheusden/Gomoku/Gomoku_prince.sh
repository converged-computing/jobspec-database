#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=0-199
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=6GB
#SBATCH --job-name=Gomoku
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gvg218@nyu.edu
#SBATCH --output=Gomoku_%j.out

i=$((${SLURM_ARRAY_TASK_ID}+39000))
allmodels=(final final_noprune final_nodelta final_noact final_notree final_nocenter final_no2conn final_no2unc final_no3 final_no4 final_nonoise final_weight_hvd final_triangle final_drop_hvd final_drop_type final_opp final_fixed_iters final_fixed_branch final_fixed_depth final_mcts_myopic final_mcts_rand final_drop_tile final_opt_weights)
allmodels2=(final final final final final final final final final final final final final final final final final_fixed_iters final_fixed_branch final_fixed_depth mcts mcts final_drop_tile final)
alldatas=(hvh gen eye learn1 learn2 learn3 tai1 tai2 tur1 tur2)
allNs=(40 40 10 50 50 50 50 40 30 15 15)

k=0
for j in 4600 9200 10350 16100 21850 27600 33350 37950 39675 41400; do
  if [ "$j" -lt "$i" ]; then
    k=$(($k+1));
  else break;
  fi;
done

data=${alldatas[$k]}
Nplayers=${allNs[$k]}

group=$(($i%5+1))
player=$((Nplayers-(($j-$i-1)/5)%$Nplayers-1))
l=$((22-($j-$i-1)/(5*$Nplayers)))

model=${allmodels[$l]}
model2=${allmodels2[$l]}

direc=$SCRATCH/Gomoku/${data}/${model}

module purge
module load matlab
export MATLABPATH=$MATLABPATH:$SCRATCH/Gomoku/Code:$SCRATCH/Gomoku/Code/mcs:$SCRATCH/Gomoku/Code/mcs/gls:$SCRATCH/Gomoku/Code/mcs/minq5:$direc

cd $direc
mkdir subject_${player}_group_${group}
cd subject_${player}_group_${group}
cp $direc/times.txt .
mkdir Output
rm Output/out*

echo $i $model $data $player $group

echo "Gomoku_optim_mcs($player,$group); getbestx($player,$group,'${data}','${model2}'); exit;" | matlab -nodisplay

echo "Done"

