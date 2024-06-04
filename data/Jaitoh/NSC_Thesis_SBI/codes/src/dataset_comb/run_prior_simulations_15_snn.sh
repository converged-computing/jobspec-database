#!/bin/bash
### Comment lines start with ## or #+space
### Slurm option lines start with #SBATCH
### Here are the SBATCH parameters that you should always consider:

#SBATCH --array=0-19

#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds
#SBATCH --ntasks=1

#SBATCH --mem 24G
#SBATCH --cpus-per-task=16

#SBATCH --job-name=gen_dataset_comb_15
#SBATCH --output=./cluster/uzh/prior_sim/gen_dataset_comb_15_%a.out
#SBATCH --error=./cluster/uzh/prior_sim/gen_dataset_comb_15_%a.err

# module load anaconda3
# source activate sbi

# # SLURM_ARRAY_TASK_ID=$1

alphas=({0..95..5})
gammas=({5..100..5})

# alphaArr=()
# gammaArr=()

for i in ${!alphas[@]}; do
    alpha=${alphas[$i]}
    gamma=${gammas[$i]}
    alphaArr+=($alpha)
    gammaArr+=($gamma)
    python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [${alpha},${gamma}] &
done

# alpha=${alphaArr[$SLURM_ARRAY_TASK_ID]}
# gamma=${gammaArr[$SLURM_ARRAY_TASK_ID]}

# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [${alpha},${gamma}]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [10,20]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [20,30]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [30,40]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [40,50]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [50,60]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [60,70]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [70,80]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [80,90]
# python3 -u ./src/analysis/prior_range.py --dur_list [15] --task_part [90,100]

# python3 -u ./src/analysis/prior_range.py --dur_list '[3,5,7,9]'
# python3 -u ./src/analysis/prior_range.py --dur_list '[11]' --task_part '[ 0,  50]'
# python3 -u ./src/analysis/prior_range.py --dur_list '[11]' --task_part '[50, 100]'

# python3 -u ./src/analysis/prior_range.py --dur_list '[13]' --task_part '[ 0,  20]'
# python3 -u ./src/analysis/prior_range.py --dur_list '[13]' --task_part '[20,  40]'
# python3 -u ./src/analysis/prior_range.py --dur_list '[13]' --task_part '[40,  60]'
# python3 -u ./src/analysis/prior_range.py --dur_list '[13]' --task_part '[60,  80]'
# python3 -u ./src/analysis/prior_range.py --dur_list '[13]' --task_part '[80, 100]'
