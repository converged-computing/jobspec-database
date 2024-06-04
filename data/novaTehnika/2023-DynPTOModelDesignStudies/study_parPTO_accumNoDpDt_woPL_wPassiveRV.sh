#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4gb
#SBATCH -t 8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simmo536@umn.edu
#SBATCH -p msismall
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err

cd ~/2023-DynPTOModelDesignStudies
module load matlab
matlab -nodisplay -r \
"iVar = ${SLURM_ARRAY_TASK_ID}; \
display(['iVar = ',num2str(iVar)]); \
SS = ${SS}; \
display(['SS = ',num2str(SS)]); \
study_parPTO_accumNoDpDt_woPL_wPassiveRV"

# Commands to use
# sbatch --export=SS=2 --array=1-675 ~/2023-DynPTOModelDesignStudies/study_parPTO_accumNoDpDt_woPL_wPassiveRV.sh
# dos2unix  study_parPTO_accumNoDpDt_woPL_wPassiveRV.sh

