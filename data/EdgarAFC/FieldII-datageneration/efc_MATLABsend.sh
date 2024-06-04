#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=thinkstation-p360
#SBATCH --nodelist=worker9
#SBATCH --output="log_get_h5.out"

srun matlab -nosplash -nodesktop -nodisplay -r "getting_h5; exit"
