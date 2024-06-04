#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stephen.krewson@yale.edu
#SBATCH -t 5:00:00
#SBATCH --mem-per-cpu=10g
#SBATCH -c 4

# was using -n instead of -c! (use squeue to check computer nodes)
# to check versions: module spider Matlab
module purge
module load Apps/Matlab/R2017b

# now run the program (cf. README for optimization repo)
echo "Starting..."
matlab -nodisplay -nosplash -nojvm -r 'try main(); catch; end; quit;'
echo "All done!"
