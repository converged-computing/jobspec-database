#!/bin/bash
#SBATCH --account=def-sponsor00
##### 
# EN: Change the configuration to request 2 nodes, 2 tasks per node and 1 CPU per task
# FR: Changer la configuration pour utiliser 2 noeuds, 2 tâches par noeud et 1 CPU par tâche
#####
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000M
#SBATCH --time=5:00
#SBATCH --job-name=ex5

SRCDIR=/project/def-sponsor00/photos/
FILTERS="grayscale edges emboss negate solarize flip flop monochrome add_noise"

####
# EN: Call the executable ../filterImage.exe with mpiexec
# FR: Appeler l'exécutable ../filterImage.exe avec mpiexec
####
../filterImage.exe --srcdir $SRCDIR --files $(ls $SRCDIR) --filters $FILTERS
