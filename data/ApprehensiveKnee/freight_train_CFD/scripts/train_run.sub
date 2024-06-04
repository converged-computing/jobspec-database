#!/bin/bash

#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1,mem=1g,walltime=1:00:00
#PBS -N freight_train

source /home/meccanica/ecabiati/.bashrc

module use /software/spack/spack/share/spack/modules/linux-rocky8-sandybridge/
module load openfoam

cd /home/meccanica/ecabiati/freight_train_CFD/simulation
#sort ${PBS_NODEFILE} | uniq -c | awk '{ printf("%s\n", $2); }' > /home/meccanica/ecabiati/mpd.nodes
time ./Allrun_cluster > /home/meccanica/ecabiati/freight_train_CFD/simulation/output.txt