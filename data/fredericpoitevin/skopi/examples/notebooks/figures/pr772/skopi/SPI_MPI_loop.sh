#!/usr/bin/env bash
#!/bin/bash

for density in 0. 0.15 0.3 0.45 0.6 0.75
do
  ntasks=16
  output_dir="/scratch/fpoitevi/skopi/6q5u/${density}/"
  mkdir -p $output_dir
  echo $output_dir
  sbatch <<EOT
#!/bin/bash

#SBATCH --partition=cryoem
#SBATCH --ntasks=$ntasks
#SBATCH --cpus-per-task=1
#SBATCH --output=${output_dir}/%j.log
#SBATCH --error=${output_dir}/%j.err
#SBATCH --gpus v100:1

nvidia-smi

singularity run --nv -B /sdf,/gpfs,/scratch,/lscratch /sdf/scratch/fpoitevi/singularity_images/skopi-ana_latest.sif /bin/bash SPI_MPI_density.sh $ntasks $density $output_dir

exit 0
EOT

done


