#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --job-name="Pawar_256"
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

module purge
module load openmpi/intel/4.0.5 
module load fftw/openmpi/intel/3.3.9
module load matlab/2023a

experiments=(
    "momentum-flux-no-clipping"
    "momentum-forcing-yes-clipping"
    "vorticity-forcing-no-clipping"
    "momentum-flux-yes-clipping"
    "vorticity-flux-no-clipping"
    "vorticity-forcing-yes-clipping"
    "momentum-forcing-no-clipping"
    "vorticity-flux-yes-clipping"
    #"lap_smag_clipping"
)

folder_with_runs="/scratch/pp2681/Decaying_turbulence/Experiments/Paper/Re_inf/256_filter/DSM_Pawar"
folder_with_scripts="/scratch/pp2681/Decaying_turbulence/decaying-turbulence-code/DSM_Pawar/256/scripts"
folder_with_exec="/scratch/pp2681/Decaying_turbulence/decaying-turbulence-code/DSM_Pawar/256/"

for experiment in "${experiments[@]}"; do
    new_folder="$folder_with_runs/$experiment"
    mkdir -p "$new_folder"
    cd "$new_folder"
    pwd
    n=1;
    max=50;
    while [ "$n" -le "$max" ]; do
      mkdir "model$n"
      cd "model$n"
      model_exe="$folder_with_exec/$experiment"
      echo "$model_exe"
      mpiexec "$model_exe" $n > out.txt

      # Diagnostics:
      "$folder_with_scripts/nse-pseq_bare" nse.dsq series.txt
      ##################
      cd ../
      n=`expr "$n" + 1`;
    done

    # Diagnostics
    cp "$folder_with_scripts"/*.m .
    matlab -r "run('series_average'); run('spectras'); run('calculate_PDF'); run('spectras_average'); run('PDF_average')"
done