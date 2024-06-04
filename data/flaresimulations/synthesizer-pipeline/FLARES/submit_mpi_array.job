#!/bin/bash -l
#SBATCH -J synthesizer_FLARES_pipeline
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH -o logs/job.%J.dump
#SBATCH -e logs/job.%J.err
#SBATCH -p cosma7-rp
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --array=0-479
#SBATCH --no-requeue
#SBATCH -t 24:00:00

module purge
module load rockport-settings
module load intel_comp/2020-update2
module load intel_mpi/2020-update2
module load ucx/1.13.0rc2
module load python/3.10.7

source /cosma7/data/dp004/dc-seey1/venvs/pyenv3.10/bin/activate

regions=({00..39})

tags=(
    "000_z015p000"
    "001_z014p000"
    "002_z013p000"
    "003_z012p000"
    "004_z011p000"
    "005_z010p000" 
    "006_z009p000"
    "007_z008p000"
    "008_z007p000" 
    "009_z006p000"
    "010_z005p000"
    "011_z004p770"
)

region_idx=$(($SLURM_ARRAY_TASK_ID/${#tags[@]}))
tag_idx=$(($SLURM_ARRAY_TASK_ID%${#tags[@]}))

out_file=./data_temp/flares_photometry_${regions[$region_idx]}_${tags[$tag_idx]}_test.hdf5

if [ -f $out_file ]; then
    echo 'File exists.'
    exit 1
else
    echo 'File does not exist. Continuing...'
fi

n_cpus=112

echo Region: ${regions[$region_idx]}
echo Tag: ${tags[$tag_idx]}
echo $out_file
echo Number of CPUs: $n_cpus

grid_name=bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03
grid_dir=/cosma7/data/dp004/dc-seey1/modules/synthesizer-sam/grids
master=/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5

mpirun -np $n_cpus python3 run_flares_dust_mpi.py ${regions[$region_idx]} ${tags[$tag_idx]} -output $out_file -grid-name $grid_name -grid-directory $grid_dir -master-file $master


echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
