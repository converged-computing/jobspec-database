#!/bin/bash
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks-per-node=1    # Number of MPI ranks per node
#SBATCH --ntasks-per-socket=1  # Number of MPI ranks per socket
#SBATCH --cpus-per-task=2      # Number of HW threads per task
#SBATCH --gres=gpu:1           # Number of requested gpus per node, can vary between 1 and 4
#SBATCH --mem=10000MB          # Memory per node
#SBATCH --time 00:45:00        # Walltime, format: HH:MM:SS
#SBATCH --mem-bind=local
#SBATCH -A tra21_hackathon
#SBATCH -p m100_usr_prod
#SBATCH --qos=m100_qos_dbg     # for higher priority, max 2 hours 2 nodes
#SBATCH --exclusive            # Request node exclusive to avoid coliding with other nsys users on same node
#SBATCH -e job.%J.err
#SBATCH -o job.%J.out

#export TMPDIR=$CINECA_SCRATCH/tmp 
#mkdir -p $CINECA_SCRATCH/tmp
rm -rf /tmp/nvidia
ln -s $TMPDIR /tmp/nvidia
module load cuda/11.0
module load hpc-sdk/2021--binary

cd build/bin/Release

#fgpu2
nsys profile --force-overwrite true -o f2-boids-s3d-163k-s5 boids_spatial3D -s 5 -r 12
ncu --set full --force-overwrite -o f2-boids-s3d-163k-s5 boids_spatial3D -s 5 -r 12

#fgpu2 rtc
nsys profile --force-overwrite true -o f2-boids-rtc-s3d-163k-s5 boids_rtc_spatial3D -s 5 -r 12
ncu --set full --force-overwrite -o f2-boids-rtc-s3d-163k-s5 boids_rtc_spatial3D -s 5 -r 12

#fgpu2 bf
nsys profile --force-overwrite true -o f2-boids-bf-163k-s5 boids_bruteforce -s 5 -r 12
ncu --set full --force-overwrite -o f2-boids-bf-163k-s5 boids_bruteforce -s 5 -r 12

#fgpu2 rtc bf
nsys profile --force-overwrite true -o f2-boids-rtc-bf-163k-s5 boids_rtc_bruteforce -s 5 -r 12
ncu --set full --force-overwrite -o f2-boids-rtc-bf-163k-s5 boids_rtc_bruteforce -s 5 -r 12

rm -rf /tmp/nvidia

# Run this script with sbatch profile.sh