#!/bin/bash
#SBATCH -J dask
#SBATCH -o dask_%j.out
#SBATCH -p htc
#SBATCH -c 1
#SBATCH --mem=6G

# Load modules
module purge
module load singularity

# Get free port based on job ID
port=$(echo "49152+$(echo ${SLURM_JOBID} | tail -c 4)" | bc)
isfree=$(netstat -taln | grep $port)
while [[ -n "$isfree" ]]; do
  port=$[port+1]
  isfree=$(netstat -taln | grep $port)
done

# Launch Jupyter Lab
singularity exec /hpc/classes/msds/hpc/dask.simg jupyter-lab --no-browser --ip=127.0.0.1 --port=${port} &

# Print connection information
sleep 1m
cat << EOF

OpenSSH:
  1. Open new terminal.
  2. Copy and paste into terminal (without quotes):
     "ssh -C -J m2.smu.edu -L ${port}:localhost:${port} ${USER}@${HOSTNAME}"
  3. Open URL above in browser.

EOF

# Wait while Jupyter is in use
wait

