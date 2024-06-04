#!/bin/bash


#SBATCH --qos=gpushort
#SBATCH --job-name=plutogpu
#SBATCH --account=tipes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=/home/linushe/outputs/pluto-%j.log
#SBATCH --mem=50G

###
# This script is to submitted via "sbatch" on the cluster.
#
# Set --cpus-per-task above to match the size of your multiprocessing run, if any.
###

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Some initial setup
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
module purge

# set a random port for the notebook, in case multiple notebooks are
# on the same compute node.
NOTEBOOKPORT=`shuf -i 8000-8500 -n 1`

# set a random port for tunneling, in case multiple connections are happening
# on the same login node.
TUNNELPORT=`shuf -i 8501-9000 -n 1`

echo "On your local machine, run:"
echo ""
echo "ssh -L$NOTEBOOKPORT:localhost:$TUNNELPORT linushe@${SLURM_SUBMIT_HOST}.pik-potsdam.de -N"
echo ""
echo "To stop this notebook, run 'scancel $SLURM_JOB_ID'"

# Set up a reverse SSH tunnel from the compute node back to the submitting host (login01 or login02)
# This is the machine we will connect to with SSH forward tunneling from our client.
ssh -R$TUNNELPORT:localhost:$NOTEBOOKPORT $SLURM_SUBMIT_HOST -N -f

# Start the notebook
JULIA_REVISE_POLL=1 JULIA_DEBUG=CUDA_Driver_jll srun -n1 /home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration_gpu -e "using Pluto; Pluto.run(port=$NOTEBOOKPORT)"

# To stop the notebook, use 'scancel'
