#! /bin/bash
#SBATCH --job-name="kT0.006"
#SBATCH --time=4:00:00
#SBATCH --partition="qmchamm"
#SBATCH --cpus-per-task=20
#SBATCH --output=kT0.006/kc.out

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
export ASE_LAMMPSRUN_COMMAND=/home/krongch2/projects/lammps/lammps/src/lmp_mpi
python -u kc.py -kT 0.006
