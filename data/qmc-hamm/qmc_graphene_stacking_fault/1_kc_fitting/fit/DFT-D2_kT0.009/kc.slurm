#! /bin/bash
#SBATCH --job-name="fit/DFT-D2_kT0.009"
#SBATCH --time=14:00:00
#SBATCH --partition="physics"
#SBATCH --cpus-per-task=20
#SBATCH --output=fit/DFT-D2_kT0.009/kc.out

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR
export ASE_LAMMPSRUN_COMMAND=/home/krongch2/projects/lammps/lammps/src/lmp_mpi
python -u kc.py --method DFT-D2 -kT 0.009
