#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 15
#SBATCH --mem-per-cpu=2048
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END
#SBATCH --partition long

module load openmpi/4.0.1
#export LD_LIBRARY_PATH=/opt/lammps-7Aug19_nnp_plumed/lib/nnp/lib


mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_131_96__0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_133_96__0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_134_96__0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_25_96__0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_26_96__0-2_0-3_500_40000_index1.lmp
rm bck.*.HILLS
