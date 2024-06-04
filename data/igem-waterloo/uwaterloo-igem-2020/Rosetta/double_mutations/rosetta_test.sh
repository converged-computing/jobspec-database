#!/bin/bash                       
#SBATCH --time=10:00:00
#SBATCH --account=def-bingalls
#SBATCH --mem 128000
#SBATCH --job-name=rosetta_test
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=jj5song@uwaterloo.ca

# path to Rosetta and input file folder
ROSETTA3="/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/MPI/intel2016.4/openmpi2.1/rosetta/3.10"
ROSETTA3_DB="/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/MPI/intel2016.4/openmpi2.1/rosetta/3.10/database"
# input_files="/home/jessys/rosetta/0923/inputs"
# output_files="/home/jessys/rosetta/0923/outputs" 
# backrub_files="/home/jessys/rosetta/0923/backrub_outputs" 
# mutated_files="/home/jessys/rosetta/1012/mutated" 
module load nixpkgs/16.09  gcc/7.3.0  openmpi/3.1.2 rosetta/3.10

# Clean PDB file with ligand using a python script in Rosetta/tools, output file added ligand HETATM afterwards
# python $ROSETTA3/tools/protein_tools/scripts/clean_pdb.py 5icu.pdb ignorechain

# Basic energy scoring jd2 based on crystallographic energy functions. Add tag to ignore phosphate and other unrecognized components.
# $ROSETTA3/bin/score_jd2.mpi.linuxiccrelease -in:file:s 5icu_HETATM.pdb -ignore_unrecognized_res -out:path:all $output_files -out:suffix _original

#relax the structure based on internal structure scoring 
# $ROSETTA3/bin/relax.mpi.linuxiccrelease -in:file:s 5icu_ignorechain.pdb -out:path:all $output_files -nstruct 10
#Backrub 
# $ROSETTA3/bin/backrub.mpi.linuxiccrelease -in:file:s 5icu_HETATM_0010.pdb -out:suffix _backrubbed -nstruct 10 -in:file:fullatom -backrub:ntrials 10000 -out:path:all $backrub_files

#Point mutation
$ROSETTA3/bin/pmut_scan_parallel.mpi.linuxiccrelease -in:file:s 5icu_HETATM_relaxed.pdb @pmut.flags

# $ROSETTA3/bin/score_jd2.mpi.linuxiccrelease -in:file:l score_pdb.txt -ignore_unrecognized_res
