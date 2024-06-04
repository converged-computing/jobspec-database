#!/bin/bash

#SBATCH -p short # specify which partition. short gets priority and is up to 12 hours, medium up to 5 days
#SBATCH -t 0-1 # time in DD-HH
#SBATCH -c 1 #number of cores-
#SBATCH --mem=2G # memory. start with smaller amount and go up if failed
#SBATCH -o slurm.%x.%A.%a.out # standard out will be printed to this file. %A is the job ID number, %a is the task ID, and %x is the job name
#SBATCH -e slurm.%x.%A.%a.err
#SBATCH --mail-user=jodymou@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --array=1-5 # number of structures to generate

module load gcc/4.8.5
module load rosetta 

PDB="ABCD.pdb"
LIG_PARAMS="LIG.params"
XML_SCRIPT="flexbb.xml"
RESFILE="resfile.txt"

OUTDIR="./"

NSTRUCT=1

/n/app/rosetta/3.13/source/bin/rosetta_scripts.default.linuxgccrelease \
    -database /n/app/rosetta/3.13/database/ \
    -s $PDB \
    -nstruct $NSTRUCT \
    -run:constant_seed \
    -run:jran ${SLURM_ARRAY_TASK_ID} \
    -extra_res_fa $LIG_PARAMS \
    -parser:protocol $XML_SCRIPT \
    -packing:resfile $RESFILE \
    -packing:multi_cool_annealer 10 \
    -packing:linmem_ig 10 \
    -out:path:all $OUTDIR \
    -out:pdb \
    -overwrite \
    -ignore_waters false \
    -beta \
    -water_hybrid_sf \

# if using water vdMs:
# rename all waters to TP5
# use the last 3 options above

