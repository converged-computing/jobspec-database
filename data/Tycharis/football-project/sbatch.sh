#!/bin/sh

##SBATCH --partition=ksu-chem-mri.q,ksu-gen-gpu.q

# Param 1 - ${1}: --bondEn: interaction energy
# Param 2 - ${2}: --iterations: total number of accepted moves
# Param 3 - ${3}: --split: frequency of saving files
# Param 4 - ${4}: --blocks: total blocks in the simulations block
# Param 5 - ${5}: --length: total units in the protein
# Param 6 - ${6}: --Xm
# Param 7 - ${7}: --Ym
# Param 8 - ${8}: --Zm
# Param 10 - ${9}: --dimensions
# Param 11 - ${10}: --writeId

# --runId: current duplicate job number

module load icc

${PROTEIN_PROJECT_DIR}/program.o --bondEn ${1} --iterations ${2} --split ${3} --blocks ${4} --length ${5} --runId ${SLURM_ARRAY_TASK_ID:-0} \
    --Xm ${6:-200} --Ym ${7:-500} --Zm ${8:-200} --dimensions ${9:-4} --writeId ${10:-1}
