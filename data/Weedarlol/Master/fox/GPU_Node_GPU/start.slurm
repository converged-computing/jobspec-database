#!/bin/bash
set -e

# List of source files
SOURCES="main.cu programs/jacobi.cu programs/cuda_functions.cu"

# Gets the first element which indicates the partition type
PARTITION="$1"

shift


# Creates a scenario for the rest of the elements
for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    WIDTH=$(echo $SCENARIO | awk -F'width' '{print $2}' | awk -F'_' '{print $1}')
    HEIGHT=$(echo $SCENARIO | awk -F'height' '{print $2}' | awk -F'_' '{print $1}')
    DEPTH=$(echo $SCENARIO | awk -F'depth' '{print $2}' | awk -F'_' '{print $1}')
    NODES=$(echo $SCENARIO | awk -F'nodes' '{print $2}' | awk -F'_' '{print $1}')
    NUM_GPUS=$(echo $SCENARIO | awk -F'gpu' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    COMPARE=$(echo $SCENARIO | awk -F'compare' '{print $2}' | awk -F'_' '{print $1}')
    OVERLAP=$(echo $SCENARIO | awk -F'overlap' '{print $2}' | awk -F'_' '{print $1}')
    TEST=$(echo $SCENARIO | awk -F'test' '{print $2}')

    TOTAL_GPUS=$((NUM_GPUS * NODES))

    # Define the variable values
    VAR1=$WIDTH # Width
    VAR2=$HEIGHT # Height
    VAR3=$DEPTH
    VAR4=$ITERATIONS  # Iterations
    VAR5=$NUM_GPUS    # GPUs
    VAR6=2
    VAR7=$COMPARE
    VAR8=$OVERLAP
    VAR9=$TEST

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}_${PARTITION}.sh"

    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --job-name=${SCENARIO}_${PARTITION}
#SBATCH --account=ec12
#SBATCH --mem-per-gpu=4G
#SBATCH -p ${PARTITION}
#SBATCH -N ${NODES}                                   # Antall Noder
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1                   # Antall ganger en vil kjøre programmet på hver node
#SBATCH --cpus-per-task=1                     # Antall CPUer en vil allokere, er 64 cores per CPU, så kan i teorien bare trenge å øke dene når -n > 64
#SBATCH --gres=gpu:2                            # Antall GPUer en allokerer per job, så totale antall GPU. NUM_GPUS*NODES
#SBATCH --gpus-per-task=${NUM_GPUS}                           # Antall GPUer en allokerer per task, så totale antall GPU delt på antall noder. NUM_GPUS
#SBATCH --gpus=1                                    # Antall GPUer en allokerer per node. NUM_GPUS
#SBATCH --output=output/${SCENARIO}_${PARTITION}.out
#SBATCH --error=error/${SCENARIO}_${PARTITION}.err

module purge
module load CUDA/11.7.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0



nvcc $SOURCES -o out_${SCENARIO}_${PARTITION} I$CPATH -L$LD_LIBRARY_PATH -lmpi -lnccl -O3

mpirun -n 2 out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4 $VAR5 $VAR6 $VAR7 $VAR8 $VAR9




exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done