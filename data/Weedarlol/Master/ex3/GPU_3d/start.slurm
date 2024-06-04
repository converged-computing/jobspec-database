#!/bin/bash
set -e

# Common flags for nvcc
NVCCFLAGS="-rdc=true"

# Directory for your library (adjust this to the actual path)
LIBDIR="/../../usr/lib/x86_64-linux-gnu/libnccl.so"

# List of source files
SOURCES="main.cu programs/jacobi.cu programs/scenarios.cu programs/cuda_functions.cu"
# Object files derived from sources
OBJECTS=$(echo "$SOURCES" | sed 's/\.cu/\.o/g')

# Gets the first element which indicates the partition type
PARTITION="$1"

shift


# Creates a scenario for the rest of the elements
for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    WIDTH=$(echo $SCENARIO | awk -F'width' '{print $2}' | awk -F'_' '{print $1}')
    HEIGHT=$(echo $SCENARIO | awk -F'height' '{print $2}' | awk -F'_' '{print $1}')
    DEPTH=$(echo $SCENARIO | awk -F'depth' '{print $2}' | awk -F'_' '{print $1}')
    NUM_GPUS=$(echo $SCENARIO | awk -F'gpu' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    COMPARE=$(echo $SCENARIO | awk -F'compare' '{print $2}' | awk -F'_' '{print $1}')
    OVERLAP=$(echo $SCENARIO | awk -F'overlap' '{print $2}' | awk -F'_' '{print $1}')
    TEST=$(echo $SCENARIO | awk -F'test' '{print $2}')

    # Define the variable values
    VAR1=$WIDTH # Width
    VAR2=$HEIGHT # Height
    VAR3=$DEPTH
    VAR4=$ITERATIONS  # Iterations
    VAR5=$NUM_GPUS    # GPUs
    VAR6=$COMPARE
    VAR7=$OVERLAP
    VAR8=$TEST

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}_${PARTITION}.sh"

    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --job-name=${SCENARIO}_${PARTITION}_GPU${NUM_GPUS}
#SBATCH -p ${PARTITION}
#SBATCH --reservation=vidaro_21
#SBATCH -N 1                                  # Antall noder, i dgx2q så er det bare 1 node bygd opp av DualProcessor AMD EPYC Milan 7763 64-core w/ 8 qty Nvidia Volta A100/80GB
#SBATCH -n 1                                  # Prosesser
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1                   # Antall ganger en vil kjøre programmet på hver node
#SBATCH --cpus-per-task=1                     # Antall CPUer en vil allokere, er 64 cores per CPU, så kan i teorien bare trenge å øke dene når -n > 64
#SBATCH --gres=gpu:${NUM_GPUS}                # Antall GPUer en allokerer per job, så totale antall GPU
#SBATCH --gpus-per-task=${NUM_GPUS}           # Antall GPUer en allokerer per task, så totale antall GPU delt på antall noder
#SBATCH --gpus=${NUM_GPUS}                    # Antall GPUer en allokerer per node, så totale antall GPU delt på antall noder
#SBATCH --output=output/${SCENARIO}_${PARTITION}.out
#SBATCH --error=error/${SCENARIO}_${PARTITION}.err

module purge
module load slurm/21.08.8
module load cuda11.8/toolkit/11.8.0


nvcc $NVCCFLAGS $SOURCES -o out_${SCENARIO}_${PARTITION} -L$LIBDIR -lnccl -O3 -lm

#nsys nvprof -o nsight/nsight_${SCENARIO}_${PARTITION} ./out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4 $VAR5 $VAR6 $VAR7 $VAR8
./out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4 $VAR5 $VAR6 $VAR7 $VAR8




exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done