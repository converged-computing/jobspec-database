#PBS -N <run_name>
#PBS -l select=2:ncpus=40:ngpus=2:gpu_model=a100:mem=245gb,walltime=72:00:00
#PBS -j oe
#PBS -o <output_log_filename_or_path>

##########################################
# These variables are run dependent 
# and shoud be set according to your 
# needs. Additionally replace 
# resource requests and < > statements
# above to fit your needs.
##########################################

# Hardcode number of gpus per node.
NGPUS=2

# Set these according to your needs. 
ENV_NAME="<your_conda_env_name_here>"
LAUNCH_SCRIPT="${PBS_O_WORKDIR}/run.sh"

########################################
#
########################################

# timestamp output directory name
# and create directory.
timestamp=$(date +%D_%H_%M_%S | tr / _)
OUTPUT_DIR="${PBS_O_WORKDIR}/output/${timestamp}"
mkdir -p $OUTPUT_DIR

# Useful for distributed debugging
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1 

# Add modules (implementation dependent)
module add cuda/11.6.2-gcc/9.5.0
module add nccl/2.11.4-1-gcc/9.5.0-cu11_6-nvP-nvV-nvA
module add openmpi/4.1.3-gcc/9.5.0-cu11_6-nvP-nvV-nvA-ucx
module add anaconda3/2022.05-gcc/9.5.0

# Activate specified environment
source activate $ENV_NAME

# Get number of nodes. This will be the same as specified above.
nnodes=$(cat $PBS_NODEFILE | wc -l)
ncpus=$NCPUS

echo "Running as ${USER} with ${nnodes} nodes."

# PBSDSH to start each process
# This should be changed according to your 
# needs and the requirements of your launch 
# script.

pbsdsh -- bash "${PBS_O_WORKDIR}/run.sh" \
        $HOSTNAME \
        $ENV_NAME \
        $nnodes \
        $ngpus \
        "./run_mlm.py \
        --model_name $model_name \
        --dataset_names bookcorpus \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --n_cpu $ncpus \
        --do_train \
        --do_eval \
        --group_by_size \
        --grad_accumulation_steps 16"
 

