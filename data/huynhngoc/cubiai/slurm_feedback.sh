#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=CubiAI_feedback      # sensible name for the job
#SBATCH --mem=32G                # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=sunniva.elisabeth.daae.steiro@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/feedback-%A.out
#SBATCH --error=outputs/feedback-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 3 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi

if [ ! -d "$TMPDIR/$USER/CubiAI" ]
    then
    echo "Didn't find dataset folder. Copying files..."
    mkdir --parents $TMPDIR/$USER/CubiAI
    fi

for f in $(ls $PROJECTS/ngoc/CubiAI/datasets/*)
    do
    FILENAME=`echo $f | awk -F/ '{print $NF}'`
    echo $FILENAME
    if [ ! -f "$TMPDIR/$USER/CubiAI/$FILENAME" ]
        then
        echo "copying $f"
        cp -r $PROJECTS/ngoc/CubiAI/datasets/$FILENAME $TMPDIR/$USER/CubiAI/
        fi
    done


echo "Finished setting up files."

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
# export ITER_PER_EPOCH=100
export NUM_CPUS=4
export RAY_ROOT=$TMPDIR/$USER/ray
singularity exec --nv deoxys.sif python feedback_model.py $1 $PROJECTS/ngoc/CubiAI/perf/pretrain/$2 --temp_folder $SCRATCH_PROJECTS/ceheads/CubiAI/pretrain/$2 --epochs $3 ${@:4}
