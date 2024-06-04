#!/bin/bash
#SBATCH --account=def-ycoady
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=t4:1
#SBATCH --time=2:59:0
#SBATCH --mail-user=ribas.w@northeastern.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --output=nerfstudio_train_model_%j.out

programname=$0
function usage {
    echo ""
    echo "Train a nerfacto model from the source of COLMAP images."
    echo ""
    echo "usage: $programname --images string --output string "
    echo ""
    echo "  --data string           the directory that contains the COLMAP images to be trained"
    echo "                          (example: /scratch/wribas/nerfstudio/data/)"
    echo "  --output string         the directory to store the trained model and all nerfstudio output files"
    echo "                          (example: /scratch/wribas/nerfstudio/outputs)"
    echo "  --mode string           OPTIONAL - default is nerfacto - the training mode to utilize. Any value supported by nerfstudio is accepted"
    echo "                          (example: nerfacto, instant-ngp, mipnerf, tensorf, etc)"
    echo "  --container string      OPTIONAL - the path to the nerfstudio singularity container. By default it loads Weder's container on Compute Canada"
    echo "                          (example: /scratch/wribas/nerfstudio/nerfstudio-cuda-11-3.sif)"
    echo "  --help                  displays this help"
    echo ""
}

function die {
    printf "Script failed: %s\n\n" "$1"
    exit 1
}

while [ $# -gt 0 ]
do
    case $1 in
        --data)
            data_path=$2
            shift 2
            ;;
        --output)
            output_path=$2
            shift 2
            ;;
        --mode)
            training_mode=$2
            shift 2
            ;;
        --container)
            nerfstudio_container=$2
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            usage
            die "Error: Unrecognized option $1"
            ;;
    esac
done

if [ -z "$data_path" ]
then
    usage
    die "Missing required parameters --data"
fi

if [ -z "$output_path" ]
then
    usage
    die "Missing required parameters --output"
fi

if [ -z "$training_mode" ]
then
    training_mode="nerfacto"
fi

if [ -z "$nerfstudio_container" ]
then
    nerfstudio_container="/scratch/wribas/nerfstudio/nerfstudio-cuda-11-3.sif"

    if [ ! -e "$nerfstudio_container" ]
    then
        die "Could not find the nerfstudio container within the given path"
    fi
fi

module load singularity
module load cuda

singularity exec --bind $data_path:/opt/nerfstudio-nu-papers/data --bind $output_path:/opt/nerfstudio-nu-papers/outputs ${nerfstudio_container} bash -c "cd /opt/nerfstudio-nu-papers && . venv/bin/activate && ns-train $training_mode --data data/ --output-dir outputs/"
