#!/bin/bash
#SBATCH --job-name=colmap_%j
#SBATCH --output=colmap_%j.log
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GTX1080Ti:4
#SBATCH --cpus-per-gpu=16

set -e

ml purge
module load CUDA/9.0.176-GCC-6.4.0-2.28
module load Singularity/2.4.2-GCC-5.4.0-2.26

sub=$1

nvidia-smi

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo

# echo Running
# echo ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec
# echo     --nv
# echo     /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg
# echo         colmap automatic_reconstructor
# echo             --workspace_path $(cat params.yaml | yq -r '.colmap_'$sub'.dataset_root')
# echo             --image_path "$(cat params.yaml | yq -r '.colmap_'$sub'.dataset_root')/images/"
# echo

# Container obtained in given folder just by running:
# singularity build singularity-2.4.2-colmap_3.6_dev.3.simg docker://colmap/colmap:latest
# Dataset root head to raw images.
# ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
#     --nv \
#     /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
#         colmap automatic_reconstructor \
#             --workspace_path $(cat params.yaml | yq -r '.colmap_'$sub'.dataset_root') \
#             --image_path "$(cat params.yaml | yq -r '.colmap_'$sub'.dataset_root')/images/"

DATASET_PATH=$(cat params.yaml | yq -r '.colmap_'$sub'.dataset_root')

echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec"
echo "    --nv"
echo "    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg"
echo "        colmap feature_extractor"
echo "           --database_path $DATASET_PATH/database.db"
echo "           --image_path $DATASET_PATH/images"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
    --nv \
    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
        colmap feature_extractor \
           --database_path $DATASET_PATH/database.db \
           --image_path $DATASET_PATH/images


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec"
echo "    --nv"
echo "    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg"
echo "        colmap exhaustive_matcher"
echo "            --database_path $DATASET_PATH/database.db"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
    --nv \
    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
        colmap exhaustive_matcher \
            --database_path $DATASET_PATH/database.db


echo
echo "Running"
echo "mkdir -p $DATASET_PATH/sparse"
echo

mkdir -p $DATASET_PATH/sparse


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec"
echo "    --nv"
echo "    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg"
echo "        colmap mapper"
echo "            --database_path $DATASET_PATH/database.db"
echo "            --image_path $DATASET_PATH/images"
echo "            --output_path $DATASET_PATH/sparse"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
    --nv \
    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
        colmap mapper \
            --database_path $DATASET_PATH/database.db \
            --image_path $DATASET_PATH/images \
            --output_path $DATASET_PATH/sparse


echo
echo "Running"
echo "mkdir $DATASET_PATH/dense"

mkdir $DATASET_PATH/dense


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec"
echo "    --nv"
echo "    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg"
echo "        colmap image_undistorter"
echo "            --image_path $DATASET_PATH/images"
echo "            --input_path $DATASET_PATH/sparse/0"
echo "            --output_path $DATASET_PATH/dense"
echo "            --output_type COLMAP"
echo "            --max_image_size 2000"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
    --nv \
    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
        colmap image_undistorter \
            --image_path $DATASET_PATH/images \
            --input_path $DATASET_PATH/sparse/0 \
            --output_path $DATASET_PATH/dense \
            --output_type COLMAP \
            --max_image_size 2000


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec"
echo "    --nv"
echo "    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg"
echo "        colmap patch_match_stereo"
echo "            --workspace_path $DATASET_PATH/dense"
echo "            --workspace_format COLMAP"
echo "            --PatchMatchStereo.geom_consistency true"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
    --nv \
    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
        colmap patch_match_stereo \
            --workspace_path $DATASET_PATH/dense \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec"
echo "    --nv"
echo "    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg"
echo "        colmap stereo_fusion"
echo "            --workspace_path $DATASET_PATH/dense"
echo "            --workspace_format COLMAP"
echo "            --input_type geometric"
echo "            --output_path $DATASET_PATH/dense/fused.ply"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
    --nv \
    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
        colmap stereo_fusion \
            --workspace_path $DATASET_PATH/dense \
            --workspace_format COLMAP \
            --input_type geometric \
            --output_path $DATASET_PATH/dense/fused.ply


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec"
echo "    --nv"
echo "    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg"
echo "        colmap poisson_mesher"
echo "            --input_path $DATASET_PATH/dense/fused.ply"
echo "            --output_path $DATASET_PATH/dense/meshed-poisson.ply"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
    --nv \
    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
        colmap poisson_mesher \
            --input_path $DATASET_PATH/dense/fused.ply \
            --output_path $DATASET_PATH/dense/meshed-poisson.ply


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec"
echo "    --nv"
echo "    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg"
echo "        colmap delaunay_mesher"
echo "            --input_path $DATASET_PATH/dense"
echo "            --output_path $DATASET_PATH/dense/meshed-delaunay.ply"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity exec \
    --nv \
    /home/kremeto1/containers/singularity-2.4.2-colmap_3.6_dev.3.simg \
        colmap delaunay_mesher \
            --input_path $DATASET_PATH/dense \
            --output_path $DATASET_PATH/dense/meshed-delaunay.ply
