#!/bin/bash

#SBATCH --job-name emb_train_multi_gpu
#SBATCH --error emb_train_error_%j.txt
#SBATCH --output emb_train_output_%j.txt
#SBATCH --mail-user pietro.girotto@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 05-00:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 50G
#SBATCH --gres=gpu:rtx:2


# Setting up vars
work_dir="/home/girottopie/Code/mr_white_game"
dataset_name="it_20M_lines_polished"
archive_path="/ext/${dataset_name}.tar.gz"
dataset_path="/ext/${dataset_name}.txt"
gdrive_id="1a6u6tUXCfswcV5AUOc5LC8hDE_ih4Yil"

# Cleaning prev downloads
rm $archive_path
rm $dataset_path

# Downloading the dataset
source /home/girottopie/.bashrc
gdrivedownload $gdrive_id $archive_path

# Extracting the archive
cd /ext
tar -xzvf $dataset_name.tar.gz

# Checking everything went all right
echo "Extracted dataset, cheking first line:"
head -1 $dataset_path
echo ""


# Setting up vars for training
context_size=2
embedding_dim=300
epochs=10
batch_size=32
save_every=2

# Starting training
cd $work_dir
srun singularity exec --bind /ext:/ext --nv /nfsd/opt/sif-images/tensorflow_latest-gpu.sif python3 model_training/main_multi_gpu.py -d $dataset_path -c $context_size -e $embedding_dim -ep $epochs -b $batch_size -s $save_every
