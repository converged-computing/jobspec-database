#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=1:gpu_cap=cuda80:gpu_mem=10gb:mem=20gb:scratch_ssd=10gb:cluster=galdor
#PBS -j oe

# Author: Roman Janík
# Metacentrum job shell script
# Training page segmentation model
##!!!!! IF YOU CHANGE WALLTIME DONT FORGET TO CHANGE TIMEOUT FOR TRAINING SCRIPT. !!!!!####
##!!!!! TRAINING SCRIPT TIMEOUT SHOULD BE AT LEAST 30 min. SHORTER THAN WALLTIME. !!!!!####

trap 'clean_scratch' TERM EXIT

HOMEPATH=/storage/praha1/home/$PBS_O_LOGNAME
DATAPATH=$HOMEPATH/datasets/            # folder with datasets
RESPATH=$HOMEPATH/program/results/      # store results in this folder
HOSTNAME=$(hostname -f)                 # hostname of local machine

printf "\-----------------------------------------------------------\n"
printf "JOB ID:             %s\n" "$PBS_JOBID"
printf "JOB NAME:           %s\n" "$PBS_JOBNAME"
printf "JOB SERVER NODE:    %s\n" "$HOSTNAME"
printf "START TIME:         %s\n" "$(date +%Y-%m-%d-%H-%M)"
#printf "GIT BRANCH:         $branch\n"
printf "\-----------------------------------------------------------\n"

cd "$SCRATCHDIR" || exit 2

# clean the SCRATCH directory
clean_scratch

# Clone the diploma_thesis_program repository
printf "Clone the diploma_thesis_program repository\n"
cp "$HOMEPATH"/.ssh/id_ed25519 "$HOMEPATH"/.ssh/known_hosts "$HOME"/.ssh
printf "Print content of .ssh dir\n"
ls -la "$HOME"/.ssh
mkdir program
cd program || exit 2
git clone git@github.com:roman-janik/diploma_thesis_program.git
if [ $? != 0 ]; then
  printf "Cloning diploma_thesis_program repository failed!\n"
  exit 1
fi
cd diploma_thesis_program || exit 2
git checkout "$branch"
cd ../..

# Download dataset
printf "Download dataset\n"
mkdir datasets datasets/page_segmentation_dataset
cp "$DATAPATH"/page_segmentation_dataset/page_segmentation_dataset.zip  datasets
unzip -d datasets/page_segmentation_dataset datasets/page_segmentation_dataset.zip

# Download model
printf "Download model\n"
mkdir program/resources/
cp -r "$HOMEPATH"/program/resources/nvidia_mit-b0 program/resources/

# Prepare directory with results
printf "Prepare directory with results\n"
if [ ! -d "$HOMEPATH"/program/results/ ]; then # test if dir exists
  mkdir "$HOMEPATH"/program/ "$HOMEPATH"/program/results/
fi

# Prepare local directory with results
mkdir program/results

# Prepare environment
printf "Prepare environment\n"
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load python
python -m venv env
source ./env/bin/activate
mkdir tmp
cd program/diploma_thesis_program || exit 2
pip install --upgrade pip
TMPDIR=../../tmp pip install torch==2.0.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt


# Start training
printf "Start training\n"
cd page_crop || exit 2
python train_page_segmentation.py
printf "Training exit code: %s\n" "$?"


# Save results
printf "\nSave results\n"
new_model_dir=$RESPATH/"page_segmentation_model"-$(date +%Y-%m-%d-%H-%M)-${stime}h
mkdir "$new_model_dir"
#grep -vx '^Loading.*arrow' ../results/experiment_results.txt > ../results/experiment_results_f.txt # Remove logs from dataset load
mv ../../results/* "$new_model_dir"

# clean the SCRATCH directory
clean_scratch
