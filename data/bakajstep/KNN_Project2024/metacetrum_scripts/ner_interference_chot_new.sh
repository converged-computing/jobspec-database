#!/bin/bash
#PBS -N batch_job_knn
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=20gb:mem=40gb:scratch_ssd=20gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m ae

# -j oe ... standard error stream of the job will be merged with the standard output stream
# -m ae ...  mail is sent when the job aborts or terminates

# The source code of this file is based on the following source:
#
# Source web: GitHub
# Link to the source: https://github.com/roman-janik/diploma_thesis_program/blob/main/ner/train_ner_model.sh
# Author: Roman Janík (https://github.com/roman-janik)

trap 'clean_scratch' TERM EXIT

HOMEPATH=/storage/praha1/home/$PBS_O_LOGNAME
DATAPATH=$HOMEPATH/program/datasets/            # folder with datasets
MODELPATH=$HOMEPATH/program/models/
RESPATH=$HOMEPATH/program/results/      # store results in this folder
HOSTNAME=$(hostname -f)                 # hostname of local machine

printf "\-----------------------------------------------------------\n"
printf "JOB ID:             %s\n" "$PBS_JOBID"
printf "JOB NAME:           %s\n" "$PBS_JOBNAME"
printf "JOB SERVER NODE:    %s\n" "$HOSTNAME"
printf "START TIME:         %s\n" "$(date +%Y-%m-%d-%H-%M)"
printf "GIT BRANCH:         $branch\n"
printf "\-----------------------------------------------------------\n"

start_time=$(date +%s)

cd "$SCRATCHDIR" || exit 2

# clean the SCRATCH directory
clean_scratch

# Clone the repository
printf "Cloning the repository ...\n"
cp "$HOMEPATH"/.ssh/id_ed25519 "$HOMEPATH"/.ssh/known_hosts "$HOME"/.ssh
printf "Print content of .ssh dir\n"
ls -la "$HOME"/.ssh
mkdir program
cd program || exit 2
git clone git@github.com:bakajstep/KNN_Project2024.git
if [ $? != 0 ]; then
  printf "Cloning repository failed!\n"
  exit 1
fi
cd KNN_Project2024 || exit 2
git checkout "$branch"
cd ../..

# Prepare directory with results
printf "Prepare directory with results\n"
if [ ! -d "$HOMEPATH"/program/results/ ]; then # test if dir exists
  mkdir "$HOMEPATH"/program/ "$HOMEPATH"/program/results/
fi

# Prepare local directory with results
mkdir program/results

# Prepare directory with datasets
printf "Prepare directory with datasets\n"
if [ ! -d "$DATAPATH" ]; then # test if dir exists
  mkdir "$DATAPATH"  
fi

# Copy converted datasets if they are created
# It is not needed to create a "datasets" directory locally because it is 
# coppied from the storage including the folder.
cp -R "$DATAPATH" program

# Copy trained models.
cp -R "$MODELPATH" program

# Prepare environment
printf "Prepare environment\n"
source /cvmfs/software.metacentrum.cz/modulefiles/5.1.0/loadmodules
module load python
python -m venv env
source ./env/bin/activate
mkdir tmp
cd program/KNN_Project2024 || exit 2
pip install --upgrade pip
TMPDIR=../../tmp pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt

# Prepare list of configurations
if [ "$config" == "all" ]; then
  config_list="configs/*"
else
  if [ "${config:0:1}" == '[' ]; then # list of configs
    config=${config#*[}
    config=${config%]*}
  fi

  config_list=$(for cfg in $config
  do
    echo "configs/$cfg.yaml"
  done)
fi

# Create all experiment results files
curr_date="$(date +%Y-%m-%d-%H-%M)"
all_exp_results="$RESPATH"all_experiment_results_"$curr_date".txt
touch "$all_exp_results"
all_exp_results_csv="$RESPATH"all_experiment_results_"$curr_date".csv


# Run training and save results for configs in list of configurations
printf "\nPreparation took %s seconds, starting training...\n" $(($(date +%s) - start_time))

config_idx=0
for config_file in $config_list
do
  config_name=${config_file#*/}
  config_name=${config_name%.*}
  printf -- '-%.0s' {1..180}; printf "\n%s. experiment\n" $config_idx
  printf "\nConfig: %s\n" "$config_name"

  # Start training
  printf "Starting evaluation \n"

  # Run the training script.
  #python cnec2_ner_trainer.py --config "$config_file" # --results_csv "$all_exp_results_csv"
  python chain_of_trust_eval.py --config "$config_file" # --results_csv "$all_exp_results_csv"
  printf "Evaluation exit code: %s\n" "$?"

  # Save results
  printf "\nSave results\n"
  new_model_dir=$RESPATH/$(date +%Y-%m-%d-%H-%M)-$config_name-${stime}h
  mkdir "$new_model_dir"
  grep -vx '^Loading.*arrow' ../results/experiment_results.txt > ../results/experiment_results_f.txt # Remove logs from dataset load
  printf -- '-%.0s' {1..180} >> "$all_exp_results"; printf "\n%s. experiment\n" $config_idx >> "$all_exp_results"
  ((config_idx++))
  cat ../results/experiment_results_f.txt >> "$all_exp_results"
  mv ../results/* "$new_model_dir"
  cp "$config_file" "$new_model_dir"
done

# Move to the root folder before copy.
cd ../..

# Copy the datasets into the directory in the storage.
cp program/datasets/*.zip "${DATAPATH}"

# clean the SCRATCH directory
clean_scratch
