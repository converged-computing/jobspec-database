#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=1:gpu_cap=cuda80:gpu_mem=20gb:mem=20gb:scratch_ssd=20gb:cluster=galdor
#PBS -j oe

# Author: Roman Janík
# Metacentrum job shell script
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

start_time=$(date +%s)

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
git clone git@github.com:xjanik20/diploma_thesis_program.git
if [ $? != 0 ]; then
  printf "Cloning diploma_thesis_program repository failed!\n"
  exit 1
fi
cd diploma_thesis_program || exit 2
git checkout "$branch"
cd ../..

# Download dataset
printf "Download dataset\n"
mkdir datasets datasets/cnec2.0_extended datasets/chnec1.0 datasets/sumeczech-1.0-ner
cp "$DATAPATH"/cnec2.0_extended/cnec2.0_extended.zip "$DATAPATH"/chnec1.0/chnec1.0.zip "$DATAPATH"/poner1.0/poner1.0.zip datasets
#cp "$DATAPATH"/sumeczech-1.0-ner/sumeczech-1.0-ner.zip datasets
unzip -d datasets/cnec2.0_extended datasets/cnec2.0_extended.zip
unzip -d datasets/chnec1.0 datasets/chnec1.0.zip
#unzip -d datasets/sumeczech-1.0-ner datasets/sumeczech-1.0-ner.zip
unzip -d datasets/poner1.0 datasets/poner1.0.zip

# Download model
printf "Download model\n"
mkdir program/resources/
cp -r "$HOMEPATH"/program/resources/robeczech-base-pytorch program/resources/
cp -r "$HOMEPATH"/program/resources/Czert-B-base-cased program/resources/
cp -r "$HOMEPATH"/program/resources/Slavic-BERT-cased program/resources/
cp -r "$HOMEPATH"/program/resources/ml_models program/resources/

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

# Prepare list of configurations
if [ "$config" == "all" ]; then
  config_list="exp_configs_ner/*"
else
  if [ "${config:0:1}" == '[' ]; then # list of configs
    config=${config#*[}
    config=${config%]*}
  fi

  config_list=$(for cfg in $config
  do
    echo "exp_configs_ner/$cfg.yaml"
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
  printf "Start training\n"
  if [ -z "$modelpath" ]; then
      python train_ner_model.py --config "$config_file" --results_csv "$all_exp_results_csv"
      printf "Training exit code: %s\n" "$?"
  else
  #    mkdir ./logs
  #    mkdir ./logs/latest_model
  #    cp -r ${modelpath}/* ./logs/latest_model
      python script_na_trenovani
      printf "Training exit code: %s\n" "$?"
  fi


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

# clean the SCRATCH directory
clean_scratch
