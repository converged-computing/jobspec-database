#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH --time=56:00:00
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=array_%A_%a.out

#Execute program located in $HOME
#$HOME/my_serial_program


cd $HOME/workspace || exit

source load.sh

exp_name=CoPrompt
trainer=CoPrompt
train_bash=scripts/base2new_train_coprompt.sh
test_bash=scripts/base2new_test_coprompt.sh

export PYTHONPATH="$PYTHONPATH:$PWD"

# Run all training in parallel, then wait for all to finish
for seed in 1; do
  for dataset in sun397; do
#  for dataset in fgvc_aircraft caltech101 oxford_pets stanford_cars food101 sun397; do
    for wstd in 0.001 0.012 0.08; do
      export WEIGHT_STD=$wstd
      for max_epoch in 8 16; do
        export MAX_EPOCH=$max_epoch
        for nctxt in 1 2 3 4; do
          export NUM_CONTEXT=$nctxt
          for momentum in 0.0 0.9; do
            export MOMENTUM=$momentum
            export exp_name=CoPrompt_m${momentum}_wstd${wstd}_nctxt${nctxt}_maxepoch${max_epoch}
            output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/train_base/${dataset}/shots_16/CoPrompt/seed${seed}"
            mkdir -p "$output_dir"
            echo "Runing the first phase job and save the output to ${output_dir}"
            bash $train_bash $dataset $seed $exp_name > "${output_dir}/output.log" 2>&1 &
          done
        done
        wait
      done
    done
  done
done



#  for dataset in fgvc_aircraft dtd ucf101 eurosat caltech101 oxford_pets stanford_cars oxford_flowers food101 sun397 imagenet; do
#    bash $train_bash $dataset $seed $exp_name &
#  done
#done
#wait

## Run all tests in parallel, then wait for all to finish
#for seed in 1 2 3; do
#  for dataset in fgvc_aircraft dtd ucf101 eurosat caltech101 oxford_pets stanford_cars oxford_flowers food101 sun397 imagenet; do
#    test_arg=8
#    [ "$dataset" = "food101" ] && test_arg=5 # Special case for food101
#    bash $test_bash $dataset $seed $exp_name $test_arg &
#  done
#done
#wait

for seed in 1; do
#  for dataset in ucf101 dtd fgvc_aircraft caltech101 oxford_pets stanford_cars food101 sun397; do
  for dataset in sun397; do
    for wstd in 0.001 0.012 0.08; do
      export WEIGHT_STD=$wstd
      for max_epoch in 8 16; do
        export MAX_EPOCH=$max_epoch
        for nctxt in 1 2 3 4; do
          export NUM_CONTEXT=$nctxt
          for momentum in 0.0 0.9; do
            export MOMENTUM=$momentum
            export exp_name=CoPrompt_m${momentum}_wstd${wstd}_nctxt${nctxt}_maxepoch${max_epoch}
            output_dir="/home/ataboadawarmer/data/fomo/output/${exp_name}/test_new/${dataset}/shots_16/CoPrompt/seed${seed}"
            mkdir -p "$output_dir"
            echo "Runing the first phase job and save the output to ${output_dir}"
            bash $test_bash $dataset $seed $exp_name $MAX_EPOCH > "${output_dir}/output.log" 2>&1 &
          done
        done
        wait
      done
    done
  done
done