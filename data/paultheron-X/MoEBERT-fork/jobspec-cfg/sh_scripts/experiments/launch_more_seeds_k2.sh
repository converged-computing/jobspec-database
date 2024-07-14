#!/bin/bash

echo "Launching larger jobs for $1"
echo "Output dir is $2"

if [ $1 = "qnli" ]
then
    bash sh_scripts/experiments/base_finetuning.sh qnli 1 2e-5 32 0 7 $2
    export eval_steps=1000
    declare -a StringArray=("19" "32" "28" "65" "88")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer_k2.sh $1 $args $eval_steps 2 True $l $2
            echo "Done with training for that experiment, deleting the intermediate checkpoints"
            rm -rf $2/$1/moebert_k2_experiment_10$j$l/model/checkpoint-*
            echo "Done with deleting the intermediate checkpoints"
        done
    done   
elif [ $1 = "qqp" ]
then
    bash sh_scripts/experiments/base_finetuning.sh qqp 1 2e-5 16 0.1 2 $2
    export eval_steps=2000
    declare -a StringArray=("10" "23" "35" "49" "88")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer_k2.sh $1 $args $eval_steps 2 True $l $2
            echo "Done with training for that experiment, deleting the intermediate checkpoints"
            rm -rf $2/$1/moebert_k2_experiment_10$j$l/model/checkpoint-*
            echo "Done with deleting the intermediate checkpoints"
        done
    done   
elif [ $1 = "sst2" ]
then
    bash sh_scripts/experiments/base_finetuning.sh sst2 1 2e-5 16 0 3 $2
    export eval_steps=1000
    declare -a StringArray=("14" "37" "92" "50" "63")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer_k2.sh $1 $args $eval_steps 2 True $l $2
            echo "Done with training for that experiment, deleting the intermediate checkpoints"
            rm -rf $2/$1/moebert_k2_experiment_10$j$l/model/checkpoint-*
            echo "Done with deleting the intermediate checkpoints"
        done
    done   
elif [ $1 = "mnli" ]
then
    bash sh_scripts/experiments/base_finetuning.sh mnli 1 3e-5 16 0.01 2 $2
    export eval_steps=2000
    declare -a StringArray=("2" "4" "44" "64" "100")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer_k2.sh $1 $args $eval_steps 2 True $l $2
            echo "Done with training for that experiment, deleting the intermediate checkpoints"
            rm -rf $2/$1/moebert_k2_experiment_10$j$l/model/checkpoint-*
            echo "Done with deleting the intermediate checkpoints"
        done
    done   
elif [ $1 = "rte" ]
then
    bash sh_scripts/experiments/base_finetuning.sh rte 1 2e-5 8 0.01 6 $2

    # bash sh_scripts/experiments/base_moebert_trainer_k2.sh $1 $name $bs $weight_decay $lr $entropy $gamma $distill $eval_steps $best_epoch_first_training $seed
    export eval_steps=1000
    declare -a StringArray=("3" "20" "35" "71" "94")
    for ((i=1; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer_k2.sh $1 $args $eval_steps 2 True $l $2
            echo "Done with training for that experiment, deleting the intermediate checkpoints"
            rm -rf $2/$1/moebert_k2_experiment_10$j$l/model/checkpoint-*
            echo "Done with deleting the intermediate checkpoints"
        done
    done    
elif [ $1 = "cola" ]
then
    bash sh_scripts/experiments/base_finetuning.sh cola 1 3e-5 32 0.01 7 $2

    export eval_steps=1000
    declare -a StringArray=("3" "39" "43" "36" "6")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer_k2.sh $1 $args $eval_steps 2 True $l $2
            echo "Done with training for that experiment, deleting the intermediate checkpoints"
            rm -rf $2/$1/moebert_k2_experiment_10$j$l/model/checkpoint-*
            echo "Done with deleting the intermediate checkpoints"
        done
    done    
elif [ $1 = "mrpc" ]
then
    bash sh_scripts/experiments/base_finetuning.sh mrpc 1 4e-5 8 0.1 6 $2

    export eval_steps=1000
    declare -a StringArray=("7" "11" "35" "56" "102")
    for ((i=0; i<${#StringArray[@]}; i++)); do
    # Access the current element of the array using the index variable
        element=${StringArray[$i]}
        j=$((i+1))
        for l in {1..5}
        do
            args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $element --exp 10$j$l)
            bash sh_scripts/experiments/base_moebert_trainer_k2.sh $1 $args $eval_steps 2 True $l $2
            echo "Done with training for that experiment, deleting the intermediate checkpoints"
            rm -rf $2/$1/moebert_k2_experiment_10$j$l/model/checkpoint-*
            echo "Done with deleting the intermediate checkpoints"
        done
    done    
fi