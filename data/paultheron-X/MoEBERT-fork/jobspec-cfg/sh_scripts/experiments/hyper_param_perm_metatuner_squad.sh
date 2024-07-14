#!/bin/bash

echo "Getting best params from pretraining for task $1"
echo "Experiment Set is $2"
echo "Best lr was $3"
echo "Best batch size was $4"
echo "Best weight decay was $5"
echo "Best epoch was $6"
echo "Output dir is $7"


export eval_steps=1000


echo "Starting training for task $1, with the given hyperparameters"

echo "Now starting distillation for task $1"

echo "Launching Experiment Set $2"

BEGIN=$((10*$2 + -9)) 
END=$((10*$2))


for i in $(eval echo "{$BEGIN..$END}")
do
    echo "--------- Launching PERM SQUAD V2 Moebert Experiment $i"
    args=$(python sh_scripts/python_helpers/launch_job_from_grid.py -n $i --perm --ds $1)

    bash sh_scripts/experiments/base_perm_moebert_trainer_squad.sh $1 $args $eval_steps $6 True

    echo "Done with Moebert Experiment $i, deleting the intermediate checkpoints"
    rm -r $7/$1/moebert_experiment_$i/model/checkpoint-*
done
