#!/bin/bash

echo "Launching larger jobs for $1"
echo "Output dir is $2"

if [ $1 = "qnli" ]
then
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh qnli 3 2e-5 32 0 7 $2
elif [ $1 = "qqp" ]
then
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh qqp 1 2e-5 16 0.1 2 $2
elif [ $1 = "sst2" ]
then
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh sst2 5 2e-5 16 0 3 $2
elif [ $1 = "mnli" ]
then
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh mnli 1 3e-5 16 0.01 2 $2
elif [ $1 = "rte" ]
then
    #bash sh_scripts/experiments/hyper_param_perm_metatuner.sh rte 1 2e-5 8 0.01 6 $2
    #bash sh_scripts/experiments/hyper_param_perm_metatuner.sh rte 2 2e-5 8 0.01 6 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh rte 3 2e-5 8 0.01 6 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh rte 4 2e-5 8 0.01 6 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh rte 5 2e-5 8 0.01 6 $2
elif [ $1 = "cola" ]
then
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh cola 1 3e-5 32 0.01 7 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh cola 2 3e-5 32 0.01 7 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh cola 3 3e-5 32 0.01 7 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh cola 4 3e-5 32 0.01 7 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh cola 5 3e-5 32 0.01 7 $2
elif [ $1 = "mrpc" ]
then
    #bash sh_scripts/experiments/hyper_param_perm_metatuner.sh mrpc 1 4e-5 8 0.1 6 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh mrpc 2 4e-5 8 0.1 6 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh mrpc 3 4e-5 8 0.1 6 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh mrpc 4 4e-5 8 0.1 6 $2
    bash sh_scripts/experiments/hyper_param_perm_metatuner.sh mrpc 5 4e-5 8 0.1 6 $2
elif [ $1 = "squad_v2" ]
then
    bash sh_scripts/experiments/hyper_param_perm_metatuner_squad.sh $1 $3 4e-5 8 0.1 6 $2
fi