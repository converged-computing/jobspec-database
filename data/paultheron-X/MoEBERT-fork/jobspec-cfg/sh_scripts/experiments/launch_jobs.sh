#!/bin/bash

echo "Launching jobs for $1"
echo "Output dir is $2"
echo "Experiment Set is $3"

if [ -z $3 ]
then
    echo "No experiment set passed"
    export experiment_set=1
else
    echo "Experiment set passed is $3"
    export experiment_set=$3
fi

### To update by passing the output dirs

if [ $1 = 'cola' ]
then
    bash sh_scripts/experiments/hyper_param_distil_metatuner.sh cola $experiment_set 3e-5 32 0.01 7 $2
elif [ $1 = 'rte' ]
then
    bash sh_scripts/experiments/hyper_param_distil_metatuner.sh rte $experiment_set 3e-5 16 0 2 $2
elif [ $1 = 'mrpc' ]
then
    bash sh_scripts/experiments/hyper_param_distil_metatuner.sh mrpc $experiment_set 4e-5 8 0 2 $2
elif [ $1 = 'sst2' ]
then
    bash sh_scripts/experiments/hyper_param_distil_metatuner.sh sst2 $experiment_set  3e-5 32 0.01 3 $2
elif [ $1 = 'qqp' ]
then
    bash sh_scripts/experiments/hyper_param_distil_metatuner.sh qqp $experiment_set 2e-5 8 0.01 2 $2
elif [ $1 = 'mnli' ]
then
    bash sh_scripts/experiments/hyper_param_distil_metatuner.sh mnli $experiment_set 3e-5 16 0.01 2 $2
elif [ $1 = 'qnli' ]
then
    bash sh_scripts/experiments/hyper_param_distil_metatuner.sh qnli $experiment_set 3e-5 16 0 9 $2
elif [ $1 = 'mrpcbig' ]
then
    bash sh_scripts/experiments/hyper_param_distil_metatuner.sh mrpc $experiment_set 4e-5 8 0 2 $2
elif [ $1 = 'squad_v2' ]
then 
    bash sh_scripts/experiments/hyper_param_distil_metatuner_squad.sh squad_v2 $experiment_set 4e-5 8 0 2 $2  # we dont care of the params 4e-5 8 0 2
fi

