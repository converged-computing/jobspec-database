#!/bin/bash

echo "Running script for task $1"
echo "-----------"
echo "Running script for mode $2"
echo "Given output dir is $3"

# bash sh_scripts/experiments/base_trainer.sh $1 experiment_num bs weight_decay lr eval_steps

export eval_steps=500

if [ $1 = "squad" ] || [ $1 = "squad_v2" ]
then
    if [ $2 = 1 ]
    then
        echo "Training mode 1"
        #bash sh_scripts/experiments/base_trainer_squad.sh $1 1 8 0.0 1e-5 $eval_steps 
        echo "-----------"
        echo "Finished experiment 1"

        #bash sh_scripts/experiments/base_trainer_squad.sh $1 2 8 0.01 2e-5 $eval_steps 
        echo "-----------"
        echo "Finished experiment 2"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 3 8 0.1 3e-5 $eval_steps 
        echo "-----------"
        echo "Finished experiment 3"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 4 8 0.0 4e-5 $eval_steps 
        echo "-----------"
        echo "Finished experiment 4"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 5 8 0.01 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 5"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 6 16 0.1 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 6"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 7 16 0.0 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 7"
    elif [ $2 = 2 ]
    then
        echo "Training mode 2"
        #bash sh_scripts/experiments/base_trainer_squad.sh $1 8 16 0.01 4e-5 $eval_steps  
        echo "-----------"
        echo "Finished experiment 8"

        #bash sh_scripts/experiments/base_trainer_squad.sh $1 9 16 0.1 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 9"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 10 16 0.0 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 10"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 11 32 0.01 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 11"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 12 32 0.1 4e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 12"

    elif [ $2 = 3 ]
    then
        echo "Training mode 3"

       # bash sh_scripts/experiments/base_trainer_squad.sh $1 15 32 0.1 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 15"

        #bash sh_scripts/experiments/base_trainer_squad.sh $1 16 64 0.0 4e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 16"

        #bash sh_scripts/experiments/base_trainer_squad.sh $1 17 64 0.01 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 17"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 18 64 0.1 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 18"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 19 64 0.0 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 19"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 20 64 0.1 4e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 20"


    elif [ $2 = 4 ]
    then
        echo "Training mode 4"
        #bash sh_scripts/experiments/base_trainer_squad.sh $1 21 8 0.1 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 21"

        #bash sh_scripts/experiments/base_trainer_squad.sh $1 22 8 0.0 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 22"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 23 8 0.01 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 23"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 24 8 0.1 4e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 24"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 25 8 0.0 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 25"

    elif [ $2 = 5 ]
    then
        echo "Training mode 5"
        #bash sh_scripts/experiments/base_trainer_squad.sh $1 28 16 0.0 4e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 28"

       # bash sh_scripts/experiments/base_trainer_squad.sh $1 29 16 0.01 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 29"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 30 16 0.1 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 30"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 31 32 0.0 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 31"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 32 32 0.01 4e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 32"
    elif [ $2 = 6 ]
    then
        echo "Training mode 6"
        #bash sh_scripts/experiments/base_trainer_squad.sh $1 35 32 0.01 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 35"

        #bash sh_scripts/experiments/base_trainer_squad.sh $1 36 64 0.1 4e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 36"

        #bash sh_scripts/experiments/base_trainer_squad.sh $1 37 64 0.0 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 37"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 38 64 0.01 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 38"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 39 64 0.01 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 39"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 40 64 0.0 4e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 40"
    elif [ $2 = 7 ]
    then
        bash sh_scripts/experiments/base_trainer_squad.sh $1 13 32 0.0 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 13"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 14 32 0.01 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 14"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 6 16 0.1 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 6"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 7 16 0.0 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 7"
    elif [ $2 = 8 ]
    then
        bash sh_scripts/experiments/base_trainer_squad.sh $1 33 32 0.1 1e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 33"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 34 32 0.0 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 34"
    else
        bash sh_scripts/experiments/base_trainer_squad.sh $1 26 16 0.01 2e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 26"

        bash sh_scripts/experiments/base_trainer_squad.sh $1 27 16 0.1 3e-5 $eval_steps
        echo "-----------"
        echo "Finished experiment 27"
        echo "Invalid mode"
    fi
else
    if [ $1 = "mnli" ] 
    then
        export eval_steps=2000
    elif [ $1 = "mrpc" ]
    then
        export eval_steps=200
    elif [ $1 = "qnli" ]
    then
        export eval_steps=1000
    elif [ $1 = "qqp" ]
    then
        export eval_steps=2000
    elif [ $1 = "rte" ]
    then
        export eval_steps=200
    elif [ $1 = "sst2" ]
    then
        export eval_steps=1000
    elif [ $1 = "cola" ]
    then
        export eval_steps=200
    fi

    echo "eval steps is $eval_steps"

    if [ $2 = 1 ]
    then
        echo "Training mode 1"
        bash sh_scripts/experiments/base_trainer.sh $1 1 8 0.0 1e-5 $eval_steps $3 
        echo "-----------"
        echo "Finished experiment 1"

        bash sh_scripts/experiments/base_trainer.sh $1 2 8 0.01 2e-5 $eval_steps $3 
        echo "-----------"
        echo "Finished experiment 2"

        bash sh_scripts/experiments/base_trainer.sh $1 3 8 0.1 3e-5 $eval_steps $3 
        echo "-----------"
        echo "Finished experiment 3"

        bash sh_scripts/experiments/base_trainer.sh $1 4 8 0.0 4e-5 $eval_steps $3 
        echo "-----------"
        echo "Finished experiment 4"

        bash sh_scripts/experiments/base_trainer.sh $1 5 8 0.01 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 5"

        bash sh_scripts/experiments/base_trainer.sh $1 6 16 0.1 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 6"

        bash sh_scripts/experiments/base_trainer.sh $1 7 16 0.0 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 7"

        if [ $1="cola" ] || [ $1="mrpc" ] || [ $1="rte" ]
        then
            echo "Launching the same script in a different config"

            bash sh_scripts/experiments/hyperparameter_metatuner.sh $1 4 $3
        fi

    elif [ $2 = 2 ]
    then
        echo "Training mode 2"
        bash sh_scripts/experiments/base_trainer.sh $1 8 16 0.01 4e-5 $eval_steps $3  
        echo "-----------"
        echo "Finished experiment 8"

        bash sh_scripts/experiments/base_trainer.sh $1 9 16 0.1 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 9"

        bash sh_scripts/experiments/base_trainer.sh $1 10 16 0.0 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 10"

        bash sh_scripts/experiments/base_trainer.sh $1 11 32 0.01 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 11"

        bash sh_scripts/experiments/base_trainer.sh $1 12 32 0.1 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 12"

        bash sh_scripts/experiments/base_trainer.sh $1 13 32 0.0 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 13"

        bash sh_scripts/experiments/base_trainer.sh $1 14 32 0.01 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 14"

        if [ $1="cola" ] || [ $1="mrpc" ] || [ $1="rte" ]
        then
            echo "Launching the same script in a different config"

            bash sh_scripts/experiments/hyperparameter_metatuner.sh $1 5 $3
        fi

    elif [ $2 = 3 ]
    then
        echo "Training mode 3"

        bash sh_scripts/experiments/base_trainer.sh $1 15 32 0.1 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 15"

        bash sh_scripts/experiments/base_trainer.sh $1 16 64 0.0 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 16"

        bash sh_scripts/experiments/base_trainer.sh $1 17 64 0.01 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 17"

        bash sh_scripts/experiments/base_trainer.sh $1 18 64 0.1 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 18"

        bash sh_scripts/experiments/base_trainer.sh $1 19 64 0.0 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 19"

        bash sh_scripts/experiments/base_trainer.sh $1 20 64 0.1 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 20"

        if [ $1="cola" ] || [ $1="mrpc" ] || [ $1="rte" ]
        then
            echo "Launching the same script in a different config"

            bash sh_scripts/experiments/hyperparameter_metatuner.sh $1 6 $3
        fi

    elif [ $2 = 4 ]
    then
        echo "Training mode 4"
        bash sh_scripts/experiments/base_trainer.sh $1 21 8 0.1 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 21"

        bash sh_scripts/experiments/base_trainer.sh $1 22 8 0.0 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 22"

        bash sh_scripts/experiments/base_trainer.sh $1 23 8 0.01 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 23"

        bash sh_scripts/experiments/base_trainer.sh $1 24 8 0.1 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 24"

        bash sh_scripts/experiments/base_trainer.sh $1 25 8 0.0 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 25"

        bash sh_scripts/experiments/base_trainer.sh $1 26 16 0.01 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 26"

        bash sh_scripts/experiments/base_trainer.sh $1 27 16 0.1 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 27"

    elif [ $2 = 5 ]
    then
        echo "Training mode 5"
        bash sh_scripts/experiments/base_trainer.sh $1 28 16 0.0 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 28"

        bash sh_scripts/experiments/base_trainer.sh $1 29 16 0.01 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 29"

        bash sh_scripts/experiments/base_trainer.sh $1 30 16 0.1 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 30"

        bash sh_scripts/experiments/base_trainer.sh $1 31 32 0.0 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 31"

        bash sh_scripts/experiments/base_trainer.sh $1 32 32 0.01 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 32"

        bash sh_scripts/experiments/base_trainer.sh $1 33 32 0.1 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 33"

        bash sh_scripts/experiments/base_trainer.sh $1 34 32 0.0 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 34"

    elif [ $2 = 6 ]
    then
        echo "Training mode 6"
        bash sh_scripts/experiments/base_trainer.sh $1 35 32 0.01 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 35"

        bash sh_scripts/experiments/base_trainer.sh $1 36 64 0.1 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 36"

        bash sh_scripts/experiments/base_trainer.sh $1 37 64 0.0 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 37"

        bash sh_scripts/experiments/base_trainer.sh $1 38 64 0.01 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 38"

        bash sh_scripts/experiments/base_trainer.sh $1 39 64 0.01 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 39"

        bash sh_scripts/experiments/base_trainer.sh $1 40 64 0.0 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 40"

    elif [ $2 = 7 ]
    then
        echo "Training mode 1"

        bash sh_scripts/experiments/base_trainer.sh $1 41 8 0.1 2e-5 $eval_steps $3 
        echo "-----------"
        echo "Finished experiment 41"

        bash sh_scripts/experiments/base_trainer.sh $1 42 8 0.01 4e-5 $eval_steps $3 
        echo "-----------"
        echo "Finished experiment 42"

        bash sh_scripts/experiments/base_trainer.sh $1 43 16 0.0 1e-5 $eval_steps $3 
        echo "-----------"
        echo "Finished experiment 43"

        bash sh_scripts/experiments/base_trainer.sh $1 44 16 0.01 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 44"

        echo "Training mode 2"
        bash sh_scripts/experiments/base_trainer.sh $1 45 16 0.1 4e-5 $eval_steps $3  
        echo "-----------"
        echo "Finished experiment 45"

        bash sh_scripts/experiments/base_trainer.sh $1 46 32 0.1 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 46"

        bash sh_scripts/experiments/base_trainer.sh $1 47 32 0.0 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 47"

        bash sh_scripts/experiments/base_trainer.sh $1 48 64 0.01 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 48"

        bash sh_scripts/experiments/base_trainer.sh $1 49 64 0.1 1e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 49"

        bash sh_scripts/experiments/base_trainer.sh $1 50 64 0.0 2e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 50"

        bash sh_scripts/experiments/base_trainer.sh $1 51 64 0.1 3e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 51"

        bash sh_scripts/experiments/base_trainer.sh $1 52 64 0.01 4e-5 $eval_steps $3
        echo "-----------"
        echo "Finished experiment 52"
    else
        echo "Invalid mode"
    fi
fi 
