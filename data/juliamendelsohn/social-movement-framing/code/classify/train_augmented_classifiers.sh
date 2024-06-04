#!/bin/bash
#SBATCH --job-name=frame-elements
#SBATCH --account=juliame0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00-12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --mail-type=BEGIN,END,FAIL

epochs=20
seed=42
batchsize=32
modelpath='/nfs/turbo/si-juliame/social-movements/ft_roberta_simpletransformers'
modelname='roberta-ft-st'

augmentation="train_with_1k_per_movement_generations"
datadir="/home/juliame/social-movements/augmented_data/data_splits_06-21-2023/${augmentation}"
baseoutdir="/nfs/turbo/si-juliame/social-movements/roberta_augmented_models/${modelname}/${augmentation}"

categories="Task4:solution Task4:tactics Task4:solidarity Task4:counter Task5:motivational"

for category in ${categories};
    do
    IFS=: read -r task shortname <<< ${category}
    # trainfile="${datadir}/train_relevant_${shortname}.tsv"
    # outdir="${baseoutdir}/${shortname}/all_movements/"
    # mkdir -p ${outdir}

    # echo "Training binary classifier for ${category}"
    # echo "Task: ${task}"
    # echo "Shortname: ${shortname}"
    # echo "Trainfile: ${trainfile}"
    # echo "Outdir: ${outdir}"

    # python train_roberta.py \
    #     --train-file=${trainfile} \
    #     --output-dir=${outdir} \
    #     --categories=${category} \
    #     --num-epochs=${epochs} \
    #     --manual-seed=${seed} \
    #     --train-batch-size=${batchsize} \
    #     --model-path=${modelpath}

    for fold in 0 1 2 3 4;
        do
        trainfile="${datadir}/fold_${fold}/train_relevant_${shortname}.tsv"
        outdir="${baseoutdir}/${shortname}/fold_${fold}/all_movements/"
        mkdir -p ${outdir}
        echo "Training binary classifier for ${category}"
        echo "Task: ${task}"
        echo "Shortname: ${shortname}"
        echo "Trainfile: ${trainfile}"
        echo "Outdir: ${outdir}"
        python train_roberta.py \
        --train-file=${trainfile} \
        --output-dir=${outdir} \
        --categories=${category} \
        --num-epochs=${epochs} \
        --manual-seed=${seed} \
        --train-batch-size=${batchsize} \
        --model-path=${modelpath}
    done
done



