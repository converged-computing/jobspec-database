#!/bin/bash

#SBATCH --job-name=pred-cross
#SBATCH --account=juliame0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00-03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --mail-type=BEGIN,END,FAIL


categories="Task1:relevance"
shortname="relevance"
split='full'

# categories="Task2:stance"
# shortname='stance'
# split='relevant'

# categories="Task3:diagnostic,Task4:prognostic,Task5:motivational"
# shortname="macro-frames"
# split='relevant'

# categories="Task3:identify,Task3:blame,Task4:solution,Task4:tactics,Task4:solidarity,Task4:counter,Task5:motivational"
# shortname="frame-elements"
# split='relevant'

modelmovement='all_movements'
modelname='roberta-ft-st'
date="06-21-2023"
datadir="/home/juliame/social-movements/annotated_data/data_splits_${date}"
modelbase="/nfs/turbo/si-juliame/social-movements/roberta_models/${modelname}/${shortname}_${date}"
outdir="/home/juliame/social-movements/predictions/${modelname}/${modelmovement}_model_preds/${shortname}_${date}"


# All movements model on each movement data
for fold in 0 1 2 3 4;
    do
    for movement in guns immigration lgbtq;
        do
        modeldir="${modelbase}/fold_${fold}/${modelmovement}/"
        evalfile="${datadir}/fold_${fold}/dev_${split}_${movement}.tsv"
        outfile="${outdir}/fold_${fold}/dev_${split}_${movement}.tsv"
        categories=${categories}
        mkdir -p ${outdir}
        python predict_roberta.py \
            --eval-file=${evalfile} \
            --model-dir=${modeldir} \
            --categories=${categories} \
            --out-file=${outfile} 
        done
    done

for movement in guns immigration lgbtq;
    do
    modeldir="${modelbase}/${modelmovement}/"
    evalfile="${datadir}/test_${split}_${movement}.tsv"
    outfile="${outdir}/test_${split}_${movement}.tsv"
    categories=${categories}
    mkdir -p ${outdir}
    python predict_roberta.py \
        --eval-file=${evalfile} \
        --model-dir=${modeldir} \
        --categories=${categories} \
        --out-file=${outfile} 
    done
