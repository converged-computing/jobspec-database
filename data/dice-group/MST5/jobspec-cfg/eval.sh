#!/bin/bash
set -eu
# Setting variables
# Directory where fine-tuned models are stored: fine-tuned_models
MODEL_ROOT_DIR=$1
# Name of the model: qald9plus-finetune_mt5-base_lc-ent
MODEL_NAME=$2
# Path to the qald test file
TEST_FILE=$3
# Root directory where a new directory with model_name will be created to store predictions and results
OUTPUT_DIR=$4
# Comma separated values: en,de,es
LANGS=$5
# linguistic context : true/false
LC=$6
# entity knowledge : true/false
EK=$7
# Perform Gerbil evaluation: true/false
GE=$8
# Knowledge Graph: DBpedia / Wikidata
KNOWLEDGE_GRAPH=$9

#KNOWLEDGE_GRAPH="DBpedia"
#KNOWLEDGE_GRAPH="Wikidata"
question_padding_length=128
entity_padding_length=64



if [ "$LC" = true ]; then
  linguistic_context="--linguistic_context"
else
  linguistic_context="--no-linguistic_context"
fi

if [ "$EK" = true ]; then
  entity_knowledge="--entity_knowledge"
else
  entity_knowledge="--no-entity_knowledge"
fi

EXTRA_PARAMS=""

if [ "$GE" = "false" ]; then
    echo "No Gerbil evaluation will be performed."
else
    echo "Gerbil evaluation will be performed."
    EXTRA_PARAMS+="--gerbil_eval "
fi

# Add --use_gold_ents to use gold entities
# Add --translate_target_lang en to translate all questions to en
python code/pred_build_eval_qald.py \
      --model "${MODEL_ROOT_DIR}/${MODEL_NAME}" \
      -t $TEST_FILE \
      --knowledge_graph ${KNOWLEDGE_GRAPH} \
      -o "${OUTPUT_DIR}/${MODEL_NAME}" \
      -l $LANGS \
      ${linguistic_context} \
      ${entity_knowledge} \
      --question_padding_length ${question_padding_length} \
      --entity_padding_length ${entity_padding_length} \
      $EXTRA_PARAMS \
      2>&1


