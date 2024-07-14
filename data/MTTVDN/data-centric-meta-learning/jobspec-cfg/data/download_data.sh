#!/bin/bash
PARENT_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

for i in {44281..44310}; do
    ARFF_PATH=$(python $PARENT_DIR/get_data.py --dataset_id $i)
    CACHED_DATA_DIR=${ARFF_PATH%/*}
    DATA_ZIP_PATH=$(find $CACHED_DATA_DIR -name "*.zip")
    DATASET_NAME=${DATA_ZIP_PATH##*/}
    DATASET_ID=${DATASET_NAME%_*}
    unzip -o $DATA_ZIP_PATH -d $PARENT_DIR
    rm $DATA_ZIP_PATH
done


