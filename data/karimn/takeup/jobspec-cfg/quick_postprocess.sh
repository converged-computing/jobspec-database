#!/usr/bin/env bash

postprocess_model () {

    echo "RUNNING SCRIPT: $1"
    echo "RUNNING VERSION: $2"
    echo "RUNNING MODEL: $3"
    echo "INPUT ARGS: $4"
    echo "OUTPUT ARGS: $5"
    echo "RUNNING ADD ARGS: ${@:6}"

    Rscript --no-save \
            --no-restore \
            --verbose \
            $1 \
	    $2 \
            --model=$3 \
            $4 \
            $5 \
            "${@:6}" \
            1 2 3 4 > temp/log/struct-postprocess_$3_$2_$1.txt 2>&1

}


postprocess_struct_models () {
    echo "RUNNING MODEL: $1"
    echo "RUNNING VERSION: $2"
    echo "Postprocess args: $3"


	module load R/4.2.0

    Rscript --no-save \
            --no-restore \
            --verbose \
            quick_ate_postprocess.R \
	    $2 \
            $3 \
            --model=$1 \
            1 2 3 4 > temp/log/struct-postprocess_$1_$2.txt 2>&1 &
    Rscript --no-save \
            --no-restore \
            --verbose \
            quick_ate_postprocess.R \
	    $2 \
            $3 \
            --model=$1 \
            --prior \
            1 2 3 4 > temp/log/struct-postprocess_$1_$2.txt 2>&1  &
    Rscript --no-save \
            --no-restore \
            --verbose \
            quick_roc_postprocess.R \
	    $2 \
            $3 \
            --model=$1 \
            1 2 3 4 > temp/log/struct-postprocess_$1_$2.txt 2>&1 &
    Rscript --no-save \
            --no-restore \
            --verbose \
            quick_submodel_postprocess.R \
	    $2 \
            $3 \
            --model=$1 \
            1 2 3 4 > temp/log/struct-postprocess_$1_$2.txt 2>&1 &
   wait
}

postprocess_rf_models () {

    echo "RUNNING MODEL: $1"
    echo "RUNNING VERSION: $2"
    echo "Postprocess args: $3"


    Rscript --no-save \
            --no-restore \
            --verbose \
            quick_ate_postprocess.R \
	    $2 \
            $3 \
            --model=$1 \
            1 2 3 4 > temp/log/rf-postprocess_$1_$2.txt 2>&1 &
    Rscript --no-save \
            --no-restore \
            --verbose \
            quick_ate_postprocess.R \
	    $2 \
            $3 \
            --model=$1 \
            --prior \
            1 2 3 4 > temp/log/rf-postprocess_$1_$2.txt 2>&1 &
     wait
}

