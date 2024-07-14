#!/usr/bin/env bash
#
#case "$USER" in
#	awarmerdam*)
#		module purge
#
#		module load 2022
#		module load Anaconda3/2022.05
#		module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
#
##		source activate img
#
#		source keys.sh
#	;;
#	ataboada*)
#		module purge
#
#		module add cuda91/toolkit/9.1.85
#	;;
#	ookah*)
#		source keys.sh
#	;;
#	*)
#		source keys.sh
#	;;
#esac



module load 2023
module load Miniconda3/23.5.2-0

export XTRACE=1
export HYDRA_FULL_ERROR=1
export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE=1
export TOKENIZERS_PARALLELISM=false
