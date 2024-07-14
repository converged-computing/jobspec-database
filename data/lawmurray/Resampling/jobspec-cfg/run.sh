#!/bin/sh

COMMON_OPTS="--Zs 9 --Ps 19 --reps 256 --without-sort --disable-assert --enable-single --enable-cuda --nthreads 8"

libbi test_resampler $CONFIG_OPTS $COMMON_OPTS --resampler multinomial --output-file results/multinomial-$OUTPUT_SUFFIX.nc
libbi test_resampler $CONFIG_OPTS $COMMON_OPTS --resampler stratified --output-file results/stratified-$OUTPUT_SUFFIX.nc
libbi test_resampler $CONFIG_OPTS $COMMON_OPTS --resampler systematic --output-file results/systematic-$OUTPUT_SUFFIX.nc
libbi test_resampler $CONFIG_OPTS $COMMON_OPTS --resampler metropolis --output-file results/metropolis-$OUTPUT_SUFFIX.nc
libbi test_resampler $CONFIG_OPTS $COMMON_OPTS --resampler rejection --output-file results/rejection-$OUTPUT_SUFFIX.nc
libbi test_resampler $CONFIG_OPTS $COMMON_OPTS --resampler sort --output-file results/sort-$OUTPUT_SUFFIX.nc
libbi test_resampler $CONFIG_OPTS $COMMON_OPTS --resampler ess --output-file results/ess-$OUTPUT_SUFFIX.nc
