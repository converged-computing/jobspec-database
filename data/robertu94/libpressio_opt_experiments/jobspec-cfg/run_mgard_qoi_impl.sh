#!/usr/bin/env bash

cd ~/projects/libpressio_opt_experiments/
source ~/git/spack/share/spack/setup-env.sh
spack env activate .

tolerance="$1"
dataload="$2"
filename=$(echo $dataload | cut -f 2 -d' ' | cut -d/ -f 3)


echo "$dataload" "tolerance=$tolerance" "mgard-qoi"
pressio \
$dataload \
-W /tmp/$filename.tol$tolerance.dec \
-m region_of_interest -m time -m size -m region_of_interest -M all \
-o mgard:s=-.5 \
-o mgard:tolerance=$tolerance \
-o mgard:qoi_use_metric=1 \
-o mgard:qoi_metric_name="region_of_interest:decomp_average" \
-o region_of_interest:start=0 \
-o region_of_interest:start=0 \
-o region_of_interest:start=0 \
-o region_of_interest:end=3 \
-o region_of_interest:end=500 \
-o region_of_interest:end=500 \
mgard

