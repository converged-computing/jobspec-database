#!/bin/bash

cd data/dbgen/
echo "Create dataSet $1"
rm -r -f 1/ 2/ 4/ 8/ 16/ 32/ 64/
rm -f *.tbl
./dbgen -f -s "$1"
mkdir "$1"
mv *.tbl "$1"
echo "Data set created ..."
