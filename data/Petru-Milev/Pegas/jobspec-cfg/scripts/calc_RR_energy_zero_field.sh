#!/bin/bash 

path=$1 

cd $path 
touch analytical_forces_matrix.csv
touch analytical_forces_matrix_tab.txt 
for atom in $(ls -d */); do 
cd $atom 

python ~/pegas/python_dir/RR_single_csv.py dE_div_dx.csv
python ~/pegas/python_dir/RR_single_csv.py dE_div_dy.csv
python ~/pegas/python_dir/RR_single_csv.py dE_div_dz.csv

paste -d "," "dE_div_dx.txt" "dE_div_dy.txt" "dE_div_dz.txt" >> ../"numerical_forces_matrix.csv"
paste -d '\t' dE_div_dx.txt dE_div_dy.txt dE_div_dz.txt >> ../numerical_forces_matrix_tab.txt
cd .. 
done 

