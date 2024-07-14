#!/bin/bash 

path=$1 

cd $path 

for atom in $(ls -d */ | xargs -n 1 basename | sort -h); do 
cd $atom; 
pwd
touch x.txt
touch miu_x_val.txt
touch miu_y_val.txt
touch miu_z_val.txt

for i in $(ls -d */ | grep "x"); do
cd $i/log 
echo "${i:1:-1}" >> ../../x.txt
awk '{print}' miu_x.txt >> ../../miu_x_val.txt
awk '{print}' miu_y.txt >> ../../miu_y_val.txt
awk '{print}' miu_z.txt >> ../../miu_z_val.txt
cd ../..
done
paste -d ',' x.txt miu_x_val.txt miu_y_val.txt miu_z_val.txt > x.csv 

rm miu_x_val.txt miu_y_val.txt miu_z_val.txt x.txt

touch y.txt
touch miu_x_val.txt
touch miu_y_val.txt
touch miu_y_val.txt

for i in $(ls -d */ | grep "y"); do
cd $i/log
echo "${i:1:-1}" >> ../../y.txt
awk '{print}' miu_x.txt >> ../../miu_x_val.txt
awk '{print}' miu_y.txt >> ../../miu_y_val.txt
awk '{print}' miu_z.txt >> ../../miu_z_val.txt
cd ../..
done
paste -d ',' y.txt miu_x_val.txt miu_y_val.txt miu_z_val.txt > y.csv

rm miu_x_val.txt miu_y_val.txt miu_z_val.txt y.txt 

touch z.txt
touch miu_x_val.txt
touch miu_y_val.txt
touch miu_y_val.txt

for i in $(ls -d */ | grep "z"); do
cd $i/log
echo "${i:1:-1}" >> ../../z.txt
awk '{print}' miu_x.txt >> ../../miu_x_val.txt
awk '{print}' miu_y.txt >> ../../miu_y_val.txt
awk '{print}' miu_z.txt >> ../../miu_z_val.txt
cd ../..
done
paste -d ',' z.txt miu_x_val.txt miu_y_val.txt miu_z_val.txt > z.csv

rm miu_x_val.txt miu_y_val.txt miu_z_val.txt z.txt
python ~/pegas/python_dir/calc_deriv_miu_dr.py
cd ..
done 
