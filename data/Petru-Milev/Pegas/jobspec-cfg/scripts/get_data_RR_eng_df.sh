#!/bin/bash 

path=$1

cd $path

for atom in $(ls -d  ${HOME}/pegas/input_geometries/*/ | xargs -n 1 basename | sort -h ); do 
cd $atom; 
for displacement in $(ls); do 
cd $displacement 
cd log 
pwd
echo "Running ~/pegas/scripts/get_miu.sh"
bash ~/pegas/scripts/get_miu.sh 
echo "Running ~/pegas/python_dir/calc_deriv.py" 
python ~/pegas/python_dir/calc_deriv.py
cd ..
cd ..
done
cd .. 
done
