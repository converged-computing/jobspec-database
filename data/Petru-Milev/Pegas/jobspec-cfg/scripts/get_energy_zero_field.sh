#!/bin/bash 

path=$1  																#Saving the input into a var 

cd $path 																#going into the saved path 

arr=("x" "y" "z") 

for atom in $(ls -d */); do 														#Iterating over all the atoms in the folder 
	cd $atom 
	for coord in "${arr[@]}"; do 													#Iterate over x y z coordinates
		touch "dE_for_${coord}.txt"												#Create temporaty files 
		touch "d${coord}.txt" 
		for displacement in $(ls -d $coord*/); do 										#Go into displacement folder (to modify later)
			cd ${displacement}
			cd log
			pwd
			awk '/SCF Done/ {print $5}' 0_0.log >> ../../"dE_for_${coord}.txt"							#Get the energy into tmp file 
			echo "${displacement:1:-1}" >> ../../"d${coord}.txt"								#Get the displacement into tmp file 
			cd ../..				
		done 
		#cat "dE_for_${coord}.txt"
		#cat "d${coord}.txt"
		paste -d "," "d${coord}.txt" "dE_for_${coord}.txt" > "dE_div_d${coord}.csv"							#Create csv file
		rm "dE_for_${coord}.txt" "d${coord}.txt"										#Remove tmp files
	
	done 
	cd ..
done
