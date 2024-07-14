#!/bin/bash 

function generate() {
local line=$1 
local linenumber=$2
local delta=$3 
local atom=$(echo $line| cut -d ' ' -f 1)
local x=$(echo $line| cut -d ' ' -f 2)
local y=$(echo $line| cut -d ' ' -f 3)
local z=$(echo $line| cut -d ' ' -f 4)

echo "This is atom $atom, x coord is $x, y coord is $y, z coord is $z"
mkdir "./input_geometries/${linenumber}_${atom}" 												#Creating folder to save the new geom from mod this line 

arr=($x $y $z) 																	#Array of x yz values
arr_displacement=(0.005 0.01 0.02 0.04 0.08)													#Array of displacement values. To modify later

#----------------------------------

i=1																		#Counting the cycle
for j in "${arr[@]}"; do															#Iterating though the x y z values

for displacement in "${arr_displacement[@]}"; do												#Iterating though the displacement values

#Making files for positive displacement 

tmp=$j+$displacement																#Creating a variable to be submited to the bc 
new_coord=$(echo $tmp | bc -l)															#Value of the modified coordinate

if [ $i -eq 1 ]; then																#A series of if values to make the new line 
local path="./input_geometries/${linenumber}_${atom}/x+${displacement}/"										#Getting a variable for the path of new file
local newline="$atom $new_coord $y $z"														#Getting a new line to be inserted 
fi 

if [ $i -eq 2 ]; then
local path="./input_geometries/${linenumber}_${atom}/y+${displacement}/"
local newline="$atom $x $new_coord $z"
fi

if [ $i -eq 3 ]; then
local path="./input_geometries/${linenumber}_${atom}/z+${displacement}/"
local newline="$atom $x $y $new_coord"
fi

mkdir $path																	#Creating the new file
awk -v var=$linenumber 'NR < var {print $1, $2, $3, $4}' $file > ${path}coord_ang								#Adding all the lines before the one being changed 
echo $newline >> ${path}coord_ang																#Changed line
awk -v var=$linenumber 'NR > var {print $1, $2, $3, $4}' $file >> ${path}coord_ang									#Adding lines after the line that was changed 

#Making files for negative displacement 

tmp=$j-$displacement                                                                                                                            #Creating a variable to be submited to the bc
new_coord=$(echo $tmp | bc -l)                                                                                                                     #Value of the modified coordinate

if [ $i -eq 1 ]; then                                                                                                                           #A series of if values to make the new line
local path="./input_geometries/${linenumber}_${atom}/x-${displacement}/"                                                                         #Getting a variable for the path of new file
local newline="$atom $new_coord $y $z"                                                                                                          #Getting a new line to be inserted
fi

if [ $i -eq 2 ]; then
local path="./input_geometries/${linenumber}_${atom}/y-${displacement}/"
local newline="$atom $x $new_coord $z"
fi

if [ $i -eq 3 ]; then
local path="./input_geometries/${linenumber}_${atom}/z-${displacement}/"
local newline="$atom $x $y $new_coord"
fi

mkdir $path                                                                                                                                     #Creating the new file
awk -v var=$linenumber 'NR < var {print $1, $2, $3, $4}' $file > ${path}coord_ang                                                                          #Adding all the lines before the one being changed
echo $newline >> ${path}coord_ang                                                                                                                          #Changed line
awk -v var=$linenumber 'NR > var {print $1, $2, $3, $4}' $file >> ${path}coord_ang                                                                      #Adding lines after the line that was changed


done
((i++))
done

}

#--------------------------------

file=$1 

file=$1																	#Assigning the input to variables  
delta=0.005
nrlines=$(wc -l < $file)														#Getting the number of lines of a file in a variable  

mkdir ./input_geometries 														#making a folder where to save all the geometries 

for i in $(seq 1 $nrlines); do														#Getting each line of the file a variabl
awk -v var=$i 'NR==var' $file
generate "$(awk -v var=$i 'NR==var' $file)" $i $delta
done
