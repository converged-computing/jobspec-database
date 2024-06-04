#!/bin/bash                                                                                                                                                                                                      
#PBS -N mpi_task                                                                                                                                                                                                 
#PBS -l nodes=2:ppn=2                                                                                                                                                                                            
#PBS -j oe                                                                                                                                                                                                       
#PBS -o mpi_output.log                                                                                                                                                                                                                                                                                         
#PBS -W stagein=statistics.log@plus7.cluster.univ.kiev.ua:/home/grid/testbed/tb493/hpc-lab2/statistics.log

cd $PBS_O_WORKDIR
                                                                                                                                                                                        
module load icc/18.0  openmpi/3.1.3                                                                                                                                                                              
module load lammps                                                                                                                                                                                               

mpirun -np 4 ./hello > mpi_output.log-$PBS_JOBID
                                                                                                                                                                                                                                                                                                                                                                                     
declare -A node_process_count                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                                                                
if [[ -f statistics.log ]]; then                                                                                                                                                                                 
    while read -r line; do                                                                                                                                                                                       
        node=$(echo $line | cut -d' ' -f1)
        count=$(echo $line | cut -d' ' -f2)
        node_process_count[$node]=$count
    done < statistics.log                                                                                                                                                                                        
fi                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                      
log_file="mpi_output.log-$PBS_JOBID"
while read -r line; do                                                                                                                                                                                           
    if [[ $line == Hello\ world\ from\ processor* ]]; then                                                                                                                                                       
        node=$(echo $line | awk '{print $5}' | cut -d'.' -f1)                                                                                                                                                    
        if [[ -n $node ]]; then                                                                                                                                                                                  
            node_process_count[$node]=$((node_process_count[$node]+1))                                                                                                                                           
        fi                                                                                                                                                                                                       
    fi                                                                                                                                                                                                           
done < "$log_file"                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                                                                                          
> statistics.log                                                                                                                                                                                                 
for node in "${!node_process_count[@]}"; do                                                                                                                                                                      
    echo "$node ${node_process_count[$node]}" >> statistics.log                                                                                                                                                  
done
