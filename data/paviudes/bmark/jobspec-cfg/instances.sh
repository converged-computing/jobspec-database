# Launch instances of the ./benchmarking executable.
# Each instances is indexed by a number from 0--max which will be the value of $SLURM_JOBID variable.
sizes=(20720 22400 24080 25886 27090 28350)
echo "Benchmarking a code with ${sizes[$1]} qubits"
# Create the input file for the benchmarking procedure
echo -e "mode simulate\nfamily 6\ncodes 1 ${sizes[$1]}\nnoise 1\np 0.005 0.005 0.001\ntrials 1000000" > hex_k${sizes[$1]}.txt 
./benchmarking hex_k${sizes[$1]}.txt