#!/bin/bash

#SBATCH -p a100q # partition (queue)
#SBATCH -t 0-15:00 # time limit (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH  --gpus-per-node 1

export PATH=$PATH:/usr/local/cuda/bin

nvidia-smi

nvcc --version
make 

# Define the path to the directory containing the matrices
MATRIX_DIR="/global/D1/homes/james/sparcity/suitesparse/mtx/amd/"

# List of input matrices
INPUT_MATRICES=("AMD/G3_circuit.mtx.gz"
"Bodendiek/CurlCurl_4.mtx.gz"
"DIMACS10/333SP.mtx.gz"
"DIMACS10/adaptive.mtx.gz"
"DIMACS10/AS365.mtx.gz"
"DIMACS10/delaunay_n24.mtx.gz"
"DIMACS10/europe_osm.mtx.gz"
"DIMACS10/hugebubbles-00020.mtx.gz"
"DIMACS10/hugetrace-00020.mtx.gz"
"DIMACS10/hugetric-00020.mtx.gz"
"DIMACS10/kron_g500-logn21.mtx.gz"
"DIMACS10/M6.mtx.gz"
"DIMACS10/NLR.mtx.gz"
"DIMACS10/rgg_n_2_24_s0.mtx.gz"
"DIMACS10/road_usa.mtx.gz"
"DIMACS10/venturiLevel3.mtx.gz"
"Fluorem/HV15R.mtx.gz"
"Freescale/circuit5M.mtx.gz"
"GAP/GAP-road.mtx.gz"
"GenBank/kmer_V2a.mtx.gz"
"Gleich/wb-edu.mtx.gz"
"Janna/Queen_4147.mtx.gz"
"LAW/uk-2005.mtx.gz"
"MAWI/mawi_201512020000.mtx.gz"
"Mycielski/mycielskian19.mtx.gz"
"Pajek/patents.mtx.gz"
"Rajat/rajat31.mtx.gz"
"Schenk_AFE/af_shell10.mtx.gz"
"Schenk/nlpkkt240.mtx.gz"
"SNAP/soc-LiveJournal1.mtx.gz"
"SNAP/com-LiveJournal.mtx.gz"
"SNAP/cit-Patents.mtx.gz"
"SNAP/sx-stackoverflow.mtx.gz"
"SNAP/wiki-Talk.mtx.gz"
"SNAP/roadNet-CA.mtx.gz"
"SNAP/wiki-topcats.mtx.gz"
"SNAP/as-Skitter.mtx.gz"
"vanHeukelum/cage15.mtx.gz"
"VLSI/stokes.mtx.gz"
"Zaoui/kkt_power.mtx.gz"
)  
       

mkdir -p /work/$USER/tmp1

for input_matrix in "${INPUT_MATRICES[@]}"; do

    # Extract the base file name (without path or extension)
    base_filename=$(basename "$input_matrix" .gz)

    # Concatenate path with matrix name
    path_to_input_matrix="$MATRIX_DIR$input_matrix"

    # Decompress the input matrix
    gunzip -c "$path_to_input_matrix" > "/work/santiago/tmp1/$base_filename"

    # Run the program with the input matrix
    srun ncu -k load_balancing_kernel -c 1 --metrics l1tex__m_xbar2l1tex_read_bytes.sum,l1tex__m_l1tex2xbar_write_bytes.sum ./spmm_csr_example "/work/santiago/tmp1/$base_filename"

    # Clean up the temporary decompressed file
    rm "/work/santiago/tmp1/$base_filename"

done


