#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="sam_job"
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=128
#SBATCH --mem=490G
#SBATCH --time=02:00:00

# Load necessary modules (e.g., for MKL, BLIS, or other dependencies)
#module load architecture/AMD           # EPYC case
#module load architecture/Intel          # THIN case
module load mkl/latest
module load openBLAS/0.3.23-omp

# Policy
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export LD_LIBRARY_PATH=/u/dssc/slippo00/PROJECT_sl_and_gv/Foundations_of_HPC_2022/Assignment/exercise2/blis/lib:$LD_LIBRARY_PATH

# Parameter
current_type=EPYC
current_policy=spread
size=10000

# For each file
# The possible files are: sgemm_mkl.x dgemm_mkl.x sgemm_oblas.x dgemm_oblas.x sgemm_blis.x dgemm_blis.x sgemm_mkl_opt.x dgemm_mkl_opt.x sgemm_oblas_opt.x dgemm_oblas_opt.x sgemm_blis_opt.x dgemm_bli>
for FILE in sgemm_blis_opt.x dgemm_blis_opt.x; do

        # Define your matrix multiplication code's executable path
        matrix_multiply_executable="./${FILE}"

        # Create a directory for storing the CSV results
        results_dir="csv_results_scalability_${current_type}"
        mkdir -p "$results_dir"

        # Define the CSV file path to store the summary results
        summary_csv="${results_dir}/${matrix_multiply_executable}_${current_type}_${size}_${current_policy}_summary.csv"

        # Define the CSV file path to store the detailed results
        detailed_csv="${results_dir}/${matrix_multiply_executable}_${current_type}_${size}_${current_policy}_detailed.csv"

        # Define the number of iterations for each size
        num_iterations=5

        # Initialize the summary CSV file with header
        echo "Number of cores,Size,Elapsed Time (s),GFLOPS" > "$summary_csv"

        # Initialize the detailed CSV file with header
        echo "Number of cores,Size,Iteration,Elapsed Time (s),GFLOPS" > "$detailed_csv"

        for num in 1 2 4 16 32 48 64 80 96 112 128; do

            export OMP_NUM_THREADS=$num
            export BLIS_NUM_THREADS=$num

            total_time=0
            total_gflops=0

            for ((iteration = 1; iteration <= num_iterations; iteration++)); do
                # Run your matrix multiplication code for the current size
                output=$($matrix_multiply_executable $size $size $size)

                # Extract time and GFLOPS from the output using grep and awk
                # Extract time and GFLOPS from the output using grep and awk
                time=$(echo "$output" | grep -oP 'Elapsed time \K[0-9.]+')
                gflops=$(echo "$output" | grep -oP '[0-9.]+ GFLOPS' | awk '{print $1}')

                total_time=$(echo "$total_time + $time" | bc -l)
                total_gflops=$(echo "$total_gflops + $gflops" | bc -l)

                # Append the detailed results to the detailed CSV file
                echo "$num,$size,$iteration,$time,$gflops" >> "$detailed_csv"
            done

            # Calculate the average time and GFLOPS
            avg_time=$(echo "scale=5; $total_time / $num_iterations" | bc)
            avg_gflops=$(echo "scale=5; $total_gflops / $num_iterations" | bc)

            # Append the summary results to the summary CSV file
            echo "$num,$size,$avg_time,$avg_gflops" >> "$summary_csv"
        done
done

