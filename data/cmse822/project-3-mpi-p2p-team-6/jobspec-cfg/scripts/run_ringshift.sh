NUM_EXCHANGES=100
OUTPUT_REPORT_FILE="results/part3_ringshift.csv"
RINGSHIFT_EXECUTABLE="bin/ring_shift"

# Remove earlier logs 
rm $OUTPUT_REPORT_FILE

# Create an array of message msg_sizes
MSG_SIZES=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)

# Vary the number of processes
NUM_PROCS=(2 4 8 16 32 64 128)

for MSG_SIZE in ${MSG_SIZES[@]}
do
    echo -e "Running ringshift for message size: $MSG_SIZE\n"
    for NUM_PROCS in ${NUM_PROCS[@]}
    do
        echo -e "Running ringshift for $NUM_PROCS processes\n"
        mpiexec -np $NUM_PROCS $RINGSHIFT_EXECUTABLE $MSG_SIZE $NUM_EXCHANGES 1 $OUTPUT_REPORT_FILE
    done
done