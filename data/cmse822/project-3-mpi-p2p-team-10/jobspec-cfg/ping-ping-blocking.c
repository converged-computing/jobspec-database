#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Check if there are exactly two processes.
    if (world_size != 2) {
        if (rank == 0) {
            fprintf(stderr, "World size must be two for %s\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    const int NUM_TESTS = 13; // From 2 bytes to 4096 bytes (2^1 to 2^12)
    const int ITERATIONS = 1000;
    double start_time, end_time;

    // Iterate over different message sizes
    for (int test = 0; test <= NUM_TESTS; ++test) {
        int buffer_size = pow(2, test); // Calculate buffer size
        char* buffer = (char*)malloc(buffer_size); // Allocate buffer

        MPI_Barrier(MPI_COMM_WORLD); // Synchronize before starting timing
        start_time = MPI_Wtime();

        for (int i = 0; i < ITERATIONS; ++i) {
            if (rank == 0) {
                // Rank 0 sends and then receives the message
                MPI_Send(buffer, buffer_size, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
                MPI_Recv(buffer, buffer_size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // Rank 1 receives and then sends the message
                MPI_Recv(buffer, buffer_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer, buffer_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            }
        }

        end_time = MPI_Wtime();

        // Only rank 0 prints the result
        if (rank == 0) {
            double avg_time = (end_time - start_time) * 1e6 / (2 * ITERATIONS); // Calculate average round-trip time
            printf("%d | %f \n", buffer_size, avg_time);
        }

        free(buffer); // Free the allocated buffer for each message size
    }

    MPI_Finalize();
    return 0;
}
