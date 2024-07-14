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

    // Ensure there are exactly two processes for this ping-pong
    if (world_size != 2) {
        fprintf(stderr, "World size must be two for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int NUM_TESTS = 13;  // From 2 bytes to 4096 bytes
    const int ITERATIONS = 1000;
    double start_time, end_time;

    for (int test = 0; test < NUM_TESTS; ++test) {
        int buffer_size = pow(2, test); // Message size in bytes
        char* buffer = (char*)malloc(buffer_size);
        MPI_Request request;
        MPI_Status status;

        start_time = MPI_Wtime();

        for (int i = 0; i <= ITERATIONS; ++i) {
            if (rank == 0) {
                MPI_Isend(buffer, buffer_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status); // Wait for the send to complete
                MPI_Irecv(buffer, buffer_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status); // Wait for the receive to complete
            } else if (rank == 1) {
                MPI_Irecv(buffer, buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status); // Wait for the receive to complete
                MPI_Isend(buffer, buffer_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status); // Wait for the send to complete
            }
        }

        end_time = MPI_Wtime();

        if (rank == 0) {
            double avg_time = (end_time - start_time) * 1e6 / (2 * ITERATIONS); // Calculate average round-trip time
            printf("%d | %f \n", buffer_size, avg_time); 
        }

        free(buffer);
    }

    MPI_Finalize();
    return 0;
}
