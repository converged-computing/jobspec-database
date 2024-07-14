#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
	char cwd[1024];
if (getcwd(cwd, sizeof(cwd)) != NULL) {
    printf("Current working directory: %s\n", cwd);
} else {
    perror("getcwd() error");
    return 1;
}


    printf("Running cOpenMP:\n\n");

    if (argc < 7) {
        fprintf(stderr, "Usage: %s <array_size> <num_threads> <num_cores> <job_name> <num_nodes> <ntasks_per_node>\n", argv[0]);
        return 1;
    }

    char *endptr;
    int n = strtol(argv[1], &endptr, 10); // Parse matrix size from command-line argument

    if (endptr == argv[1] || *endptr != '\0' || n <= 0) {
        fprintf(stderr, "The number of rows/columns must be a positive integer.\n");
        return 1;
    }

    int num_threads = strtol(argv[2], &endptr, 10); // Parse the number of threads from command-line argument

    if (endptr == argv[2] || *endptr != '\0' || num_threads <= 0) {
        fprintf(stderr, "The number of threads must be a positive integer.\n");
        return 1;
    }
    
    omp_set_num_threads(num_threads); // Set the number of threads for OpenMP to use

    printf("Creating 3 arrays of size %dx%d...\n\n", n, n);
    int *array1 = malloc(n * n * sizeof(int));
    int *array2 = malloc(n * n * sizeof(int));
    int *array3 = malloc(n * n * sizeof(int));

    if (!array1 || !array2 || !array3) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    printf("Initializing matrix values...\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            array1[i * n + j] = 1;
            array2[i * n + j] = 1;
            array3[i * n + j] = 0; // Initialize to 0
        }
    }

    printf("Performing matrix multiplication using OpenMP with %d threads...\n", num_threads);
    double start_time = omp_get_wtime();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                array3[i * n + j] += array1[i * n + k] * array2[k * n + j];
            }
        }
    }

    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;

    if (n <= 15) {
        printf("Resultant matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", array3[i * n + j]);
            }
            printf("\n");
        }
    } else {
        printf("Result matrix of size %dx%d is too large to display.\n", n, n);
    }

    printf("Writing statistics to file...\n");
    // Write statistics to file
    FILE *csv_file;
    csv_file = fopen("report/data.csv", "a"); // Open file in append mode
    if (csv_file != NULL) {
        fprintf(csv_file, "%s,%s,%s,%s,%s,%s,%s,%4f\n", __FILE__, argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], elapsed);
        fclose(csv_file);
    } else {
        perror("Unable to open the CSV file");
    }

    free(array1);
    free(array2);
    free(array3);

    return 0;
}
