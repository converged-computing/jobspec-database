// Title: cNoThreads
// Author: Anders Jensen
// Description: Matrix Multiplication in c with no threading involved
// Version: 1.0 (02/27/2024)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    printf("Running cNoThreads:\n\n");

    if (argc < 7) {
        fprintf(stderr, "Usage: %s <array_size> <num_threads> <num_cores> <job_name> <num_nodes> <ntasks_per_node>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]); // n = number of rows and columns

    // Check if n is valid
    if (n <= 0) {
        fprintf(stderr, "The number of rows/columns must be greater than zero.\n");
        return 1;
    }

    int array1[n][n];
    int array2[n][n];
    int array3[n][n];

    // Initialize elements to 1 for array1 and array2
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            array1[i][j] = 1;
            array2[i][j] = 1;
        }
    }

    // Timing multiplication
    clock_t start = clock();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            array3[i][j] = 0;
            for (int k = 0; k < n; k++) {
                array3[i][j] += array1[i][k] * array2[k][j];
            }
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    // Conditionally print the resulting matrix
    if (n <= 15) {
        printf("Resulting Matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", array3[i][j]);
            }
            printf("\n");
        }
    } else {
        printf("Matrix of size %dx%d created but too large to display.\n", n, n);
    }

    // Write statistics to file
    FILE *csv_file;
    csv_file = fopen("report/data.csv", "a"); // Open file in append mode
    if (csv_file != NULL) {
        fprintf(csv_file, "%s,%s,%s,%s,%s,%s,%s,%4f\n", __FILE__, argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], time_spent);
        fclose(csv_file);
    } else {
        perror("Unable to open the CSV file");
    }

    return 0;
}
