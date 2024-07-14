#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <string.h>

typedef struct {
    int thread_id;
    int n;
    int num_threads;
    int *array1;
    int *array2;
    int *array3;
} ThreadData;

int get_index(int row, int col, int n) {
    return row * n + col;
}
void *multiply(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int n = data->n;
    int num_threads = data->num_threads;

    int rows_per_thread = n / num_threads;
    int start_row = data->thread_id * rows_per_thread;
    int end_row = start_row + rows_per_thread;

    if (data->thread_id == num_threads - 1) {
        end_row = n;  // Last thread may do more work if n is not divisible by num_threads
    }

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += data->array1[get_index(i, k, n)] * data->array2[get_index(k, j, n)];
            }
            data->array3[get_index(i, j, n)] = sum;
        }
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    printf("Running cPThread:\n\n");

    if (argc < 7) {
        fprintf(stderr, "Usage: %s <array_size> <num_threads> <num_cores> <job_name> <num_nodes> <ntasks_per_node>\n", argv[0]);
        return 1;
    }

    char *endptr;
    int n = strtol(argv[1], &endptr, 10); // Parse n from command-line argument
    if (endptr == argv[1] || *endptr != '\0' || n <= 0) {
        fprintf(stderr, "The number of rows/columns must be a positive integer.\n");
        return 1;
    }

    int num_threads = strtol(argv[2], &endptr, 10); // Parse num_threads from command-line argument
    if (endptr == argv[2] || *endptr != '\0' || num_threads <= 0) {
        fprintf(stderr, "The number of threads must be a positive integer.\n");
        return 1;
    }

    int *array1 = malloc(n * n * sizeof(int));
    int *array2 = malloc(n * n * sizeof(int));
    int *array3 = malloc(n * n * sizeof(int));
    if (!array1 || !array2 || !array3) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Initialize array1 and array2 with ones, array3 with zeros
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            array1[get_index(i, j, n)] = 1;
            array2[get_index(i, j, n)] = 1;
            array3[get_index(i, j, n)] = 0;
        }
    }

    ThreadData *thread_data = malloc(num_threads * sizeof(ThreadData));
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    if (!thread_data || !threads) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    struct timespec start, finish; // For timing
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create threads to perform matrix multiplication
    for (int i = 0; i < num_threads; i++) {
        thread_data[i] = (ThreadData){ .thread_id = i, .n = n, .num_threads = num_threads, .array1 = array1, .array2 = array2, .array3 = array3 };
        pthread_create(&threads[i], NULL, multiply, (void *)&thread_data[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish); // End of timing
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    // Conditionally print the result matrix
    if (n <= 15) {
        printf("Resultant matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%d ", array3[get_index(i, j, n)]);
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
        fprintf(csv_file, "%s,%s,%s,%s,%s,%s,%s,%4f\n", __FILE__, argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], elapsed);
        fclose(csv_file);
    } else {
        perror("Unable to open the CSV file");
    }

    // Cleanup
    free(array1);
    free(array2);
    free(array3);
    free(thread_data);
    free(threads);

    return 0;
}
