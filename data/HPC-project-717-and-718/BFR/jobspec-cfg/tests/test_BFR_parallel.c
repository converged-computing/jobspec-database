#include "../lib/BFR_parallel.h"
#include <time.h>


    

// void test_hierchical_clustering_multi() {
//     // Create a sample CompressedSets object
//     CompressedSets C;
//     // Initialize C with some values
//     int i;
//     for ( i = 0; i < 10; i++) {
//         C.sets[i].size = 10;
//         int j;
//         for (j = 0; j < 10; j++) {
//             C.sets[i].elements[j] = i * 10 + j;
//         }
//     }
    
//     // Call the hierchical_clustering_multi function
//     hierchical_clustering_multi(&C);
    
//     // Add assertions to verify the correctness of the function
    

//     // print the results
//     int i;
//     for (i = 0; i < 10; i++) {
//         printf("Cluster %d: ", i);
//         int j;
//         for (j = 0; j < C.sets[i].size; j++) {
//             printf("%d ", C.sets[i].elements[j]);
//         }
//         printf("\n");
//     }
// }

// void test_initClustersWithCentroids(){
//     // create a set of clusters K
//     int K = 10;
//     int n = 1000;
//     int d = 10;

//     Cluster *clusters = (Cluster *)malloc(K * sizeof(Cluster));

//     // create a list of random points in d dimensions
//     Point *points = (Point *)malloc(n * sizeof(Point));

//     for (int i = 0; i < n; i++) {
//         points[i].coordinates = (double *)malloc(d * sizeof(double));
//         int j;
//         for (j = 0; j < d; j++) {
//             srand(time(NULL));
//             points[i].coordinates[j] = ((double) rand() %%1000)/ 100;
//         }
//     }


//     // call the initClustersWithCentroids function
//     initClustersWithCentroids(points, n, d, K, clusters);
    

//     // Add assertions to verify the correctness of the function

//     // print the results
//     for (int i = 0; i < K; i++) {
//         printf("Cluster %d: ", i);
//         int j;
//         for (j = 0; j < clusters[i].size; j++) {
//             printf("%d ", clusters[i].elements[j]);
//         }
//         printf("\n");
//     }
// }


void test_hierchical_clustering_par(bool debug, int rank, int size){
    int DIM = 10;
    // Create a sample CompressedSets object
    CompressedSets C_st;
    CompressedSets *C = (CompressedSets *)malloc(sizeof(CompressedSets));
    // Initialize C with some values
    C -> number_of_sets = 10;

     
    if (DEBUG){
        MPI_Barrier(MPI_COMM_WORLD);
        printf("initialization on node %d, with %d sets\n", rank, C->number_of_sets);
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // create 50 compressed sets
    C->sets = (CompressedSet *)malloc(10 * sizeof(CompressedSet));
    int i;
    for (i = 0; i < 10; i++) {
        CompressedSet * new_set = malloc(sizeof(CompressedSet));
        C->sets[i] = *new_set;
        free(new_set);
        C->sets[i].number_of_points = 10;
        
        //fill sum and sum of squares
        
        srand(time(NULL));
        int j;
        for (j = 0; j < 10; j++) {
            C->sets[i].sum[j] =(double)(j*i*23 % 100);
            C->sets[i].sum_square[j] = (double)(j*i*23 % 100);
        }
    }
    
    // create a copy of the original compressed sets
    CompressedSets *C_copy = (CompressedSets *)malloc(sizeof(CompressedSets));
    C_copy->number_of_sets = C->number_of_sets;
    C_copy->sets = (CompressedSet *)malloc(C->number_of_sets * sizeof(CompressedSet));
    for (i = 0; i < C->number_of_sets; i++) {
        CompressedSet * new_set = malloc(sizeof(CompressedSet));
        C_copy->sets[i] = *new_set;
        free(new_set);
        C_copy->sets[i].number_of_points = C->sets[i].number_of_points;
        int j;
        for (j = 0; j < DIM; j++) {
            C_copy->sets[i].sum[j] = (double)C->sets[i].sum[j];
            C_copy->sets[i].sum_square[j] = (double)C->sets[i].sum_square[j];
        }
    }
;

    // if (rank== MASTER && DEBUG) print_compressedsets(*C);
    
    if (DEBUG){
        MPI_Barrier(MPI_COMM_WORLD);
        printf("starts operations on node %d\n", rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    
    // calculate the time taken by the parallel version
    clock_t start, end;

    


    start = clock();
    // Call the hierchical_clustering_par function
    hierachical_clust_parallel(C, rank, size);
    end = clock();

    if (DEBUG)MPI_Barrier(MPI_COMM_WORLD);

    double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by parallel version: %f\n", time_taken);

    Cluster *clusters = (Cluster *)malloc(10 * sizeof(Cluster));


    if (rank == MASTER){
        // print the results of the parallel version
        for (i = 0; i < C->number_of_sets; i++) {
            printf("Compressed sets %d \n", i);
            printf("size: %d \n", C->number_of_sets);
            printf("centroid coordinates: \n");
            int j;
            for (j = 0; j < DIM; j++) {
                double coord = C->sets[i].sum[j]/C->sets[i].number_of_points;

                printf("%lf ", coord);
            };
            printf("\n");
        };

        // calculate the time taken by the sequential version
        start = clock();
        // Call the hierchical_clustering function
        merge_compressedsets_and_miniclusters(C_copy, clusters, 0);
        end = clock();

        
        time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("Time taken by sequential version: %f\n", time_taken);

        // free C_copy
        
        free(C_copy);
        free(C);
        
        // Add assertions to verify the correctness of the function
        



        


        // print the results of the parallel version
        for (i = 0; i < C_copy->number_of_sets; i++) {
            printf("Compressed sets %d \n", i);
            printf("size: %d \n", C_copy->number_of_sets);
            printf("centroid coordinates: \n");
            int j;
            for (j = 0; j < DIM; j++) {
                double coord = C_copy->sets[i].sum[j]/C_copy->sets[i].number_of_points;

                printf("%lf ", coord);
            };
            printf("\n");
        };
    }else{
        free(C);
    }
    
}

int main(int argc, char** argv) {   
    int rank, size;

    //set the number of threads
    omp_set_num_threads(NUMBER_OF_THREADS);

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // catch exceptions
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
        printf("Error: MPI_Comm_rank\n");
        exit(1);
    }

    if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS) {
        printf("Error: MPI_Comm_size\n");
        exit(1);
    }


    test_hierchical_clustering_par(false, rank, size);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}