// Description: This file contains the parallel implementation of the BFR algorithm.
#include "../lib/BFR_parallel.h"

// TODO: there will some problem here I hear u bro

// initialize time variables
double local_start, local_finish, local_elapsed, elapsed;
double sec_clock_start, sec_clock_end, sec_clock_elapsed;


bool UpdateCentroidsMultithr(Cluster *clusters, int index, int number_of_clusters, int dimension) {
    bool flag_error = false;

    // the function has the same structure of the serial version but in multithreaded version
    int i, j;
    // printf("New coords:");
    # pragma omp parallel for shared(clusters, index, number_of_clusters, dimension) // pragmaa
    for (j = 0; j < dimension; j++) {
        clusters[index].centroid.coords[j] = clusters[index].sum[j] / clusters[index].size;
        // printf(" %lf", clusters[index].centroid.coords[j]);    
    }
    // printf("\n");

    flag_error = true;
    return flag_error;
}

Cluster *initClustersWithCentroids(Point *data_buffer, int size, int number_of_clusters, int dimension) {
    // the function has the same structure of the serial version but in multithread version and with some adjustment
    
    int i, j, random_index;
    // create a set of clusters number_of_clusters
    Cluster *clusters = (Cluster *)malloc(number_of_clusters * sizeof(Cluster));

    //set the indexes of the clusters
    for(i = 0; i < number_of_clusters; i++) {
        clusters[i].index = i;
        clusters[i].size = 1;
    }

    // take a random point from the data buffer and make it the centroid of the cluster
    srand(time(NULL));
    // random_index = rand() % size;
    random_index = 0;
    clusters[0].centroid = data_buffer[random_index];

    // Point centroids[number_of_clusters];
    // centroids[0] = data_buffer[random_index];

    double max_distance = 0., min_distance = DBL_MAX;
    int index_of_max = 0, index_of_min = 0;
    for(j = 0; j < size; j++){
        double current_distance = 0.;
        int k = 0;
        current_distance += distance((Pointer) & (clusters[0].centroid), (Pointer) & (data_buffer[j]));
        if (current_distance < min_distance && current_distance != 0.)
        {
            min_distance = current_distance;
            index_of_min = j;
        }
    }

    # pragma omp parallel for shared(clusters, data_buffer, random_index, index_of_min) // pragmaa
    for (j = 0; j < dimension; j++) {
        clusters[0].sum[j] = data_buffer[random_index].coords[j] + data_buffer[index_of_min].coords[j];
        clusters[0].sum_squares[j] =  pow(data_buffer[random_index].coords[j], 2) + pow(data_buffer[index_of_min].coords[j], 2);
    }
    clusters[0].size = 2;

    if(DEBUG) printf("Indexes of points chosen for cluster 0: %d %d.\n", random_index, index_of_min);
    if(DEBUG) printf("Choosing K-1 centroids.\n");

    // update the centroids of the clusters
    UpdateCentroidsMultithr(clusters, 0, number_of_clusters, dimension);
    

    // choose the other number_of_clusters-1 centroids seeking the farthest point from the previous centroids
    for (i = 1; i < number_of_clusters; i++) {
        int farthest_point_index = 0;
        double farthest_distance = 0.;
        double min_distance = DBL_MAX;
        int index_of_min = 0;
        int j;
        // multithread the for loop to find the farthest point from the previous centroids
        # pragma omp parallel for shared(clusters, data_buffer, farthest_point_index, farthest_distance) // pragmaa
        for (j = 0; j < size; j++) {
            double current_distance = 0;
            int k;
            // # pragma omp parallel for shared(clusters, data_buffer, farthest_point_index, farthest_distance, current_distance)
            for (k = 0; k < i; k++) {
                current_distance += distance((Pointer) &(clusters[k].centroid), (Pointer) & data_buffer[j]);
            }
            // critical section to update the farthest point
            # pragma omp critical // pragmaa
            if (current_distance > farthest_distance) {
                farthest_distance = current_distance;
                farthest_point_index = j;
            }
        }

        // set the farthest point as the centroid of the cluster
        clusters[i].centroid = data_buffer[farthest_point_index];
        clusters[i].size = 1;
        # pragma omp parallel for shared(clusters, data_buffer, min_distance, index_of_min) // pragmaa
        for(j = 0; j < size; j++){
            double current_distance = 0.;
            int k = 0;
            current_distance += distance((Pointer) & (clusters[i].centroid), (Pointer) & (data_buffer[j]));
            
            # pragma omp critical // pragmaa
            if (current_distance < min_distance && current_distance != 0.)
            {
                min_distance = current_distance;
                index_of_min = j;
            }
        }

        // multithread the for loop to update the sum and sum_square of the cluster
        # pragma omp parallel for shared(clusters, data_buffer, farthest_point_index, index_of_min) // pragmaa
        for(j = 0; j < dimension; j++){
            clusters[i].sum[j] = data_buffer[farthest_point_index].coords[j] + data_buffer[index_of_min].coords[j];
            clusters[i].sum_squares[j] = pow(data_buffer[farthest_point_index].coords[j], 2) + pow(data_buffer[index_of_min].coords[j], 2);
        }
        clusters[i].size = 2;

        // update the centroids of the clusters in multithreaded version
        UpdateCentroidsMultithr(clusters, i, number_of_clusters, dimension);
        if(DEBUG) printf("Indexes of points and relative distances chosen for cluster %d: %d - %lf, %d - %lf.\n", i, farthest_point_index, farthest_distance, index_of_min, min_distance);
    }
    if(DEBUG){
        int j = 0;
        for(; j<K; j++){
            printf("Centroid %d: ", j);

            int i = 0;
            for(; i<M; i++){
                printf("%lf ", clusters[j].centroid.coords[i]);
            }
            printf("\n");
        }
    }
    return clusters;
}

CompressedSet *merge_cset(CompressedSet *c1, CompressedSet *c2) {
    // allocate memory for the new compressed set
    CompressedSet * new_cset = (CompressedSet *)malloc(sizeof(CompressedSet));


    // merge the two compressed sets 
    new_cset -> number_of_points = c1 -> number_of_points + c2 -> number_of_points;
    
    int i;
    # pragma omp parallel for shared(new_cset, c1, c2)
    for ( i = 0; i < M; i++){
        new_cset -> sum[i] = c1 -> sum[i] + c2 -> sum[i];
        new_cset -> sum_square[i] = c1 -> sum_square[i] + c2 -> sum_square[i];
    }

    if (DEBUG){
        printf("created new cset with %d points\n", new_cset->number_of_points);
    }

    return new_cset;
}

void add_cset_to_compressed_sets(CompressedSets *C, CompressedSet * c) {
    // compressedSets -> number_of_sets += 1;

    // compressedSets -> sets = (CompressedSet *)realloc(compressedSets -> sets, compressedSets -> number_of_sets * sizeof(CompressedSet));

    // CompressedSet temp = *c;

    // compressedSets -> sets[compressedSets -> number_of_sets - 1] = temp;

    // free(c);
    (*C).number_of_sets += 1;
    (*C).sets = realloc(C->sets, C->number_of_sets * sizeof(CompressedSet));
    if (C->sets == NULL){
        perror("Error: could not allocate memory\n");
        exit(1);
    }
    (*C).sets[C->number_of_sets - 1].number_of_points = c->number_of_points;
    int j;
    // printf("\nsum   sumsquare\n");
    for (j=0; j<M; j++){
        (*C).sets[C->number_of_sets - 1].sum[j] = c->sum[j];
        (*C).sets[C->number_of_sets - 1].sum_square[j] = c->sum_square[j];
        // printf("%lf %lf\n", (*C).sets[C->number_of_sets - 1].sum[j], (*C).sets[C->number_of_sets - 1].sum_square[j]);
    }
}

// void hierachical_clustering_thr(CompressedSets * C){
//     // TODO: implement this function
//     // hierchieal clustering of the compressed sets

//     bool stop_criteria = false;
//     bool changes = false;

//     float **distance_matrix =  malloc (sizeof(float *) * C->number_of_sets);
//     int i;
//     for (i = 0; i < C->number_of_sets; i++) {
//         distance_matrix[i] = malloc (sizeof(float) * C->number_of_sets);
//     }

//     while(!stop_criteria){
//         // 1. compute the distance matrix
//         if (changes) {
//             int i, j;
//             # pragma omp parallel for shared(distance_matrix, C) private(j)
//             for (i = 0; i < C->number_of_sets; i++) {
//                 for (j = 0; j < C->number_of_sets; j++) {
//                     // TODO: implement distance function
//                     distance_matrix[i][j] = distance((Pointer) & (C->sets[i]), (Pointer) & (C->sets[j]));
//                     distance_matrix[j][i] = distance_matrix[i][j];
//                 }
//             }
//         }

//         // 3. find the minimum distance in the distance matrix
//         float min_distance = FLT_MAX;
//         int min_i, min_j;

//         int i, j;
//         # pragma omp parallel for shared(distance_matrix, min_distance, min_i, min_j) private(j)
//         for (i = 0; i < C->number_of_sets; i++) {
//             for (j = 0; j < C->number_of_sets; j++) {
//                 # pragma omp critical
//                 if (distance_matrix[i][j] < min_distance) {
//                     min_distance = distance_matrix[i][j];
//                     min_i = i;
//                     min_j = j;
//                 }
//             }
//         }

//         if (min_distance == FLT_MAX) {
//             stop_criteria = true;
//             break;
//         }

//         // 4. merge the two compressed set with the minimum distance
//         CompressedSet * new_cset = merge_cset(&(C->sets[min_i]), &(C->sets[min_j]));

//         // 5. check the tightness of the new cluster
//         if (tightness_evaluation_cset(*new_cset)) {
//             // 6. add the new cluster to the compressed sets
//             add_cset_to_compressed_sets(C, new_cset);
//             // 7. remove the two clusters from the compressed sets
//             remove_cset_from_compressed_sets(C, &(C->sets[min_i]), min_i);
//             remove_cset_from_compressed_sets(C, &(C->sets[min_j]), min_j);
//             changes = true;
//         } else {
//             // 8. set the distance between the two clusters to infinity
//             distance_matrix[min_i][min_j] = FLT_MAX;
//             distance_matrix[min_j][min_i] = FLT_MAX;
//             changes = false;

//             free(new_cset);
//         }

//     }
// }

bool primary_compression_criteria(Cluster *clusters, Cluster *clusters_copy, Point p) {
    // function copied from the serial version
    /*
    * Description:
    * Use the Mahalanobis distance to determine whether or not a point 
    * can be added directly to a cluster.
    *
    * Algorithm:
    *   1. check if point is close enough to some centroid, by using mahalanobis radius as a confidence measure
    *   2. if it is, add it to the cluster
    *   3. if not check the secondary compression criteria
    *
    * Parameters:
    *   - clusters: array of clusters
    *   - p: point to check
    *
    * Returns:
    *   - true if point is added to cluster
    *   - false if point is not added to cluster
    * IMPORTANT: the two primary compression criteria are separate and mutually exclusive!
    */

    int i = 0, min_cluster;
    double min_distance = DBL_MAX, current_distance;
    // if(DEBUG) {
    //     printf("      Clusters BEFORE.\n");
    //     print_clusters(clusters);
    //     print_clusters(clusters_copy);
    // }

    // if(DEBUG) printf("      Checking mahalanobis distance for point.\n");
    # pragma omp parallel for shared(min_distance, min_cluster, clusters, clusters_copy, p) private(current_distance)// pragmaa
    for (i = 0; i < K; i++){
        current_distance = mahalanobis_distance(clusters_copy[i], p);
        // if(DEBUG) printf("      Current distance: %lf.\n", current_distance);
        # pragma omp critical // pragmaa
        if (current_distance < min_distance) {
            min_distance = current_distance;
            min_cluster = i;
        }
    }

    if (min_distance < T){
        // if(DEBUG) printf("      Minimal distance is %lf and is under threshold, updating cluster %d with point.\n", min_distance, min_cluster);
        if(DEBUG) printf("  Adding point %lf %lf to cluster %d.\n", p.coords[0], p.coords[1], min_cluster);
        //add point to cluster whose distance from the centroid is minimal, if distance != 0. (the point is not the starting centroid)
        if(min_distance != 0.) update_cluster(&clusters[min_cluster], p); //ALERT: Function implemented in the file "bfr_structure.c", since has costant complexity it remains unvariated
        // if(DEBUG) printf("      Cluster %d updated.\n", min_cluster);
        return true;
    }
    
    // if(DEBUG) printf("      No cluster fits the criteria, minimal distance is %lf.\n", min_distance);

    // if(DEBUG) printf("      Clusters AFTER.\n");
    // print_clusters(clusters);

    return false;
}


Cluster *cluster_retained_set_thrs(RetainedSet *R, int *k, int rank, int size, MPI_Datatype PointType){

    if(DEBUG && rank == MASTER) printf("          Initializing standard kmeans data.\n");
    Cluster * miniclusters = init_cluster((*k));

    if(DEBUG && rank == MASTER) printf("          Broadcasting number of retainedSet points: %d.\n", (*R).number_of_points);
    MPI_Bcast(&((*R).number_of_points), 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    if(rank != MASTER) R->points = (Point *)malloc(R->number_of_points * sizeof(Point)); 
    MPI_Bcast(R->points, R->number_of_points, PointType, MASTER, MPI_COMM_WORLD);
    // TODO: discuss a correct limit for not running standard kmeans
    // as of now, do not run kmeans if the number of points is < k
    // we may want to run kmeans when we have more than k*constant number of points
    if((*R).number_of_points < (*k)){
        if(DEBUG && rank == MASTER) printf("          Retained set has less points (%d) than clusters(%d). Returning empty miniclusters.\n", (*R).number_of_points, (*k));
                return miniclusters;
    }

    kmeans_config config = init_kmeans_config((*k), R, true, rank, size);
    if(DEBUG && rank == MASTER) printf("          Executing standard kmeans.\n");

    // if(rank == MASTER){
        kmeans_result result = kmeans(&config);

        if(DEBUG && rank == MASTER) printf("          Iteration count: %d\n", config.total_iterations);
        if(DEBUG && rank == MASTER) printf("          Num objs: %d\n", config.num_objs);
        if(DEBUG && rank == MASTER) printf("          Transferring kmeans cluster data to miniclusters.\n");
        
        int i;
        for (i = 0; i < config.num_objs; i++){
            Point *pt = (Point *)(config.objs[i]);

            update_cluster(&miniclusters[config.clusters[i]], *pt);
        }

        // create new correct retained set with only the points left alone in their clusters
        RetainedSet new_R = init_retained_set();
        int * tightness_flag;
        tightness_flag = calloc((*k), sizeof(int));
        for (i = 0; i < config.num_objs; i++){
            Point *pt = (Point *)(config.objs[i]);
            // TODO: use a different measure to determine a minicluster's tightness
            int index = config.clusters[i];

            if (!tightness_evaluation_cluster(miniclusters[index], tightness_flag, index)){
                add_point_to_retained_set(&new_R, *pt);
            }
            // else {
            //     if(DEBUG && rank == MASTER) printf("Point not added to retained set: %g\t%g\t%d\n", pt->coords[0], pt->coords[1], config.clusters[i]);
            // }
        }

        // if(DEBUG && rank == MASTER){
        //     printf("          Old retained set:\n");
        //     print_retainedset(*R);
        // }

        // TODO: discuss whether this is correct or not
        // free old retained set and replace with new one
        free((*R).points);
        (*R).points = new_R.points;
        (*R).number_of_points = new_R.number_of_points;

        if(DEBUG && rank == MASTER){
            printf("          New retained set:\n");
            print_retainedset(*R);
        }

        if(DEBUG && rank == MASTER) printf("          Freeing previously allocated data for standard kmeans.\n");

        //update miniclusters, retain only with tightness_flag = 2
        (*k) = update_miniclusters(&miniclusters, tightness_flag, (*k));

        // free the kmeans' config data
        if (config.objs != NULL){
            free(config.objs);
        }
        if (config.centers != NULL){
            free(config.centers);
        }
        if (config.clusters != NULL){
            free(config.clusters);
        }
        if (tightness_flag != NULL){
            // free tightness flag
            free(tightness_flag);
        }

        // update miniclusters' centroids
        update_centroids(&miniclusters, (*k));
    // }
    // else{
    //     kmeans(&config);
    //     (*k) = 0;
    //     free(config.objs);
    //     free(config.centers);
    //     free(config.clusters);
    // }

    // If MASTER, will be correctly filled. If not, it's an empty array
    return miniclusters;
}

void secondary_compression_criteria(Cluster *clusters, RetainedSet *retainedSet, CompressedSets *compressedSets, int rank, int size, MPI_Datatype PointType) {
    // TODO: implement the function
    /*
    * Description:
    * find a way to aggregate the retained set into miniclusters,
    * and miniclusters with more than one element can be summarized in the compressed set.
    * Outliers are kept in the retained set.
    *
    * Algorithm:
    *   1. cluster retained set R with classical K-Means, creating k2 clusters
    *   2. clusters that have a tightness measure above a certain threshold are added to the CompressedSet C
    *   3. outliers are kept in the RetainedSet R
    *   4. try aggregating compressed sets using statistics and hierchieal clustering
    *
    * Parameters:
    *   - R: retained set
    *   - clusters: array of clusters
    *   - C: compressed sets
    *
    * Returns:
    *   - void
    */

    // 1. cluster retained set R with classical K-Means, creating k2 clusters, also keep in R the outlayers
    // TODO: implement the function K-Means with open mp
    // if (DEBUG){
    //     // printf("I'm node %d and I'm waiting to start kmeans\n", rank);
    //     // MPI_Barrier(MPI_COMM_WORLD);
    //     // printf("start of kmeans node %d\n\n\n", rank);
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    int k2 = K;
    Cluster *k2_clusters = cluster_retained_set_thrs(retainedSet, &k2, rank, size, PointType);

    // TODO: consider parallelizing this
    add_miniclusters_to_compressedsets(compressedSets, k2_clusters, k2);
    // if (DEBUG){
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (rank == MASTER) {
    //         printf("\n%d: Printing old compressed sets BEFORE.\n", rank);
    //         print_compressedsets(*compressedSets);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    
    free(k2_clusters);
}

void UpdateRetainedSet(RetainedSet *R, RetainedSet *tempRetainedSet){
    int old_number_of_points = R->number_of_points;
    R->number_of_points += tempRetainedSet->number_of_points;
    R->points = realloc(R->points, R->number_of_points * sizeof(Point));
    if (R->points == NULL){
        perror("Error: could not allocate memory\n");
        exit(1);
    }

    int i;
    for (i = 0; i < tempRetainedSet->number_of_points; i++){
        R->points[old_number_of_points + i] = tempRetainedSet->points[i];
    }
}

bool read_point(Point * data_buffer, Point * p, long int size_of_data_buffer, int * offset){
    //ALERT: FUNCTION COPIED FROM SERIAL VERSION
    
    /*
    * Read a point from data buffer
    *
    * Algorithm:
    *   1. read M coordinates from data buffer
    *   2. if data buffer end is reached return false
    *   3. else return true
    *
    * Parameters:
    *   - data_buffer: buffer to read data from
    *   - p: point to read
    *
    * Returns:
    *   - true if point is read successfully
    *   - false if EOF is reached
    */

    // if(DEBUG) printf("  Reading point.\n");
    if (*offset >= size_of_data_buffer){
        return false;
    }

    //read M coordinates from data buffer
    int i = 0;
    for (i = 0; i < M; i++){
        p->coords[i] = (data_buffer[*offset]).coords[i];
    }
    (*offset)++;
    return true;
}

void StreamPoints(Cluster *clusters, Cluster *clusters_copy, CompressedSets *compressedSets, RetainedSet *retainedSet, Point *data_buffer, int size) {
    // perform primary compression criteria and secondary compression criteria

    // primary compression criteria
    // for each point in the data buffer
    // TODO: revise this part of the code
    Point p;
    int offset = 0;
    // if(DEBUG) printf("  Offset: %d\n", offset);
    // # pragma omp parallel for private(p) shared(data_buffer, clusters, retainedSet) //reduction(+:offset)
    while(read_point(data_buffer, &p, size, &offset)) {
        // if(DEBUG) printf("  Checking primary compression criteria for point %lf %lf.\n", p.coords[0], p.coords[1]);
        if (!primary_compression_criteria(clusters, clusters_copy, p)) {
            if(DEBUG) printf("  Adding point %lf %lf to RetainedSet.\n", p.coords[0], p.coords[1]);
            // this function is the same of the serial version, can be found in "bfr_structure.h"
            // if(DEBUG) printf("  Adding point to RetainedSet.\n");
            add_point_to_retained_set(retainedSet, p);
        }
    }
}


void PlotResults(Cluster *clusters, RetainedSet *retainedSet, CompressedSets *compressedSets) {
    print_clusters(clusters);
    print_compressedsets(*compressedSets);
    print_retainedset(*retainedSet);
    // TODO: consider printing the results in a file
}

void remove_cset_from_compressed_sets(CompressedSets *C, int index1, int index2) {
    int old_number_of_sets = C->number_of_sets;
    C->number_of_sets -= 2;
    // ALERT: c'era già una funzione per inizializzare i compressed sets, staticamente. Quindi visto che questi son dichiarati dinamicamente prevedo che delle free falliscano
    // printf("Merging %d and %d.\n", index1, index2);
    int i, counter = 0;
    CompressedSet * temp_sets = (CompressedSet *)malloc(C->number_of_sets * sizeof(CompressedSet));
    for (i = 0; i < old_number_of_sets; i++) {
        if (i != index1 && i != index2) {
            temp_sets[counter] = C->sets[i];
            counter++;
        }
    }

    C->sets = realloc(C->sets, C->number_of_sets * sizeof(CompressedSet));

    for (i = 0; i < C->number_of_sets; i++) {
        C->sets[i] = temp_sets[i];
    }

    free(temp_sets);
}

double dist_cset(CompressedSet c1, CompressedSet c2){
    // calculate the two centroids coordinates
    double * coord_c1 = malloc(M * sizeof(double));
    double * coord_c2 = malloc(M * sizeof(double));
    int i;

    // calculate centroid for c1
    // if (DEBUG) printf("coords c1\n");
    for (i = 0; i < M; i++) {
        coord_c1[i] = (double)c1.sum[i] / c1.number_of_points;
        // if(DEBUG) printf("coord: %lf\n", coord_c1[i]);
    }

    // calculate centroid for c2
    // if (DEBUG) printf("coords c2\n");
    for (i = 0; i < M; i++) {
        coord_c2[i] = (double)c2.sum[i] / c2.number_of_points;
        // if(DEBUG) printf("coord: %lf\n", coord_c1[i]);
    }

    
    //if (DEBUG) printf("checkpoint 1\n");
    // calculate the distance between the two centroids
    double distance = 0.;
    for (i = 0; i < M; i++) {
        //if (DEBUG) printf("checkpoint 1.3\n");
        distance += pow(coord_c1[i] - coord_c2[i], 2);
        //if (DEBUG) printf("checkpoint 1.5\n");
    }
    distance = sqrt(distance);
    if (DEBUG) printf("checkpoint 1.7 distance %lf\n", distance);
    // free the allocated memory
    free(coord_c1);
    free(coord_c2);

    //if (DEBUG) printf("checkpoint 2\n");

    return distance;
}

bool hierachical_clust_parallel(CompressedSets *C, int rank, int size){
    /*
    * Description:
    * hierchieal clustering of the compressed sets
    *
    * Algorithm:
    *   1. compute the distance matrix
    *   2. find the minimum distance in the distance matrix
    *   3. merge the two compressed set with the minimum distance
    *   4. check the tightness of the new cluster
    *   5. if the new cluster is tight enough add it to the compressed sets
    *   6. remove the two clusters from the compressed sets
    *   7. else set the distance between the two clusters to infinity
    *   8. repeat from step 1 until the stop criteria is reached
    *
    * Parameters:
    *   - C: compressed sets
    *
    * Returns:
    *   - void
    */
    // initialize the distance matrix
   
    double **distance_matrix =  malloc (sizeof(double *) * C->number_of_sets);
    int i;
    for (i = 0; i < C->number_of_sets; i++) {
        distance_matrix[i] = malloc (sizeof(double) * C->number_of_sets);
    }
    if (DEBUG)printf("node %d: distance matrix declared\n",rank );
    bool stop_criteria = false;
    bool changes = true;
    bool not_firstr =true;
    int iterations = 0;


    while(!stop_criteria){
        if (C->number_of_sets <= K3) {
            break;
        }
        int start, number_of_sets;

        // setup a new communicator
        MPI_Comm has_sets_to_parse;
        int color = (rank < C->number_of_sets) ? 1 : MPI_UNDEFINED;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &has_sets_to_parse);

        int new_comm_rank, new_comm_size = 0;
        if (rank < C->number_of_sets) {
            MPI_Comm_rank(has_sets_to_parse, &new_comm_rank);
            MPI_Comm_size(has_sets_to_parse, &new_comm_size);
        }

        // 1. compute the distance matrix
        if (changes) {
            // free the distance matrix
            if (not_firstr){
                for (i = 0; i < C->number_of_sets; i++) {
                    if (distance_matrix[i] != NULL) free(distance_matrix[i]);
                }
                if (distance_matrix != NULL)free(distance_matrix);
                not_firstr = true;
            }
            

            if (DEBUG)printf("Node %d erased the distance matrix\n", rank);

            distance_matrix =  malloc (sizeof(double *) * C->number_of_sets);

            for (i = 0; i < C->number_of_sets; i++) {
                distance_matrix[i] = malloc (sizeof(double) * C->number_of_sets);
                int p;
                for (p=0; p< C->number_of_sets;p++){
                    distance_matrix[i][p]=DBL_MAX_HC;
                }
            }
            if (DEBUG)printf("Node %d reinitiallized the distance matrix\n", rank);

            // if (DEBUG) MPI_Barrier(MPI_COMM_WORLD);
            if (size < C->number_of_sets ){
                // case where at least one process has to compute the distance vector more than once
                number_of_sets = C->number_of_sets / size;
                int original = number_of_sets;

                if (rank == size - 1){
                    if (number_of_sets * size < C->number_of_sets){
                        number_of_sets = number_of_sets + C->number_of_sets % size;
                    }else{
                        number_of_sets = C->number_of_sets - (size - 1) * number_of_sets;
                    }
                }

                start = rank * original; 

                if (DEBUG ){
                    if (rank == MASTER) printf("case 1\n");
                    printf("node %d will start at sets %d and calculate %d sets\n", rank, start, number_of_sets);
                }
            }else if (size >= C->number_of_sets){
                // case where each process has to compute the distance vector only once
                if (rank >= C->number_of_sets){
                    stop_criteria = true;
                    break;
                }


                number_of_sets = 1;
                start = rank;

                
                if (DEBUG ){
                    if (rank == MASTER) printf("case 2\n");
                    printf("node %d will start at sets %d and calculate %d sets\n", rank, start, number_of_sets);
                }
            }

            // if (DEBUG){
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     if (rank == MASTER) print_compressedsets(*C);
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     printf("node %d is starting to fill the distance vector\n", rank);
            //     printf("node %d is starting from %d and has %d sets to parse\n", rank, start, number_of_sets);
            //     // MPI_Barrier(MPI_COMM_WORLD);
            // }

            if (DEBUG) printf("%d: C size is %d\n", rank, C->number_of_sets);
            // for each process compute the distance vectors for the sets assigned
            int i, j;
            // # pragma omp parallel for shared(distance_matrix, C, start, number_of_sets) private(j)
            for (i = start; i < start + number_of_sets; i++) {
                if (DEBUG)printf("\n%d: distance matrix i offset %d\n", rank, i);
                for (j = 0; j < C->number_of_sets; j++) {
                    if (DEBUG) printf("%d: index i %d, j %d\n", rank, i, j);
                    double dist = dist_cset(C->sets[i],  C->sets[j]);
                    if (DEBUG) printf("%lf \n", dist);
                    if (dist < 0) dist = 0.;
                    if(dist > DBL_MAX_HC) dist = DBL_MAX_HC;
                    distance_matrix[i][j] = dist;
                }
            }



            if (DEBUG){
                printf("distance vectors calculated in node %d \n", rank);
                // MPI_Barrier(MPI_COMM_WORLD);
            }

            if (rank == MASTER) {
                // case where the master has to compute the distance vector for the remaining sets
                i = number_of_sets;
                for (; i < C->number_of_sets; i++) {
                    // the rank of the process that should have computed the distance vector for the set i: i / number_of_points - 1 
                    // if the rank is the master the distance vector is already computed
                    // example: if the number of sets is 10 and the number of processes is 4, the first 3 processes will compute the distance vector for 3 sets, the last process will compute the distance vector for 1 set
                    // sets i would have been computed by the process i / number_of_points - 1, in the example the set 4 would have been computed by the process 1
                    int expected_rank = floor(i / number_of_sets);
                    if (expected_rank>=size )expected_rank = size -1;
                    if (DEBUG) printf("master is waiting of receiving the distance vector %d from process %d\n ", i , expected_rank);
                    MPI_Recv(distance_matrix[i], C->number_of_sets, MPI_DOUBLE, expected_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (DEBUG){
                        printf("distance matrix received by node %d \n", expected_rank);

                        int oi=0;
                        for(;oi<C->number_of_sets;oi++){
                            printf("%lf ",distance_matrix[i][oi] );
                        }
                        printf("\n");
                    }
                }

                if(DEBUG) printf("received all matrix\n");



                //fill the upper part of the distance matrix
                //ALERT: REVISE THIS PART
                // for (i = 0; i < C->number_of_sets; i++) {
                //     for (j = 0; j < i; j++) {
                //         distance_matrix[j][i] = distance_matrix[i][j];
                //     }
                // }



                // if (DEBUG){
                //     printf("distance matrix in master\n\n");
                //     for (i = 0; i < C->number_of_sets;i++){
                //             for (j = 0; j < C->number_of_sets;j++){
                //             printf("%lf ", distance_matrix[i][j]);
                //             }
                //             printf("\n");
                //     }
                // }

                // //Broadcast the distance matrix to the other processes
                // for(i = 1; i < size; i++) {
                //     MPI_Send(distance_matrix[i], number_of_sets, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                // }
            } else {
                for(i = start; i < start + number_of_sets; i++) {
                    int size_of_dist_vector = C->number_of_sets-i;
                    if (DEBUG)printf("%d: sending distance vector of size %d\n", rank, C->number_of_sets);
                    MPI_Send(distance_matrix[i], C->number_of_sets, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
                }

                //if (DEBUG)printf("node %d sent distance vector\n", rank);

                // //Receive the distance matrix from the master
                // for (i = 0; i < C->number_of_sets; i++) {
                //     MPI_Recv(distance_matrix[i], C->number_of_sets, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // }

                // if (DEBUG) printf("node %d receveid the distance from the master");
            }  
            // if (DEBUG) MPI_Barrier(MPI_COMM_WORLD);
            
            

            //if(DEBUG)printf("checkpoint 1\n");
            for (i = 0; i < C->number_of_sets;i++){
                MPI_Bcast(distance_matrix[i], C->number_of_sets, MPI_DOUBLE, MASTER, has_sets_to_parse);
            }     

            if (DEBUG && rank == 1){
                for (i = 0; i < C->number_of_sets;i++){
                     for (j = 0; j < C->number_of_sets;j++){
                        printf("%lf ", distance_matrix[i][j]);
                     }
                     printf("\n");
                }
            }
            // if (DEBUG)MPI_Barrier(MPI_COMM_WORLD);
        }

        // 3. find the minimum distance in the distance matrix

        if (size < C->number_of_sets ){
            // case where at least one process has to compute the distance vector more than once
            number_of_sets = C->number_of_sets / size;

            if (rank == size - 1){
                if (number_of_sets * size < C->number_of_sets){
                    number_of_sets = number_of_sets + C->number_of_sets % size;
                }else{
                    number_of_sets = C->number_of_sets - (size - 1) * number_of_sets;
                }
            }

            start = rank * number_of_sets;
        }else if (size >= C->number_of_sets){
            // case where each process has to compute the distance vector only once
            if (rank >= C->number_of_sets){
                stop_criteria = true;
                break;
            }

            number_of_sets = 1;
            start = rank;
        }

        double min_distance = DBL_MAX_HC;
        int min_i, min_j;

        int i, j;
        # pragma omp parallel for shared(distance_matrix, min_distance, min_i, min_j) private(j)
        for (i = start; i < start + number_of_sets; i++) {
            for (j = i + 1; j < C->number_of_sets; j++) {
                # pragma omp critical
                if (distance_matrix[i][j] < min_distance) {
                    min_distance = distance_matrix[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        // the master process gets the minimum distance from the other processes
        if (rank == MASTER) {
            double temp_min_distance;
            int temp_min_i, temp_min_j;

            if (new_comm_size != 0) size = new_comm_size;
            for (i = 1; i < size; i++) {
                MPI_Recv(&temp_min_distance, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&temp_min_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&temp_min_j, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (temp_min_distance < min_distance) {
                    min_distance = temp_min_distance;
                    min_i = temp_min_i;
                    min_j = temp_min_j;
                }
            }
        } else {
            MPI_Send(&min_distance, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
            MPI_Send(&min_i, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            MPI_Send(&min_j, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        }

        MPI_Bcast(&min_distance, 1, MPI_DOUBLE, MASTER,  has_sets_to_parse);
        MPI_Bcast(&min_i, 1, MPI_INT, MASTER,  has_sets_to_parse);
        MPI_Bcast(&min_j, 1, MPI_INT, MASTER,  has_sets_to_parse);

        if(DEBUG){
            // MPI_Barrier(MPI_COMM_WORLD);
            printf("minimum distance is %lf between cset %d and cset %d\n\n", min_distance, min_i, min_j);
            // MPI_Barrier(MPI_COMM_WORLD);
        }
        
        if (min_distance >= DBL_MAX_HC - 100. || iterations > UPPER_BOUND_ITERATIONS) {
            stop_criteria = true;
            break;
        }
        // 4. merge the two compressed set with the minimum distance in the master
        // if (rank == MASTER){
        //     new_cset = merge_cset(&(C->sets[min_i]), &(C->sets[min_j]));

        //     // 5. check the tightness of the new cluster
        //     if (tightness_evaluation_cset(*new_cset)) {
        //         // 6. add the new cluster to the compressed sets
        //         CompressedSet * temp = (CompressedSet *)malloc(sizeof(CompressedSet));
        //         *temp = *new_cset;

        //         add_cset_to_compressed_sets(C, temp);
        //         // 7. remove the two clusters from the compressed sets
        //         remove_cset_from_compressed_sets(C, &C->sets[min_i], min_i);
        //         remove_cset_from_compressed_sets(C, &C->sets[min_j], min_j);
        //         changes = true;

        //         MPI_Bcast(&(new_cset->number_of_points), 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        //         MPI_Bcast(new_cset->sum, M, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
        //         MPI_Bcast(new_cset->sum_square, M, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

        //         free(new_cset);
        //     } else {
        //         // 8. set the distance between the two clusters to infinity
        //         distance_matrix[min_i][min_j] = FLT_MAX;
        //         distance_matrix[min_j][min_i] = FLT_MAX;
        //         changes = false;

        //         free(new_cset);
        //     }

        //     //Broadcast the changes to the other processes
        //     MPI_Bcast(&changes, 1, MPI_C_BOOL, MASTER, MPI_COMM_WORLD);
        //     MPI_Bcast(&min_i, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        //     MPI_Bcast(&min_j, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
        // }else{
        //     MPI_Recv(&changes, 1, MPI_C_BOOL, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     MPI_Recv(&min_i, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     MPI_Recv(&min_j, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //      if (changes) {
        //         CompressedSet * new_cset = (CompressedSet *)malloc(sizeof(CompressedSet));
        //         int number_of_points;
        //         float sum[M], sum_square[M];

        //         MPI_Recv(&number_of_points, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //         MPI_Recv(sum, M, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //         MPI_Recv(sum_square, M, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //         new_cset -> number_of_points = number_of_points;
        //         int i;
        //         # pragma omp parallel for shared(new_cset, sum, sum_square)
        //         for (i = 0; i < M; i++){
        //             new_cset -> sum[i] = sum[i];
        //             new_cset -> sum_square[i] = sum_square[i];
        //         }

        //         add_cset_to_compressed_sets(C, new_cset);
        //         remove_cset_from_compressed_sets(C, &C->sets[min_i], min_i);
        //         remove_cset_from_compressed_sets(C, &C->sets[min_j], min_j);

        //     }else{
        //         distance_matrix[min_i][min_j] = FLT_MAX;
        //         distance_matrix[min_j][min_i] = FLT_MAX;
        //     }
        // }
        CompressedSet * new_cset = (CompressedSet *)malloc(sizeof(CompressedSet));
        new_cset->number_of_points = 0;
        if (rank == MASTER){
            new_cset = merge_cset(&(C->sets[min_i]), &(C->sets[min_j]));
        }

        MPI_Bcast(&(new_cset->number_of_points), 1, MPI_INT, MASTER, has_sets_to_parse);
        MPI_Bcast(&(new_cset->sum), M, MPI_DOUBLE, MASTER, has_sets_to_parse);
        MPI_Bcast(&(new_cset->sum_square), M, MPI_DOUBLE, MASTER, has_sets_to_parse);

        MPI_Comm_free(&has_sets_to_parse);
        

        if (DEBUG){
            // MPI_Barrier(MPI_COMM_WORLD);
            // for (i = 0; i<M; i++){
            //     printf("node %d cset coord sum %lf coord sum_squared %lf\n",rank, new_cset->sum[i], new_cset->sum_square[i]);
            // }
            // MPI_Barrier(MPI_COMM_WORLD);
        }

        // 5. check the tightness of the new cluster
        if (tightness_evaluation_cset(*new_cset)) {
            // 6. add the new cluster to the compressed sets
            CompressedSet * temp = (CompressedSet *)malloc(sizeof(CompressedSet));
            *temp = *new_cset;

            if (DEBUG){
                // MPI_Barrier(MPI_COMM_WORLD);
                // printf("cset successfully added\n");
                // MPI_Barrier(MPI_COMM_WORLD);
            }
            // 7. remove the two clusters from the compressed sets
            remove_cset_from_compressed_sets(C, min_i, min_j);
            if (DEBUG){
                printf("cset successfully removed\n");
            }
            // remove_cset_from_compressed_sets(C, &C->sets[min_j], min_j);
            // if (DEBUG){
            //     printf("cset successfully removed\n");
            // }
            // if (DEBUG){printf("Address of new compressedsets is %p.\n", C);};

            add_cset_to_compressed_sets(C, temp);

            if (DEBUG){
                // MPI_Barrier(MPI_COMM_WORLD);
                printf("cset successfully added\n");
                // MPI_Barrier(MPI_COMM_WORLD);
            }
            changes = true;

        }else{
            distance_matrix[min_i][min_j] = DBL_MAX_HC;
            distance_matrix[min_j][min_i] = DBL_MAX_HC;
        }

        free(new_cset);

        if (C->number_of_sets < K3){
            stop_criteria =true;
            break;
        }

        if (DEBUG){
                // MPI_Barrier(MPI_COMM_WORLD);
                printf("Iteration %d performed, we will continue\n", iterations);
                // MPI_Barrier(MPI_COMM_WORLD);
            }
        iterations++;
    }
    

    // synchronize the processes
    // if ( DEBUG ) {
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (rank == MASTER) print_compressedsets(*C);
    //     printf("Done hierchical clustering");
    // }

    // free the distance matrix
    for (i = 0; i < C->number_of_sets; i++) {
        free(distance_matrix[i]);
    }
    free(distance_matrix);

    return true;
}

void defineArrayclusterType(MPI_Datatype *ArrayclusterType, MPI_Datatype PointType){
    /*
    typedef struct {
        Point centroid;
        int size;
        double sum[M];
        double sum_squares[M];
        int index;
    } Cluster;
    */
    int lengths[5] = {1, 1, M, M, 1};
    MPI_Aint displacements[5];

    Cluster dummy_cluster;
    MPI_Aint base_address;

    MPI_Get_address(&dummy_cluster, &base_address);
    MPI_Get_address(&dummy_cluster.centroid, &displacements[0]);
    MPI_Get_address(&dummy_cluster.size, &displacements[1]);
    MPI_Get_address(&dummy_cluster.sum[0], &displacements[2]);
    MPI_Get_address(&dummy_cluster.sum_squares[0], &displacements[3]);
    MPI_Get_address(&dummy_cluster.index, &displacements[4]);

    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);
    displacements[3] = MPI_Aint_diff(displacements[3], base_address);
    displacements[4] = MPI_Aint_diff(displacements[4], base_address);
 

    MPI_Datatype types[5] = {PointType, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

    MPI_Type_create_struct(5, lengths, displacements, types, ArrayclusterType);
    MPI_Type_commit(ArrayclusterType);

}

    
void definePointType(MPI_Datatype *PointType) {
    int lengths[2] = {M, 1};
    MPI_Aint displacements[2];

    Point dummy_point;
    MPI_Aint base_address;

    MPI_Get_address(&dummy_point, &base_address);
    MPI_Get_address(&dummy_point.coords[0], &displacements[0]);
    MPI_Get_address(&dummy_point.cluster, &displacements[1]);

    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
 

    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};

    MPI_Type_create_struct(2, lengths, displacements, types, PointType);
    MPI_Type_commit(PointType);
}

void copy_clusters(Cluster *clusters, Cluster *clusters_copy) {
    /*
        typedef struct {
            Point centroid;
            int size;
            double sum[M];
            double sum_squares[M];
            int index;
        } Cluster;
    */
    int i, d;
    # pragma omp parallel for private(d)
    for (i = 0; i < K; i++){
        clusters_copy[i].centroid = clusters[i].centroid;
        clusters_copy[i].size = clusters[i].size;
        clusters_copy[i].index = clusters[i].index;
        for (d = 0; d < M; d++){
            clusters_copy[i].sum[d] = clusters[i].sum[d];
            clusters_copy[i].sum_squares[d] = clusters[i].sum_squares[d];
        }
    }
}

int main(int argc, char **argv) {

    if (argc != 2)
    {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    int rank, size;

    // set the number of threads
    omp_set_num_threads(NUMBER_OF_THREADS);

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // catch exceptions
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
    {
        printf("Error: MPI_Comm_rank\n");
        exit(1);
    }

    if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS)
    {
        printf("Error: MPI_Comm_size\n");
        exit(1);
    }

    if (DEBUG)
    {
        printf("Starting Parallel BFR.\n");
        printf("Hello from %d of %d\n", rank, size);
    }

    // initialize the clusters the retained set and the compressed sets
    Cluster *clusters = (Cluster *)malloc(K * sizeof(Cluster));
    RetainedSet retainedSet_normal = (init_retained_set());
    CompressedSets compressedSets_normal = (init_compressed_sets());
    RetainedSet *retainedSet = &retainedSet_normal;
    // CompressedSets *compressedSets = &compressedSets_normal;
    CompressedSets *compressedSets = (CompressedSets *)malloc(sizeof(CompressedSets));
    compressedSets->sets = NULL;
    compressedSets->number_of_sets = 0;

    // keep a cluster copy to keep track of each round's additions
    Cluster *clusters_copy = (Cluster *)malloc(K * sizeof(Cluster));

    // initialize the data streams from the input files
    data_streamer stream_cursor;
    if (rank == MASTER)
    {
        stream_cursor = data_streamer_Init(argv[1], "r");
    }

    // data buffer used by each node
    Point data_buffer[DATA_BUFFER_SIZE];

    // data buffer used by MASTER to fill other node's buffers
    Point data_buffer_master[MAX_SIZE_OF_BUFFER];

    int offset;
    int round = 0;

    // use derived datatype to send the clusters to all processes
    MPI_Datatype ArrayclusterType; // array of cluster
    // MPI_Datatype RetainedSetType;
    MPI_Datatype PointType;

    definePointType(&PointType);
    defineArrayclusterType(&ArrayclusterType, PointType);
    // defineRetainedSetType(&RetainedSetType);

    int stop_criteria = 0;
    long node_data_buffer_size = 0, size_of_data_buffer_copy = 0;

    //initialization finished start measuring time
    MPI_Barrier(MPI_COMM_WORLD);
    local_start = MPI_Wtime();
    do
    {
        // update the offset for each node
        offset = rank * DATA_BUFFER_SIZE + round * size * DATA_BUFFER_SIZE;
        if (DEBUG_TIME){
            sec_clock_start = MPI_Wtime();
        }
        if (rank == MASTER)
        {
            long size_of_data_buffer = 0;
            if (DEBUG)
                printf("Loading buffer.\n");
            if (!load_data_buffer(stream_cursor, data_buffer_master, MAX_SIZE_OF_BUFFER, &size_of_data_buffer))
            {
                stop_criteria = 0;
                size_of_data_buffer_copy = size_of_data_buffer;
            }
            else
            {
                stop_criteria = 1;
            }

            if (DEBUG)
                printf("Stop criteria is %d.\n", stop_criteria);
            if (!stop_criteria)
            {
                if (DEBUG)
                    printf("Computing points per node.\n");
                // pass the correct number of points to each node
                long points_per_node = size_of_data_buffer / size;
                if (points_per_node < MIN_DATA_BUFFER_SIZE_LAST_ROUND)
                {
                    points_per_node = MIN_DATA_BUFFER_SIZE_LAST_ROUND;
                }
                // if (points_per_node > DATA_BUFFER_SIZE) {
                //     points_per_node = DATA_BUFFER_SIZE;
                // }
                long points_per_node_all = points_per_node;

                if (DEBUG)
                    printf("Setting own number of points and initializing own data point buffer.\n");
                // set own number of data points to parse and initialize point buffer
                node_data_buffer_size = points_per_node;
                if (size_of_data_buffer - points_per_node < 0)
                {
                    node_data_buffer_size = size_of_data_buffer;
                }

                if (DEBUG)
                    printf("Own number of points is %d, DATA_BUFFER_SIZE is %d.\n", node_data_buffer_size, DATA_BUFFER_SIZE);
                // copy values from master buffer to master's data buffer
                int l;
                for (l = 0; l < node_data_buffer_size; l++)
                {
                    data_buffer[l] = data_buffer_master[l];
                }

                size_of_data_buffer -= points_per_node;

                if (DEBUG)
                    printf("Computing points to assign to each node. Points per node: %d, data buffer size: %d.\n", points_per_node, node_data_buffer_size);

                int i;
                for (i = 1; i < size && size_of_data_buffer > 0; i++)
                {
                    // send stop_criteria = false
                    MPI_Send(&stop_criteria, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

                    // compute number of points to send
                    if (size_of_data_buffer - points_per_node < 0 || i == size - 1)
                    {
                        points_per_node = size_of_data_buffer;
                    }

                    // send number of points that are sent
                    MPI_Send(&points_per_node, 1, MPI_LONG, i, 0, MPI_COMM_WORLD);

                    // send points
                    Point *data_buffer_temp = data_buffer_master + i * points_per_node_all;
                    MPI_Send(data_buffer_temp, points_per_node, PointType, i, 0, MPI_COMM_WORLD);

                    size_of_data_buffer -= points_per_node;
                }

                // update new active node size to be correct
                int j, newsize = i;
                if (DEBUG)
                    printf("New size of active nodes is %d.\n", newsize);
                for (j = 1; j < newsize; j++)
                {
                    MPI_Send(&newsize, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                }

                if (DEBUG)
                    printf("Stopping %d nodes as they are not needed.\n", size - i);
                int stop_criterion_last_round = 1;
                // the remaining nodes receive stop_criteria = true and exit BFR.
                for (; i < size; i++)
                {
                    MPI_Send(&stop_criterion_last_round, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }

                // update self size
                size = newsize;
                // if(DEBUG) printf("Stop criteria is %d.\n", stop_criteria);
            }
            else
            {
                // data buffer can't be loaded, communicate status to all nodes
                if (DEBUG)
                    printf("Data buffer can't be loaded, communicating status to all nodes.\n");
                int i;
                for (i = 1; i < size; i++)
                {
                    MPI_Send(&stop_criteria, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            // receive status and, if any, the points to parse
            MPI_Recv(&stop_criteria, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (DEBUG)
                printf("\n%d: Received stop criteria %d.\n", rank, stop_criteria);
            if (!stop_criteria)
            {
                // node can receive points
                if (DEBUG)
                    printf("\n%d: Receiving points.\n", rank);
                MPI_Recv(&node_data_buffer_size, 1, MPI_LONG, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (DEBUG)
                    printf("\n%d: Node data buffer size: %d.\n", rank, node_data_buffer_size);
                MPI_Recv(&data_buffer, node_data_buffer_size, PointType, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (DEBUG)
                    printf("\n%d: Receiving new size of active nodes.\n", rank);
                MPI_Recv(&size, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        if(DEBUG_TIME){
            sec_clock_end = MPI_Wtime();
            sec_clock_elapsed = sec_clock_end - sec_clock_start;
            MPI_Reduce(&sec_clock_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank== MASTER)printf("input read and mapped in time %lf\n", sec_clock_elapsed);
        }

        if (!stop_criteria)
        {

            if (DEBUG)
                printf("\n%d: Starting round %d.\n", rank, round);
            
            if (round == 0)
            {
                if (DEBUG_TIME){
                    sec_clock_start = MPI_Wtime();
                }
                // init of the clusters made by the master
                if (rank == MASTER)
                {
                    if (DEBUG)
                        printf("Initializing clusters and centroids.\n");
                    // free clusters as they will be reinitialized in initClustersWithCentroids()
                    free(clusters);
                    // init the clusters with the centroids
                    // can be done as multithreaded version
                    // as reference to implement see take_k_centorids() in serial version
                    // use the whole first buffer to ensure that results are equal to serial version
                    clusters = initClustersWithCentroids(data_buffer_master, size_of_data_buffer_copy, K, M);

                    // print_clusters(clusters);

                    if (DEBUG)
                        printf("Sending clusters info to other nodes.\n");
                    int i;
                    for (i = 1; i < size; i++)
                    {
                        // TODO: adjust this send and next receive
                        if (DEBUG)
                            printf("Sending clusters info to node %d.\n", i);
                        MPI_Send(clusters, K, ArrayclusterType, i, 0, MPI_COMM_WORLD);
                    }
                }
                else
                {
                    if (DEBUG)
                        printf("\n%d: Receiving clusters info from MASTER.\n", rank);
                    // receive the clusters from the master
                    MPI_Recv(clusters, K, ArrayclusterType, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                copy_clusters(clusters, clusters_copy);
                if(DEBUG_TIME){
                    sec_clock_end = MPI_Wtime();
                    sec_clock_elapsed = sec_clock_end - sec_clock_start;
                    MPI_Reduce(&sec_clock_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                    if (rank== MASTER)printf("initializtion in time %lf\n", sec_clock_elapsed);
                }
            }


            if (DEBUG)
                printf("Streaming points.\n");
            if (DEBUG_TIME){
                sec_clock_start = MPI_Wtime();
            }
            // perform primary compression criteria
            StreamPoints(clusters, clusters_copy, compressedSets, retainedSet, data_buffer, node_data_buffer_size);
            if(DEBUG_TIME){
                sec_clock_end = MPI_Wtime();
                sec_clock_elapsed = sec_clock_end - sec_clock_start;
                MPI_Reduce(&sec_clock_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                if (rank== MASTER)printf("primary comp criteria in time %lf\n", sec_clock_elapsed);
            }
            // the master process gets the clusters data from the other processes
            // and updates the clusters
            if (DEBUG_TIME){
                sec_clock_start = MPI_Wtime();
            }
            if (rank == MASTER)
            {
                if (DEBUG)
                    printf("Receiving clusters.\n");
                // the master receives the clusters from the other processes
                // create a temporary clusters to store the clusters from the other processes
                Cluster *tempClusters;

                tempClusters = (Cluster *)malloc(K * sizeof(Cluster));

                int i = 1;
                for (; i < size; i++)
                {
                    MPI_Recv(tempClusters, K, ArrayclusterType, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // update the clusters with the clusters from the other processes
                    // TODO: is important to acknowledge that after the first round the values of the clusters for each M should be erased, if not the clusters will be updated readding also the values of the previous rounds but only the master will have the correct values

                    // The effective change for each cluster equals the node's clusters - the last copy's values
                    int j, d;
                    // # pragma omp parallel for shared(tempClusters, clusters, clusters_copy) private(d)
                    for (j = 0; j < K; j++)
                    {
                        if (DEBUG)
                            printf("size %d tempsize %d copysize %d.\n", clusters[j].size, tempClusters[j].size, clusters_copy[j].size);
                        if (tempClusters[j].size - clusters_copy[j].size == 0)
                            continue;
                        clusters[j].size += (tempClusters[j].size - clusters_copy[j].size);
                        for (d = 0; d < M; d++)
                        {
                            if (DEBUG)
                                printf("sum %lf tempsum %lf copysum %lf.\n", clusters[j].sum[d], tempClusters[j].sum[d], clusters_copy[j].sum[d]);
                            clusters[j].sum[d] += (tempClusters[j].sum[d] - clusters_copy[j].sum[d]);
                            if (DEBUG)
                                printf("sum_squares %lf tempsum_squares %lf copysum_squares %lf.\n", clusters[j].sum_squares[d], tempClusters[j].sum_squares[d], clusters_copy[j].sum_squares[d]);
                            clusters[j].sum_squares[d] += (tempClusters[j].sum_squares[d] - clusters_copy[j].sum_squares[d]);
                        }
                    }
                }
                free(tempClusters);
                if (DEBUG)
                    printf("Updating centroids.\n");
                for (i = 0; i < K; i++)
                {
                    // update the clusters
                    UpdateCentroidsMultithr(clusters, i, K, M);
                }

                // print_clusters(clusters);

                // send the updated clusters to the other processes
                if (DEBUG)
                    printf("Sending updated clusters.\n");
                for (i = 1; i < size; i++)
                {
                    MPI_Send(clusters, K, ArrayclusterType, i, 0, MPI_COMM_WORLD);
                }
            }
            else
            {
                if (DEBUG)
                    printf("\n%d: Sending own clusters.\n", rank);
                // send the clusters to the master
                MPI_Send(clusters, K, ArrayclusterType, MASTER, 0, MPI_COMM_WORLD);

                // create a temporary clusters to store the clusters from the master
                Cluster *tempClusters;

                tempClusters = (Cluster *)malloc(K * sizeof(Cluster));

                if (DEBUG)
                    printf("\n%d: Receiving updated clusters.\n", rank);
                // receive the updated clusters from the master
                MPI_Recv(tempClusters, K, ArrayclusterType, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // update the clusters centroids with the clusters from the master
                int j, d;
                #pragma omp parallel for shared(tempClusters, clusters) private(d)
                for (j = 0; j < K; j++)
                {
                    clusters[j].size = tempClusters[j].size;
                    // clusters[j].size = 1;
                    clusters[j].centroid = tempClusters[j].centroid;
                    for (d = 0; d < M; d++)
                    {
                        clusters[j].sum[d] = tempClusters[j].sum[d];
                        clusters[j].sum_squares[d] = tempClusters[j].sum_squares[d];
                    }
                }

                if (DEBUG)
                    printf("\n%d: Freeing tempClusters.\n", rank);
                free(tempClusters);
            }


            // use a "freezed version" of the clusters to measure distances
            copy_clusters(clusters, clusters_copy);
            if(DEBUG_TIME){
                sec_clock_end = MPI_Wtime();
                sec_clock_elapsed = sec_clock_end - sec_clock_start;
                MPI_Reduce(&sec_clock_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                if (rank== MASTER)printf("update cluster in time %lf\n", sec_clock_elapsed);
            }

            // TODO: we need to decide if the retained set and the compressed sets should be updated in the master or keep in the local processes
            // in the first case we need to send the retained set and the compressed sets to the master and then the master will update the retained set and the compressed sets

            if (DEBUG)
                printf("Reducing Retained Set.\n");
            // Reducing the retained set, by gathering the retained set from the other processes and updating the retained set
            if (DEBUG_TIME){
                sec_clock_start = MPI_Wtime();
            }
            if (rank == MASTER)
            {
                // reduce the retained set
                RetainedSet *tempRetainedSet;
                tempRetainedSet = (RetainedSet *)malloc(sizeof(RetainedSet));
                tempRetainedSet->number_of_points = 0;

                int i = 1;
                for (; i < size; i++)
                {
                    if (DEBUG)
                        printf("Receiving Retained Set from node %d.\n", i);
                    MPI_Recv(&(tempRetainedSet->number_of_points), 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    tempRetainedSet->points = (Point *)malloc(tempRetainedSet->number_of_points * sizeof(Point));
                    MPI_Recv(tempRetainedSet->points, tempRetainedSet->number_of_points, PointType, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // update the retained set with the retained set from the other processes
                    if (DEBUG)
                        printf("Updating Retained Set with the one received from node %d.\n", i);
                    UpdateRetainedSet(retainedSet, tempRetainedSet);

                    free(tempRetainedSet->points);
                }

                free(tempRetainedSet);
            }
            else
            {
                if (DEBUG)
                    printf("\n%d: Sending number of points to MASTER: %d.\n", rank, retainedSet->number_of_points);
                // send the retained set and its size to the master
                MPI_Send(&(retainedSet->number_of_points), 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
                if (DEBUG)
                    printf("\n%d: Sending Retained Set to MASTER.\n", rank);
                MPI_Send(retainedSet->points, retainedSet->number_of_points, PointType, MASTER, 0, MPI_COMM_WORLD);

                free(retainedSet->points);
                // free(retainedSet);

                // retainedSet = (RetainedSet *)malloc(sizeof(RetainedSet));
                retainedSet_normal = init_retained_set();
                retainedSet = &retainedSet_normal;
                retainedSet->number_of_points = 0;
            }
            if(DEBUG_TIME){
                sec_clock_end = MPI_Wtime();
                sec_clock_elapsed = sec_clock_end - sec_clock_start;
                MPI_Reduce(&sec_clock_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                if (rank== MASTER)printf("reducing ret set in time %lf\n", sec_clock_elapsed);
            }

            // TODO: perform the secondary compression criteria in parallel version
            if (DEBUG_TIME){
                sec_clock_start = MPI_Wtime();
            }
            // TODO: insert the kmeans clustering in the parallel version, implemented in "kmeans.c"
            secondary_compression_criteria(clusters, retainedSet, compressedSets, rank, size, PointType);
            if (rank != MASTER)
            {
                retainedSet->number_of_points = 0;
                free(retainedSet->points);
                retainedSet->points = NULL;
            }

            if(DEBUG_TIME){
                sec_clock_end = MPI_Wtime();
                sec_clock_elapsed = sec_clock_end - sec_clock_start;
                MPI_Reduce(&sec_clock_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                if (rank== MASTER)printf("clustering ret set in time %lf\n", sec_clock_elapsed);
            }

            // TODO: filtering of cluster in the master by tightness criterion

            // TODO: merge the clusters in the compressed sets in the master

            // TODO: perform the hierachical clustering in parallel version

            // if (rank == MASTER){
            //     printf("\n%d: Printing old compressed sets.\n");
            //     print_compressedsets(*compressedSets);
            // }
            if (DEBUG_TIME){
                sec_clock_start = MPI_Wtime();
            }
            if (compressedSets->number_of_sets >= K3) {
                // if (rank == MASTER){
                //     printf("\n%d: Printing old compressed sets BEFORE.\n");
                //     print_compressedsets(*compressedSets);
                // }
                hierachical_clust_parallel(compressedSets, rank, size);
            
                // reduce compressed sets to all nodes
                // printf("\n%d: Broadcasting compressedSets' number_of_sets.\n", rank);
                MPI_Bcast(&(compressedSets->number_of_sets), 1, MPI_INT, MASTER, MPI_COMM_WORLD);
                if (rank != MASTER) {
                    // printf("\n%d: Freeing sets.\n", rank);
                    free(compressedSets->sets);
                    // printf("\n%d: Reallocating sets.\n", rank);
                    (*compressedSets).sets = (CompressedSet *)malloc(compressedSets->number_of_sets * sizeof(CompressedSet));
                }
                // printf("\n%d: Broadcasting compressedSets.\n", rank);
                int i;
                for (i = 0; i < compressedSets->number_of_sets; i++){
                    MPI_Bcast(&(compressedSets->sets[i].number_of_points), 1, MPI_INT, MASTER, MPI_COMM_WORLD);
                    MPI_Bcast((*compressedSets).sets[i].sum, M, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                    MPI_Bcast((*compressedSets).sets[i].sum_square, M, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
                }    
            }
            if(DEBUG_TIME){
                sec_clock_end = MPI_Wtime();
                sec_clock_elapsed = sec_clock_end - sec_clock_start;
                MPI_Reduce(&sec_clock_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

                if (rank== MASTER)printf("hierarchical clustering in time %lf\n", sec_clock_elapsed);
            }
            // if (DEBUG){
            //     MPI_Barrier(MPI_COMM_WORLD);
            //     if (rank == MASTER) print_compressedsets(*compressedSets);
            //     MPI_Barrier(MPI_COMM_WORLD);
            // }
            // TODO: synchronize the processes

            if (rank == MASTER)
            {
                if (DEBUG)
                    print_clusters(clusters);
            }

            // increment the round
            round = round + 1;
        }
    } while (!stop_criteria);
    // stop measuring time
    // initialize time variables
    local_finish = MPI_Wtime();

    local_elapsed = local_finish - local_start;

    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // printf("node %d finished the program bye bye\n", rank);

    MPI_Barrier(MPI_COMM_WORLD);   
    // Print time elapsed
    if (rank == MASTER)
    {   
        printf("The maximum elapsed time is: %f\n", elapsed);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double sum;
    MPI_Reduce(&local_elapsed, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double avg;
    if(rank == 0) { // If this is the root process
        avg = sum / size; // num_processes is the total number of processes
        printf("The average elapsed time is: %f\n", avg);
    }

    if (rank ==0){
        double secPerRound= avg/round;
        printf("Time per round: %f\n", secPerRound);
    }
    // Close the input file
    if (rank == MASTER)
    {   
        fclose((FILE *)stream_cursor);
    }

    // if (rank == MASTER)
    // {
    //     // TODO: try merge compressed sets with clusters in the master, using tightness criterion evaluate if the clusters can be merged
    // }

    if (rank == MASTER)
    {
        // print the results
        PlotResults(clusters, retainedSet, compressedSets);
    }

    if (DEBUG)
        printf("%d: Freeing data.\n", rank);
    if (clusters != NULL)
    {
        if (DEBUG)
            printf("%d: Freeing clusters.\n", rank);
        free(clusters);
    }
    if (clusters_copy != NULL)
    {
        if (DEBUG)
            printf("%d: Freeing clusters_copy.\n", rank);
        free(clusters_copy);
    }
    if (retainedSet->points != NULL)
    {
        if (DEBUG)
            printf("%d: Freeing retainedSet points.\n", rank);
        free(retainedSet->points);
    }
    if (compressedSets->sets != NULL)
    {
        if (DEBUG)
            printf("%d: Freeing compressedSets sets.\n", rank);
        free(compressedSets->sets);
    }
    if (compressedSets != NULL)
    {
        if (DEBUG)
            printf("%d: Freeing compressedSets.\n", rank);
        free(compressedSets);
    }

    if (DEBUG)
        printf("%d: Freeing MPI datatypes.\n", rank);
    // free the derived datatypes
    MPI_Type_free(&ArrayclusterType);
    // MPI_Type_free(&RetainedSetType);
    MPI_Type_free(&PointType);

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
