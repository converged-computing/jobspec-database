#include "bfr_structures.h"
#include <unistd.h>

RetainedSet init_retained_set(){
    /*
    * Initialize retained set
    *
    * Algorithm:
    *   1. allocate memory for retained set
    *   2. initialize retained set as NULL
    *
    * Parameters:
    *   - void
    *
    * Returns:
    *   - retained set
    */
    RetainedSet R;
    R.points = NULL;
    R.number_of_points = 0;
    return R;
}

CompressedSets init_compressed_sets(){
    /*
    * Initialize compressed sets
    *
    * Algorithm:
    *   1. allocate memory for compressed sets
    *   2. initialize each compressed set as NULL
    *
    * Parameters:
    *   - void
    *
    * Returns:
    *   - array of compressed sets
    */
    CompressedSets C;
    C.sets = NULL;
    C.number_of_sets = 0;
    return C;
}

data_streamer duplicate_data_stream(data_streamer original) {
    int original_fd = fileno(original);
    int new_fd = dup(original_fd);
    if (new_fd == -1) {
        perror("Failed to duplicate file descriptor");
        return NULL;
    }
    data_streamer new_stream = fdopen(new_fd, "r");
    if (!new_stream) {
        perror("Failed to open new file stream");
        close(new_fd);
        return NULL;
    }
    return new_stream;
}

Cluster * init_cluster(int k){
    /*
    * Initialize clusters
    *
    * Algorithm:
    *   1. allocate memory for clusters
    *   2. initialize each cluster
    *
    * Parameters:
    *   - void
    *
    * Returns:
    *   - array of clusters
    */
    Cluster * clusters = malloc(k * sizeof(Cluster));
    
    if (clusters == NULL){
        perror("Error: could not allocate memory\n");
        exit(1);
    }

    int i = 0;
    for (i = 0; i < k; i++){
        Cluster  c;
        int j = 0;
        for (j = 0; j < M; j++){
            c.centroid.coords[j] = 0.;
            c.sum[j] = 0.;
            c.sum_squares[j] = 0.;
        }
        c.size = 0;
        c.index = i;
        clusters[i] = c;
    }

    return clusters;
}

data_streamer data_streamer_Init(char * file_name, char * mode){
    /*
    * Initialize data streamer
    *
    * Algorithm:
    *   1. open file
    *   2. return file pointer
    *
    * Parameters:
    *   - file_name: name of file to open
    *   - mode: mode to open file in
    *
    * Returns:
    *   - file pointer
    */
    FILE * file = fopen(file_name, mode);
    if (file == NULL){
        perror("Error: could not open file\n");
        exit(1);
    }

    //get size of file ---> not needed?
    // fseek(file, 0, SEEK_END);
    // *size_of_file = ftell(file);
    // fseek(file, 0, SEEK_SET);


    return file;
}

PriorityQueue createPriorityQueue(int capacity) {
    /*
    * Create priority queue
    *
    * Algorithm:
    *   1. allocate memory for priority queue
    *   2. allocate memory for priority queue data
    *   3. initialize priority queue size
    *   4. initialize priority queue capacity
    *
    * Parameters:
    *   - capacity: capacity of priority queue
    *
    * Returns:
    *   - pointer to priority queue
    */
    PriorityQueue pq;
    pq.data = (hierc_element*)malloc(capacity * sizeof(hierc_element));
    pq.size = 0;
    pq.capacity = capacity;
    return pq;
}

void print_clusters(Cluster * clusters){
    printf("Clusters:\n");
    int i = 0;
    for (i = 0; i < K; i++){
        printf("Cluster %d: ", i);
        int j = 0;
        for (j = 0; j < M; j++){
            printf("%lf ", clusters[i].centroid.coords[j]);
        }
        printf("\n");

        printf("Cluster %d size: %d\n", i, clusters[i].size);

        printf("Cluster %d sum: ", i);

        for (j = 0; j < M; j++){
            printf("%lf ", clusters[i].sum[j]);
        }

        printf("\n");

        printf("Cluster %d sum_squares: ", i);

        for (j = 0; j < M; j++){
            printf("%lf ", clusters[i].sum_squares[j]);
        }

        printf("\n");
    }
}

void print_compressedsets(CompressedSets C){
    printf("Compressed Sets:\n");
    int i = 0;
    for(; i < C.number_of_sets; i++){
        printf("Set %d size: %d\n", i, C.sets[i].number_of_points);

        printf("Set %d sum: ", i);
        int j = 0;
        for (j = 0; j < M; j++){
            printf("%lf ", C.sets[i].sum[j]);
        }
        printf("\n");

        printf("Set %d sum_squares: ", i);
        j = 0;
        for (j = 0; j < M; j++){
            printf("%lf ", C.sets[i].sum_square[j]);
        }
        printf("\n");
    }
}

void print_retainedset(RetainedSet R){
    printf("Retained Set:\n");
    printf("Retained Set size: %d\n", R.number_of_points);
    int i = 0;
    for (; i < R.number_of_points; i++){
        printf("Point %d: ", i);
        int j = 0;
        for (j = 0; j < M; j++){
            printf("%lf ", R.points[i].coords[j]);
        }
        printf("\n");
    }
}

void print_priorityqueue(PriorityQueue* pq){
    //print priority queue in order of priority
    printf("Priority Queue: \n");
    PriorityQueue pq2_actual = createPriorityQueue(pq->capacity);
    PriorityQueue * pq2 = &pq2_actual;
    pq2->size = pq->size;
    int j = 0;
    for (j = 0; j < pq->size; j++){
        pq2->data[j].distance = pq->data[j].distance;
        pq2->data[j].index_of_cset_1 = pq->data[j].index_of_cset_1;
        pq2->data[j].index_of_cset_2 = pq->data[j].index_of_cset_2;
    }
    while (pq2->size != 0){
        hierc_element * hd2 = top_from_pqueue(*pq2);
        pop_from_pqueue(pq2);
        if ( hd2 == false ){
            perror("error in pop function, the size vlaue do not correspond to real size of vector\n");
            exit(2);
        }
        printf("%lf ", hd2->distance);
        free(hd2);
    }
    printf("\n");
    free(pq2->data);
}

void add_point_to_retained_set(RetainedSet * R, Point p){
    (*R).number_of_points += 1;
    (*R).points = realloc(R->points, R->number_of_points * sizeof(Point));
    if (R->points == NULL){
        perror("Error: could not allocate memory\n");
        exit(1);
    }
    (*R).points[R->number_of_points - 1] = p;
    // int i = 0;
    // printf("Added point");
    // for (i = 0; i < M; i++){
    //     printf(" %lf", p.coords[i]);
    // }
    // printf(" to RetainedSet.\n");
}

void update_cluster(Cluster * cluster, Point p){
    /*
    * Update cluster
    *
    * Algorithm:
    *   1. update cluster size
    *   2. update cluster sum
    *   3. update cluster sum_squares
    *
    * Parameters:
    *   - cluster: cluster to update
    *   - p: point to add to cluster
    *
    * Returns:
    *   - void
    */
    cluster->size += 1;
    int i = 0;
    for (i = 0; i < M; i++){
        cluster->sum[i] += p.coords[i];
        cluster->sum_squares[i] += pow(p.coords[i], 2);
    }
}

void update_centroids(Cluster ** clusters, int number_of_clusters){
    /*
    * Update centroids of clusters
    *
    * Algorithm:
    *   1. iter over clusters
    *   2. if cluster size is greater than 1, update centroid
    *
    * Parameters:
    *   - clusters: array of clusters
    *   - number_of_clusters: number of clusters
    *
    * Returns:
    *   - void
    */

    int i;
    for (i = 0; i < number_of_clusters; i++){
        int size = (*clusters)[i].size;
        if (size >= 1){
            Point new_centroid;
            int j = 0;
            for (j = 0; j < M; j++){
                new_centroid.coords[j] = (double) (*clusters)[i].sum[j] / (double) (*clusters)[i].size; 
            }
            new_centroid.cluster = (*clusters)[i].centroid.cluster;
            (*clusters)[i].centroid = new_centroid;
        }
    }
}

void add_miniclusters_to_compressedsets(CompressedSets * C, Cluster * miniclusters, int number_of_miniclusters){
    int i;
    for (i=0; i<number_of_miniclusters; i++){
        // if (miniclusters[i].size > 1){
            // add minicluster to compressed sets
            (*C).number_of_sets += 1;
            (*C).sets = realloc(C->sets, C->number_of_sets * sizeof(CompressedSet));
            if (C->sets == NULL){
                perror("Error: could not allocate memory\n");
                exit(1);
            }
            (*C).sets[C->number_of_sets - 1].number_of_points = miniclusters[i].size;
            int j;
            for (j=0; j<M; j++){
                (*C).sets[C->number_of_sets - 1].sum[j] = miniclusters[i].sum[j];
                (*C).sets[C->number_of_sets - 1].sum_square[j] = miniclusters[i].sum_squares[j];
            }
        // }
    }
}

CompressedSet merge_compressedsets(CompressedSet C1, CompressedSet C2){
    CompressedSet C;
    C.number_of_points = C1.number_of_points + C2.number_of_points;
    int i;
    for (i=0; i<M; i++){
        C.sum[i] = C1.sum[i] + C2.sum[i];
        C.sum_square[i] = C1.sum_square[i] + C2.sum_square[i];
    }
    return C;
}

void remove_compressedset(CompressedSets * C, int i, int j, bool ** cset_validity){
    if(DEBUG) printf("Setting cset_validity's following values to false: %d %d\n", i, j);    
    (*cset_validity)[i] = false;
    (*cset_validity)[j] = false;

    if (C->sets == NULL){
        perror("Error: could not allocate memory\n");
        exit(1);
    }

    if(DEBUG) {
        int i = 0;
        for(; i<C->number_of_sets; i++) {
            printf("%d ", (*cset_validity)[i]);
        }
        printf("\n");
    }
    if(DEBUG) printf("Set cset_validity's following values to false: %d %d\n", i, j);    
}

bool * add_compressedset(CompressedSets * C, CompressedSet C1, bool * cset_validity){
    /*
    * Add compressed set
    *
    * Algorithm:
    *   1. increase number of sets
    *   2. reallocate memory for sets
    *   3. add compressed set to sets
    *
    * Parameters:
    *   - C: compressed sets
    *   - C1: compressed set to add
    *
    * Returns:
    *   - temp_validity: array of bools indicating if compressed set is valid, if temp_validity[i] == true, then i can be added to the new CompressedSets
    */
    (*C).number_of_sets += 1;

    if(DEBUG) printf("Reallocating C.sets and cset_validity with size: %d\n", C->number_of_sets);
    (*C).sets = realloc(C->sets, C->number_of_sets * sizeof(CompressedSet));
    if ((*C).sets == NULL){
        perror("Error: could not allocate memory\n");
        exit(1);
    }

    bool* temp_validity = realloc(cset_validity, C->number_of_sets * sizeof(bool));
    if (temp_validity == NULL){
        perror("Error: could not allocate memory\n");
        exit(1);
    }

    temp_validity[C->number_of_sets - 1] = true;

    
    if(DEBUG) printf("Reallocated C.sets and cset_validity with size: %d\n", C->number_of_sets);
    C->sets[C->number_of_sets - 1] = C1;

    if(DEBUG) {
        int i = 0;
        for(; i<C->number_of_sets; i++) {
            printf("%d ", temp_validity[i]);
        }
        printf("\n");
    }

    return temp_validity;
}

bool pop_from_pqueue(PriorityQueue *pq){
    /*
    * Pop from priority queue
    *
    * Algorithm
    */

    // If the priority queue is empty, return NULL.
    if (pq->size == 0) {
        return false;
    }

    // Save the top of the priority queue.
    hierc_element * hd = malloc(sizeof(hierc_element));
    hd->distance = pq->data[0].distance;
    hd->index_of_cset_1 = pq->data[0].index_of_cset_1;
    hd->index_of_cset_2 = pq->data[0].index_of_cset_2;

    // Move the last element to the top of the priority queue.
    pq->data[0] = pq->data[pq->size - 1];

    // Decrease the size of the priority queue.
    pq->size -= 1;

    // "Sift down" the element that was moved to the top.
    int i = 0;
    while (2 * i + 1 < pq->size) {
        // Find the minimum child.
        int min_child = 2 * i + 1;
        if (2 * i + 2 < pq->size && pq->data[2 * i + 2].distance < pq->data[min_child].distance) {
            min_child = 2 * i + 2;
        }

        // If the parent is greater than the minimum child, swap them.
        if (pq->data[i].distance > pq->data[min_child].distance) {
            hierc_element temp = pq->data[i];
            pq->data[i] = pq->data[min_child];
            pq->data[min_child] = temp;
        }else{
            break;
        }

        // Move to the child.
        i = min_child;
    }

    if (DEBUG) printf("Popped from priority queue: %lf\n", hd->distance);

    free(hd);

    return true;
}

bool remove_from_pqueue(PriorityQueue *pq, int i1, int i2){
    /*
    * Remove from priority queue
    *
    * Algorithm:
    *   1. create new priority queue
    *   2. add all elements from priority queue except those with index i1 or i2
    *   3. free old priority queue
    *   4. set old priority queue to new priority queue
    *
    * Parameters:
    *   - pq: priority queue
    *   - i1: index of compressed set 1
    *   - i2: index of compressed set 2
    *
    * Returns:
    *   - void
    */
    PriorityQueue pq2 = createPriorityQueue(pq->capacity);
    pq2.size = 0;
    int j = 0;
    for (j = 0; j < pq->size; j++){
        if (pq->data[j].index_of_cset_1 != i1 && pq->data[j].index_of_cset_2 != i2 && pq->data[j].index_of_cset_1 != i2 && pq->data[j].index_of_cset_2 != i1){
            add_to_pqueue(&pq2, pq->data[j]);
            if(DEBUG) printf("Adding couple %d %d to prioqueue.\n\n\n", pq->data[j].index_of_cset_1, pq->data[j].index_of_cset_2);
        }
    }

    hierc_element * temporary_pointer = pq->data;
    pq->data = NULL;
    free(temporary_pointer);
    
    pq->data = pq2.data;
    pq->size = pq2.size;
    pq->capacity = pq2.capacity;
    if (DEBUG) printf("Items deleted from queue, new size: %d, new capacity: %d\n", pq->size, pq->capacity);
}

bool add_to_pqueue(PriorityQueue *pq, hierc_element hd){
    /*
    * Add to priority queue
    *
    * Algorithm:
    *   1. if priority queue is full, return
    *   2. add data to priority queue
    *   3. increase priority queue size
    *   4. sift up priority queue data
    *
    * Parameters:
    *   - pq: priority queue
    *   - hd: data to add to priority queue
    *
    * Returns:
    *   - void
    */
    if (DEBUG) printf("Adding to priority queue: %lf, size: %d, capacity: %d\n", hd.distance, pq->size, pq->capacity);
    if (pq->size == pq->capacity){
        if (DEBUG) printf("Size has reached capacity: %d = %d\n", pq->size, pq->capacity);
        return false;
    }
    if(DEBUG) printf("Adding hd: %lf %d %d\n", hd.distance, hd.index_of_cset_1, hd.index_of_cset_2);
    pq->data[pq->size] = hd;
    pq->size += 1;

    // "Sift up" the new element.
    int i = pq->size - 1;
    if (DEBUG) printf("     prioqueue's current i: %d, value: %lf\n", i, pq->data[i].distance);
    while (i != 0 && pq->data[i].distance < pq->data[(i - 1) / 2].distance) {
        // Swap the new element with its parent.
        hierc_element temp = pq->data[i];
        pq->data[i] = pq->data[(i - 1) / 2];
        pq->data[(i - 1) / 2] = temp;

        // Move to the parent.
        i = (i - 1) / 2;
        if (DEBUG) printf("     prioqueue's current i: %d, value: %lf\n", i, pq->data[i].distance);
    }
    if (DEBUG) printf("Added to priority queue: %lf\n\n", hd.distance);
    
    return true;
}

bool can_merge(CompressedSet c1, CompressedSet c2){
    //TODO: implement can_merge
    return true;
}

double distance_compressedsets(CompressedSet c1, CompressedSet c2){
    // calculate the two centroids coordinates
    double * coord_c1 = malloc(M * sizeof(double));
    double * coord_c2 = malloc(M * sizeof(double));
    int i;

    // calculate centroid for c1
    for (i = 0; i < M; i++) {
        coord_c1[i] = (double)c1.sum[i] / c1.number_of_points;
    }

    // calculate centroid for c2
    for (i = 0; i < M; i++) {
        coord_c2[i] = (double)c2.sum[i] / c2.number_of_points;
    }

    // calculate the distance between the two centroids
    double distance = 0;
    for (i = 0; i < M; i++) {
        distance += pow(coord_c1[i] - coord_c2[i], 2);
    }
    distance = sqrt(distance);

    // free the allocated memory
    free(coord_c1);
    free(coord_c2);

    return distance;
}

hierc_element * top_from_pqueue(PriorityQueue pq){
    /*
    * Top from priority queue
    *
    * Algorithm:
    *   1. if priority queue is empty, return NULL
    *   2. return top of priority queue
    *
    * Parameters:
    *   - pq: priority queue
    *
    * Returns:
    *   - top of priority queue
    */
    if (pq.size == 0){
        return NULL;
    }

    hierc_element * hp = malloc(sizeof(hierc_element));
    hp->index_of_cset_1 = pq.data[0].index_of_cset_1;
    hp->index_of_cset_2 = pq.data[0].index_of_cset_2;
    hp->distance = pq.data[0].distance;

    return hp;
}

// unsigned int hash_compressedset(CompressedSet cs) {
//     unsigned int hash = 0;
//     for (int i = 0; i < M; i++) {
//         hash += cs.sum[i];
//         hash += (hash << 10);
//         hash ^= (hash >> 6);
//     }
//     hash += (hash << 3);
//     hash ^= (hash >> 11);
//     hash += (hash << 15);
//     return hash;
// }

bool is_empty_pqueue(PriorityQueue * pq){
    if (pq->size == 0){
        return true;
    }else{
        return false;
    }
}

bool * restore_csets(CompressedSets * C, bool * cset_validity){
    /*
    * Restore compressed sets
    *
    * Algorithm:
    *   1. create new compressed sets
    *   2. add all valid compressed sets to new compressed sets
    *   3. free old compressed sets
    *   4. set old compressed sets to new compressed sets
    *
    * Parameters:
    *   - C: compressed sets
    *   - cset_validity: array of bools indicating if compressed set is valid, if cset_validity[i] == true, then i can be added to the new CompressedSets
    *
    * Returns:
    *   - cset_validity: array of bools indicating if compressed set is valid, if cset_validity[i] == true, then i can be added to the new CompressedSets
    */
    CompressedSets C2;
    C2.number_of_sets = 0;
    C2.sets = NULL;
    
    int i;
    for (i=0; i<C->number_of_sets; i++){
        bool * newtemp_cv = cset_validity;
        if (cset_validity[i]){
            newtemp_cv = add_compressedset(&C2, C->sets[i], cset_validity);
        }
        if (newtemp_cv != cset_validity) {
            // free(cset_validity);
            cset_validity = newtemp_cv;
        }
    }

    free(C->sets);
    C->number_of_sets = C2.number_of_sets;
    C->sets = C2.sets;
    return cset_validity;
}

bool tightness_evaluation_cset(CompressedSet c){
    /*
    * Tightness evaluation for compressed set
    *
    * Algorithm:
    *   1. calculate max value
    *   2. if max value is less than BETA, return true
    *   3. else, return false
    *
    * Parameters:
    *   - c: compressed set
    *
    * Returns:
    *   - bool indicating if compressed set is tight
    */
    int x_sub[M];
    int i = 0;
    for (i = 0; i < M; i++){
        x_sub[i] = c.sum[i];
        x_sub[i] = x_sub[i] / c.number_of_points;
    }
    int j = 0;
    double max_value = 0;
    for (j = 0; j < M; j++){
        double value = c.sum_square[j] - (c.number_of_points * pow(x_sub[j], 2));
        value = value / c.number_of_points ;
        value = sqrt(value);
        if (value > max_value){
            max_value = value;
        }
    }
    if (DEBUG) printf("              Max value for tightness constraint: %lf.\n", max_value);
    if (max_value < BETA){
        return true;
    }else{
        return false;
    }
}

void merge_compressedsets_and_miniclusters(CompressedSets * C, Cluster * miniclusters, int number_of_miniclusters){
    /*
    * Algorithm:
    *   1. add miniclusters to compressed sets
    *   2. calculate distances between compressed sets
    *   3. priority queue implementation of hierchical clustering over miniclusters and compressed sets
    *
    * Parameters:
    *   - C: compressed sets
    *   - miniclusters: miniclusters
    *   - number_of_miniclusters: number of miniclusters
    *
    * Returns:
    *   - void
    */
    if (number_of_miniclusters > 0){
        add_miniclusters_to_compressedsets(C, miniclusters, number_of_miniclusters);
    }
    if(DEBUG) printf("\n\nNumber of compressed sets after adding miniclusters: %d\n\n", C->number_of_sets);

    bool stop_merging = false;
    bool * cset_validity = malloc(C->number_of_sets * sizeof(bool));
    // bool cset_validity[MAX_SIZE_OF_BUFFER * (MAX_SIZE_OF_BUFFER - 1) / 2];
    int i;
    for (i=0; i<C->number_of_sets; i++){
        cset_validity[i] = true;
    }

    // calculate distances between compressed sets
    PriorityQueue pq = createPriorityQueue(C->number_of_sets * (C->number_of_sets - 1) / 2);
    if(DEBUG) printf("Adding all possible combinations of sets.\n");
    for (i=0; i<C->number_of_sets; i++){
        int j;
        for (j=i+1; j<C->number_of_sets; j++){
            if(DEBUG) printf("i: %d, j: %d\n", i, j);
            hierc_element hd;
            hd.distance = distance_compressedsets(C->sets[i], C->sets[j]);
            hd.index_of_cset_1 = i;
            hd.index_of_cset_2 = j;
            add_to_pqueue(&pq, hd);
        }
    };
    if(DEBUG) printf("Added all possible combinations of sets.\n");

    int number_of_merges = 0;
    // naive implementation of hierchical clustering over miniclusters and compressed sets
    while (!stop_merging && pq.size > K3){
        // if(DEBUG) print_priorityqueue(pq);
        // top element of priority queue
        hierc_element * hd = top_from_pqueue(pq);
        if (hd == NULL){
            stop_merging = true;
        }else{
            CompressedSet merged = merge_compressedsets(C->sets[hd->index_of_cset_1], C->sets[hd->index_of_cset_2]);
            pop_from_pqueue(&pq);
            if(tightness_evaluation_cset(merged)){
                number_of_merges++;
                //TODO: merge and remove from priority queue all element with same index and calculate new distance between merged and all other compressed sets
                // remove_from_pqueue(pq, hd->index_of_cset_1, hd->index_of_cset_2);
                bool * newtemp_cv = add_compressedset(C, merged, cset_validity);
                // free(cset_validity);
                cset_validity = newtemp_cv;

                if(DEBUG) printf("Merging %d and %d.\n", hd->index_of_cset_1, hd->index_of_cset_2);
                remove_compressedset(C, hd->index_of_cset_1, hd->index_of_cset_2, &cset_validity);

                remove_from_pqueue(&pq, hd->index_of_cset_1, hd->index_of_cset_2);
                
                //calculate distances between merged and all other compressed sets
                int i;
                for (i=0; i<(C->number_of_sets-1); i++){
                    if (cset_validity[i]){
                        if(DEBUG) printf("i: %d\n", i);
                        hierc_element hd;
                        hd.distance = distance_compressedsets(C->sets[i], merged);
                        hd.index_of_cset_1 = i;
                        hd.index_of_cset_2 = C->number_of_sets - 1;
                        add_to_pqueue(&pq, hd);
                    }
                }
                if(DEBUG) printf("Finished adding new combinations containing new set.\n");
            }
        }
        if (is_empty_pqueue(&pq) || pq.size <= K3){
            stop_merging = true;
        }
        if (hd != NULL) free(hd);
    }

    if(DEBUG) printf("Restoring compressed sets based n cset_validity.\n");
    bool* temp_validity = restore_csets(C, cset_validity);
    // free(cset_validity);
    cset_validity = temp_validity;

    if(cset_validity != NULL) {
        bool* cset_pointer = cset_validity;
        // cset_validity = NULL;
        free(cset_pointer);
        cset_pointer = NULL;
        cset_validity = NULL;
    }
    if(pq.data != NULL) free(pq.data);
}