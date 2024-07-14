/*-------------------------------------------------------------------------
*
* kmeans.c
*    Generic k-means implementation
*
* Copyright (c) 2016, Paul Ramsey <pramsey@cleverelephant.ca>
*
*------------------------------------------------------------------------*/


#include "kmeans.h"

static void
update_r(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->num_objs; i++)
	{
		double distance, curr_distance;
		int cluster, curr_cluster;
		Pointer obj;

		assert(config->objs != NULL);
		assert(config->num_objs > 0);
		assert(config->centers);
		assert(config->clusters);

		obj = config->objs[i];

		/*
		* Don't try to cluster NULL objects, just add them
		* to the "unclusterable cluster"
		*/
		if (!obj)
		{
			config->clusters[i] = KMEANS_NULL_CLUSTER;
			continue;
		}

		/* Initialize with distance to first cluster */
		curr_distance = (config->distance_method)(obj, config->centers[0]);
		curr_cluster = 0;

		/* Check all other cluster centers and find the nearest */
		for (cluster = 1; cluster < config->k; cluster++)
		{
			distance = (config->distance_method)(obj, config->centers[cluster]);
			if (distance < curr_distance)
			{
				curr_distance = distance;
				curr_cluster = cluster;
			}
		}

		Point* clusterArray = (Point*)config->centers[curr_cluster];
		double* cluster_coords_ptr = clusterArray->coords;

		Point* pointArray = (Point*)obj;
		double* coords_ptr = pointArray->coords;
		// printf("\n%d: Node %lf %lf added to cluster %d (%lf, %lf) with dist %lf.\n",config->rank, coords_ptr[0], coords_ptr[1], curr_cluster, cluster_coords_ptr[0], cluster_coords_ptr[1], curr_distance);
		/* Store the nearest cluster this object is in */
		config->clusters[i] = curr_cluster;
	}
}

static void
update_means(kmeans_config *config)
{
	int i;

	for (i = 0; i < config->k; i++)
	{
		/* Update the centroid for this cluster */
		(config->centroid_method)(config->objs, config->clusters, config->num_objs, i, config->centers[i]);
	}
}

#ifdef KMEANS_THREADED

static void * update_r_parallel_main(void *args)
{
	kmeans_config *config = (kmeans_config*)args;
	update_r(config);
}

static void update_r_parallel(kmeans_config *config)
{
	int obs_per_node = config->num_objs / config->size;

	/* For each node, create a config copy with the objects, clusters and num_objs offset correctly*/
	kmeans_config node_config;

	// printf("\n%d: Copying memory.\n", config->rank);
	memcpy(&(node_config), config, sizeof(kmeans_config));
	node_config.objs += config->rank*obs_per_node;
	node_config.clusters += config->rank*obs_per_node;
	node_config.num_objs = obs_per_node;
	if (config->rank == config->size-1)
	{
		node_config.num_objs += config->num_objs - config->size*obs_per_node;
	}

	// printf("\n%d: Updating r main.\n", config->rank);
	/* Run the node, on its subset of the data */
	update_r_parallel_main((void *) &node_config);

	// Determine displacement and count for each process
    int* displs = (int*)malloc(sizeof(int) * config->size);
    int* rcounts = (int*)malloc(sizeof(int) * config->size);
    int total_elements = 0, i;
	for (i = 0; i < config->size; i++) {
        displs[i] = total_elements;
        rcounts[i] = obs_per_node;
		if (i == config->size-1){
			rcounts[i] += config->num_objs - config->size*obs_per_node;
		}
        total_elements += obs_per_node;
    }

	int *clusters_copy = malloc(node_config.num_objs*sizeof(int));

	// printf("\n%d: Copying local results.\n",config->rank);
	# pragma omp parallel for 
	for (i = 0; i < node_config.num_objs; i++) {
		clusters_copy[i] = node_config.clusters[i];
	}

	
	// printf("\n%d: Gathering results.\n", config->rank);
	MPI_Allgatherv(clusters_copy, rcounts[config->rank], MPI_INT, config->clusters, rcounts, displs, MPI_INT, MPI_COMM_WORLD);	


	free(clusters_copy);
	free(displs);
	free(rcounts);
}


static void
update_means_parallel(kmeans_config *config)
{
	/* What we want to do here is assign each node a number of clusters. Each will update its center and the results will be shared to all nodes. */
	if(config->k <= config->size){
		// printf("\n%d: Means are less than nodes.\n", config->rank);

		if(config->rank < config->k){
			// printf("\n%d: Receiving one mean to edit.\n", config->rank);
			/* Assign one cluster to each node until no cluster can be assigned */
			int clusters_per_node = 1;
			int clusters_per_node_all = clusters_per_node;

			/* For each node, recompute mean and send new centroid coordinates to MASTER */
			int offset;
			offset = config->rank;
			// printf("\n%d: Calling centroid method for mean of offset %d. Coords were %lf, %lf.\n", config->rank, offset, ((Point*)config->centers[offset])->coords[0], ((Point*)config->centers[offset])->coords[1]);
			(config->centroid_method)(config->objs, config->clusters, config->num_objs, offset, config->centers[offset]);
			
			Point* pointArray = (Point*)config->centers[offset];
	
			double* coords_ptr = pointArray->coords;

			// printf("\n%d: Point coords are %lf, %lf.\n", config->rank, coords_ptr[0], coords_ptr[1]);
			if(config->rank != MASTER) {
				// printf("\n%d: Sending coords to MASTER.\n", config->rank);
				MPI_Send(coords_ptr, M, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
			}
		}

		/* MASTER has to receive each centroid coordinates and update them accordingly */
		if(config->rank == MASTER){
			// printf("\n%d: Receiving coords (MASTER).\n", config->rank);
			double* temp_coords_ptr = (double*)malloc(M*sizeof(double));
			/* Receive new centers */
			int i, offset, k;
			for(i = 1; i < config->k; i++){
				offset = i;
				MPI_Recv(temp_coords_ptr, M, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				
				Point* pointArray = (Point*)config->centers[offset];
				// Copy received values to the coords array
				# pragma omp parallel for
				for (k = 0; k < M; k++) {
					pointArray->coords[k] = temp_coords_ptr[k];
    			}
			}
			// printf("\n%d: Freeing temp coords ptr.\n", config->rank);
			free(temp_coords_ptr);
		}

	}
	else{
		// printf("\n%d: There are more means than nodes.\n", config->rank);
		/* Assign a (possibly) equal number of clusters to each node*/
		int clusters_per_node = config->k / config->size;
		int clusters_per_node_all = clusters_per_node;
		if (config->rank == config->size-1)
		{
			/* Last node gets the remainder */
			clusters_per_node += config->k - config->size*clusters_per_node;
		}

		/* For each node, recompute mean and send new centroid coordinates to MASTER */
		int i, offset;
		for(i = 0; i < clusters_per_node; i++){
			offset = i + config->rank * clusters_per_node_all;
			(config->centroid_method)(config->objs, config->clusters, config->num_objs, offset, config->centers[offset]);
            
			Point* pointArray = (Point*)config->centers[offset];
	
			double* coords_ptr = pointArray->coords;
			if(config->rank != MASTER) MPI_Send(coords_ptr, M, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
		}

		/* MASTER has to receive each centroid coordinates and update them accordingly */
		if(config->rank == MASTER){
			double* temp_coords_ptr = (double*)malloc(M*sizeof(double));
			/* Receive new centers */
			int j, k;
			for(i = 1; i < config->size; i++){
				clusters_per_node = config->k / config->size;
				clusters_per_node_all = clusters_per_node;
				if (i == config->size-1){
					clusters_per_node += config->k - config->size*clusters_per_node;
				}

				for(j = 0; j < clusters_per_node; j++){
					offset = j + i * clusters_per_node_all;
					MPI_Recv(temp_coords_ptr, M, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					
					Point* pointArray = (Point*)config->centers[offset];

					// Copy received values to the coords array
					# pragma omp parallel for
					for (k = 0; k < M; k++) {
						pointArray->coords[k] = temp_coords_ptr[k];
    				}
				}
			}
			free(temp_coords_ptr);
		}
	}

	// printf("\n%d: Broadcasting coordinates.\n", config->rank);
	int i, k;
	for (i = 0; i < config->k; i++){
		Point* pointArray = (Point*)config->centers[i];
		double* coords_ptr = pointArray->coords;
		MPI_Bcast(coords_ptr, M, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		if(config->rank != MASTER){
			// printf("\n%d: Updating coordinates.\n", config->rank);
			/* Update centroid coordinates */
			# pragma omp parallel for
			for (k = 0; k < M; k++) {
				pointArray->coords[k] = coords_ptr[k];
    		}
			// printf("\n%d: New coords are: %lf %lf.\n", config->rank, pointArray->coords[0], pointArray->coords[1]);
		}
	}
}

#endif /* KMEANS_THREADED */

kmeans_result
kmeans(kmeans_config *config)
{
	int iterations = 0;
	int *clusters_last;
	size_t clusters_sz = sizeof(int)*config->num_objs;

	assert(config);
	assert(config->objs);
	assert(config->num_objs);
	assert(config->distance_method);
	assert(config->centroid_method);
	assert(config->centers);
	assert(config->k);
	assert(config->clusters);
	assert(config->k <= config->num_objs);
	// assert(config->parallel);
	// assert(config->rank);
	// assert(config->size);

	/* Zero out cluster numbers, just in case user forgets */
	memset(config->clusters, 0, clusters_sz);
	// printf("\n%d: Starting kmeans.\n", config->rank);

	/* Set default max iterations if necessary */
	if (!config->max_iterations)
		config->max_iterations = KMEANS_MAX_ITERATIONS;

	/*
	 * Previous cluster state array. At this time, r doesn't mean anything
	 * but it's ok
	 */
	clusters_last = kmeans_malloc(clusters_sz);

	while (1)
	{
		// if(config->rank == MASTER){
		// 	// printf("0: cluster assignment BEFORE\n");
		// 	int i;
		// 	for (i = 0; i < config->num_objs; i++){
		// 		// printf("%d ", config->clusters[i]);
		// 	}
		// 	// printf("\n");
		// }
		// printf("\n%d: Iterating.\n", config->rank);

		/* Store the previous state of the clustering */
		memcpy(clusters_last, config->clusters, clusters_sz);

#ifdef KMEANS_THREADED
		if(config->parallel && config->num_objs > config->size && config->size > 1){
			// printf("\n%d: In parallel, updating r.\n", config->rank);
			/* At this point, all nodes have the same config. Have master coordinate the clustering, then broadcast the results. */
			update_r_parallel(config);
			// printf("\n%d: In parallel, updating means.\n", config->rank);
			update_means_parallel(config);
		}
		else{
#endif
			update_r(config);
			update_means(config);
#ifdef KMEANS_THREADED
		}
#endif
		// if(config->rank == MASTER){
		// 	// printf("0: cluster assignment AFTER\n");
		// 	int i;
		// 	for (i = 0; i < config->num_objs; i++){
		// 		// printf("%d ", config->clusters[i]);
		// 	}
		// 	// printf("\n");
		// }
		/*
		 * if all the cluster numbers are unchanged since last time,
		 * we are at a stable solution, so we can stop here
		 */
		if (memcmp(clusters_last, config->clusters, clusters_sz) == 0)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_OK;
		}

		if (iterations++ > config->max_iterations)
		{
			kmeans_free(clusters_last);
			config->total_iterations = iterations;
			return KMEANS_EXCEEDED_MAX_ITERATIONS;
		}
	}

	kmeans_free(clusters_last);
	config->total_iterations = iterations;
	return KMEANS_ERROR;
}


