#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --nodes=__NODES__
#SBATCH --chdir=__CLUSTER_JOB_DIR__
#SBATCH -o __LOG_FILE__
#SBATCH -e __LOG_FILE__
#SBATCH --partition=__PARTITION__


# Description:
# This script demonstrates running a multinode MPI job in a SLURM cluster using Docker.
# The scheduler directive --ntasks-per-node directive is set to 1 because each node hosts
# a single docker container.
# If using all available cores use the --exclusive directive

# Use all available cores
CPUS_PER_NODE=$(grep -c '^processor' /proc/cpuinfo)
NP=$((SLURM_JOB_NUM_NODES*CPUS_PER_NODE))


OMPI_CONTAINER_REPO="__OMPI_DOCKER_REPO__"
# All containers have the same name but run on different nodes
OMPI_CONTAINER_NAME="mpitest"
# All containers share the same network
OMPI_DOCKER_NETWORK="mpinet"

# Start Docker service on all compute nodes
srun sudo service docker start

# Create a Docker swarm cluster
sudo docker swarm init | grep -A 2 "docker swarm join " | head -n1 > docker-swarm-join.sh

# Join all compute nodes to the swarm cluster
srun sudo bash docker-swarm-join.sh

# Create a network for all the containers
sudo docker network create --attachable -d overlay ${OMPI_DOCKER_NETWORK}

# Start a container on each compute node
srun sudo docker run -d --rm --name ${OMPI_CONTAINER_NAME} --network ${OMPI_DOCKER_NETWORK} -v `pwd`:`pwd` -w `pwd` ${OMPI_CONTAINER_REPO} sleep infinity

# Obtain internal IP of each container
rm -rf mpi-container.hosts
srun sudo docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${OMPI_CONTAINER_NAME} >> mpi-container.hosts

# Create a hello world MPI code
cat <<EOF > mpi_hello_world.c
// Author: Wes Kendall
// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// An intro MPI hello world program that uses MPI_Init, MPI_Comm_size,
// MPI_Comm_rank, MPI_Finalize, and MPI_Get_processor_name.
//
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\\n",
         processor_name, world_rank, world_size);

  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();
}
EOF

# Compile MPI code
sudo docker exec ${OMPI_CONTAINER_NAME} mpicc -o mpi_hello_world mpi_hello_world.c
sudo chmod +x mpi_hello_world

# Run MPI code
sudo docker exec ${OMPI_CONTAINER_NAME} mpirun \
    --allow-run-as-root \
    --mca plm_rsh_agent "ssh -q -o StrictHostKeyChecking=no" \
    -np ${NP} \
    --hostfile mpi-container.hosts \
    ./mpi_hello_world &> mpi_hello_world.out

cat mpi_hello_world.out

# Stop and remove all containers
srun sudo docker stop ${OMPI_CONTAINER_NAME}

# Remove all nodes from the swarm cluster
srun sudo docker swarm leave --force
