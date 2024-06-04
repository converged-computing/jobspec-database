#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N nccl-tests-msccl-1024-2
#PBS -o nccl-tests-msccl-1024-2.out
#PBS -e nccl-tests-msccl-1024-2.error

set -x



# echo "########################################   MSCCL TEST  #####################################################"

# module load cudatoolkit-standalone/11.4.4

# export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
# export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/
# export NCCL_TEST_HOME=/home/yuke/ncclPG/nccl-tests-msccl-test

# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# MSCCL_SRC_LOCATION="/home/yuke/ncclPG/msccl_test"
# MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/msccl-tools-lyd"

# echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch1_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch8_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1 


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch1_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL_gpu64_ch8_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch1_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch8_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch1_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_LL128_gpu64_ch8_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: Simple ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch1_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch8_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 2 PROTOCOL: Simple ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch1_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch8_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL128_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_LL128_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_Simple_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1



# echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins1_hierarchical.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins1_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_LL128_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_BINOMIAL_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1




# echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 1 PROTOCOL: LL ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_LL_nodes_16_gpus_4_ins1_hierarchical.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_LL_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_LL128_nodes_16_gpus_4_ins1_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_LL128_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # export NCCL_DEBUG=TRACE
# # export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins1_hierarchical.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: HIERARCHICAL_4_NOMIAL_TREE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_4_nomial_Simple_nodes_16_gpus_4_ins2_hierarchical.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# echo "########################################   NCCL PROFILE TEST  #####################################################"

# cd /home/yuke/ncclPG/nccl-tests-profile

# #module swap PrgEnv-nvhpc PrgEnv-gnu
# #module load nvhpc-mixed
# #source env.sh
# module load gcc
# module load cudatoolkit-standalone/11.4.4

# export NCCL_MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
# export NCCL_CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4
# export NCCL_HOME=/home/yuke/ncclPG/nccl_profile/build

# #export PATH=${NCCL_MPI_HOME}/bin:$PATH
# export LD_LIBRARY_PATH=${NCCL_CUDA_HOME}/lib64:${NCCL_MPI_HOME}/lib:${NCCL_HOME}/lib:$LD_LIBRARY_PATH

# #make MPI=1 NCCL_MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1 NCCL_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/comm_libs/nccl

# echo "######################### LIBRARY: NCCL ALGORITHM: TREE PROTOCOL: Simple MSSAGE SIZE: 512M ##############################################"

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 128M -e 512M -f 2 -g 1

echo "########################################   NCCL TEST  #####################################################"

cd /home/yuke/ncclPG/nccl-tests-cu116

#module swap PrgEnv-nvhpc PrgEnv-gnu
#module load nvhpc-mixed
#source env.sh
module load nvhpc/23.1
module load cudatoolkit-standalone/11.4.4

export NCCL_MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/cray/10.0
export NCCL_CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4
export NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/comm_libs/nccl

#export PATH=${NCCL_MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${NCCL_CUDA_HOME}/lib64:${NCCL_MPI_HOME}/lib:${NCCL_HOME}/lib:$LD_LIBRARY_PATH

#make MPI=1 NCCL_MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1 NCCL_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda NCCL_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/comm_libs/nccl

# echo "######################### LIBRARY: NCCL ALGORITHM: RING PROTOCOL: LL ##############################################"

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Ring
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

echo "######################### LIBRARY: NCCL ALGORITHM: RING PROTOCOL: SIMPLE ##############################################"

export NCCL_DEBUG=INFO
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple

mpiexec -n 4 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 96 -e 96M -f 2 -g 1

# # echo "######################### LIBRARY: NCCL ALGORITHM: RING PROTOCOL: LL128 ##############################################"

# # export NCCL_DEBUG=INFO
# # export NCCL_ALGO=Ring
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# echo "######################### LIBRARY: NCCL ALGORITHM: TREE PROTOCOL: LL ##############################################"

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Tree
# export NCCL_PROTO=LL

# mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# echo "######################### LIBRARY: NCCL ALGORITHM: TREE PROTOCOL: SIMPLE ##############################################"

# export NCCL_DEBUG=INFO
# export NCCL_ALGO=Tree
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1

# # echo "######################### LIBRARY: NCCL ALGORITHM: TREE PROTOCOL: LL128 ##############################################"

# # export NCCL_DEBUG=INFO
# # export NCCL_ALGO=Tree
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ./build/all_reduce_perf -b 8 -e 512M -f 2 -g 1



# # echo "########################################   MSCCL TEST  #####################################################"

# # module load cudatoolkit-standalone/11.4.4

# # export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
# # export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/
# # export NCCL_TEST_HOME=/home/yuke/ncclPG/nccl-tests-msccl-test

# # export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# # MSCCL_SRC_LOCATION="/home/yuke/ncclPG/msccl_test"
# # MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/msccl-tools-lyd"


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: LL ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL128_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: LL128 ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_LL128_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=LL128

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 1 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_Simple_gpu64_ins1.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# # echo "######################### LIBRARY: MSCCL ALGORITHM: RECV_HALV_DOUBLE INSTANCE: 2 PROTOCOL: SIMPLE ##############################################"

# # export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# # # export NCCL_DEBUG=TRACE
# # # export NCCL_DEBUG_SUBSYS=INIT,ENV
# # export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_Simple_gpu64_ins2.xml
# # export NCCL_ALGO=MSCCL,TREE,RING
# # export NCCL_PROTO=Simple

# # mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1



echo "########################################   MSCCL TEST  #####################################################"

module load cudatoolkit-standalone/11.4.4

export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.4.4/
export NCCL_TEST_HOME=/home/yuke/ncclPG/nccl-tests-msccl-test

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

MSCCL_SRC_LOCATION="/home/yuke/ncclPG/msccl_test"
MSCCL_TOOLS_SRC_LOCATION="/home/yuke/ncclPG/msccl_tools_lyd"


# echo "######################### LIBRARY: MSCCL ALGORITHM: RING INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST RING time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_ring_Simple_gpu64_ch8_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: AllPAIRS INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST AllPAIRS time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_allpairs_v2_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINARY INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINARY time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINARY-PIPE INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINARY-PIPE time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINARY-PIPE INSTANCE: 1 CHANNEL: 4 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINARY-PIPE time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_ch4_h_p_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINARY-PIPE INSTANCE: 1 CHANNEL: 8 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINARY-PIPE time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_ch8_h_p_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINOMIAL INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINOMIAL time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: H-BINOMIAL INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-BINOMIAL time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_h_Simple_gpus_64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1


# echo "######################### LIBRARY: MSCCL ALGORITHM: H-4-NOMIAL INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST H-4-NOMIAL time: $(date)"

# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_4_nomial_tree_h_Simple_gpus_64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 8 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 1-INTRA-2-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 1-INTRA-2-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_1ch_intra_2ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 2-INTRA-4-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 2-INTRA-4-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_4ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 4-INTRA-2-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 4-INTRA-2-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_4ch_intra_2ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: HIE-ALLREDUCE INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST HIE-ALLREDUCE time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_allreduce_hierarchical_allreduce_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: REC-DOUB-HALV INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST REC-DOUB-HALV time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_rec_doub_halv_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINO-H INSTANCE: 1 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINO-H time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binomial_tree_h_ch4_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 1-INTRA-2-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 1-INTRA-2-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_1ch_intra_2ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 2-INTRA-4-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 2-INTRA-4-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_4ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P INSTANCE: 1 CHANNEL: 4-INTRA-2-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P 4-INTRA-2-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_4ch_intra_2ch_inter_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-2NIC-4chunk INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-2NIC-4chunk 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_2nic_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-2NIC-4gpuspipe-8chunk INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-2NIC-8chunk 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2ch_intra_1ch_inter_2nic_4gpu_p_Simple_gpu64_ins1.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-2NIC-8chunk 2-INTRA-1-INTER end time: $(date)"









# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-2NIC-4gpuspipe-16chunk-8ch INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-2NIC-16chunk-8ch 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_16_nch_8.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 512MB -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-2NIC-16chunk-8ch 2-INTRA-1-INTER end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-2NIC-4gpuspipe-32chunk-8ch INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-2NIC-32chunk-8ch 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_32_nch_8.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 512MB -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-2NIC-32chunk-8ch 2-INTRA-1-INTER end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-2NIC-4gpuspipe-64chunk-8ch INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-2NIC-64chunk-8ch 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_64_nch_8.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 512MB -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-2NIC-64chunk-8ch 2-INTRA-1-INTER end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-2NIC-4gpuspipe-128chunk-8ch INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-2NIC-128chunk-8ch 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_128_nch_8.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 512MB -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-2NIC-128chunk-8ch 2-INTRA-1-INTER end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-2NIC-4gpuspipe-256chunk-8ch INSTANCE: 1 CHANNEL: 2-INTRA-1-INTER PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-2NIC-256chunk-8ch 2-INTRA-1-INTER time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nic_4gpu_p_Simple_gpu64_ins1_nchunk_256_nch_8.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 512MB -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-2NIC-256chunk-8ch 2-INTRA-1-INTER end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-2NIC-4gpuspipe-4chunk-8ch INSTANCE: 1 CHANNEL: 8 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-2NIC-4gpuspipe-4chunk-8ch INSTANCE: 1 CHANNEL: 8 time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_8_ch_2nic_4gpu_pipe.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-2NIC-4gpuspipe-4chunk-8ch INSTANCE: 1 CHANNEL: 8 end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-4chunk-8ch-intra-pipe-inter-2nicPtree INSTANCE: 1 CHANNEL: 8 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-4chunk-8ch-intra-pipe-inter-2nicPtree INSTANCE: 1 CHANNEL: 8 time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_8_ch_intra_pipe_inter_2nicPtree.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-4chunk-8ch-intra-pipe-inter-2nicPtree INSTANCE: 1 CHANNEL: 8 end time: $(date)"



# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-8chunk-16ch-intra-pipe-inter-2nicPtree INSTANCE: 1 CHANNEL: 16 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-8chunk-16ch-intra-pipe-inter-2nicPtree INSTANCE: 1 CHANNEL: 16 time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_16_ch_intra_pipe_inter_2nicPtree.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-8chunk-16ch-intra-pipe-inter-2nicPtree INSTANCE: 1 CHANNEL: 16 end time: $(date)"


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-4chunk-2nicPtree INSTANCE: 1 CHANNEL: 4 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-4chunk-2nicPtree INSTANCE: 1 CHANNEL: 4 time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nicPtree_ch_4.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-4chunk-2nicPtree INSTANCE: 1 CHANNEL: 4 end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-8chunk-2:1-2nicPtree INSTANCE: 1 CHANNEL: 4 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-8chunk-2:1-2nicPtree INSTANCE: 1 CHANNEL: 4 time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nicPtree_ch_8_intra_8_inter_4.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-8chunk-2:1-2nicPtree INSTANCE: 1 CHANNEL: 4 end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-16chunk-2:1-2nicPtree INSTANCE: 1 CHANNEL: 8 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-16chunk-2:1-2nicPtree INSTANCE: 1 CHANNEL: 8 time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nicPtree_ch_16_intra_16_inter_8.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-16chunk-2:1-2nicPtree INSTANCE: 1 CHANNEL: 8 end time: $(date)"


# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-16chunk-4:1-2nicPtree INSTANCE: 1 CHANNEL: 10 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-16chunk-4:1-2nicPtree INSTANCE: 1 CHANNEL: 10 time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nicPtree_ch_16_intra_8_inter_2.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-16chunk-4:1-2nicPtree INSTANCE: 1 CHANNEL: 10 end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-16chunk-4:1-2nicPtree-aggre INSTANCE: 1 CHANNEL: 10 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-16chunk-4:1-2nicPtree-aggre INSTANCE: 1 CHANNEL: 10 time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nicPtree_ch_16_intra_8_inter_2_aggre.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 32 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-16chunk-4:1-2nicPtree-aggre INSTANCE: 1 CHANNEL: 10 end time: $(date)"

# echo "######################### LIBRARY: MSCCL ALGORITHM: BINARY-H-P-16chunk-16:8-2nicPtree INSTANCE: 1 CHANNEL: 24 PROTOCOL: Simple ##############################################"

# # Print the current time
# echo "MSCCL TEST BINARY-H-P-16chunk-16:8-2nicPtree INSTANCE: 1 CHANNEL: 24 start time: $(date)"


# export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=INIT,ENV
# export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_binary_tree_h_p_2nicPtree_ch_16_intra_16_inter_8.xml
# export NCCL_ALGO=MSCCL,TREE,RING
# export NCCL_PROTO=Simple

# mpiexec -n 64 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 64 -e 512MB -f 2 -g 1

# echo "MSCCL TEST BINARY-H-P-16chunk-16:8-2nicPtree INSTANCE: 1 CHANNEL: 24 end time: $(date)"

echo "######################### LIBRARY: MSCCL ALGORITHM: A100-RING-SCRATCH INSTANCE: 1 CHANNEL: 24 PROTOCOL: Simple ##############################################"

# Print the current time
echo "MSCCL TEST A100-RING-SCRATCH INSTANCE: 1 CHANNEL: 24 start time: $(date)"


export LD_LIBRARY_PATH=${MSCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,ENV
export MSCCL_XML_FILES=${MSCCL_TOOLS_SRC_LOCATION}/examples/xml/allreduce_a100_ring_ch24_manul_scratch.xml
export NCCL_ALGO=MSCCL,TREE,RING
export NCCL_PROTO=Simple

mpiexec -n 4 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 96 -e 96MB -f 2 -g 1

echo "MSCCL TEST A100-RING-SCRATCH INSTANCE: 1 CHANNEL: 24 end time: $(date)"