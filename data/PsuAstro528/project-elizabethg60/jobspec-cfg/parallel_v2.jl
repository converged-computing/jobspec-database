using Distributed

addprocs(16)

@everywhere using MyProject

@everywhere MyProject.get_kernels()

MyProject.parallel_v2()