#!/bin/bash
#PBS -P kr97
#PBS -q {{q}} 
#PBS -l walltime={{walltime}}
#PBS -l ncpus={{ncpus}}
#PBS -l mem={{mem}}
#PBS -N {{{name}}}
#PBS -l wd
#PBS -o {{{o}}}
#PBS -e {{{e}}} 

source {{{modules}}}

mpiexec -n {{n}} julia --project={{{projectdir}}} -O3 -e\
      'using ClusterBenchmarks; ClusterBenchmarks.assembly_neighbors_main(np={{np}},case=:{{case}},title="{{{title}}}")'
