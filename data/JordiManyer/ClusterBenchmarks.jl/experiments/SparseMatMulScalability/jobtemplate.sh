#!/bin/bash
#PBS -P bt62
#PBS -q {{q}} 
#PBS -l walltime={{walltime}}
#PBS -l ncpus={{ncpus}}
#PBS -l mem={{mem}}
#PBS -N {{{name}}}
#PBS -l wd
#PBS -o {{{o}}}
#PBS -e {{{e}}} 

source {{{modules}}}

mpiexec -n {{n}} julia --project={{{projectdir}}} -O3 --check-bounds=no -e\
      'using ClusterBenchmarks; ClusterBenchmarks.sparse_matmul_main(nc={{nc}},np={{np}},order={{order}},title="{{{title}}}")'
