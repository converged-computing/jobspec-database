#!/bin/bash

# > Nodes & CPUs:
#PBS -l nodes=1:ppn=8
#
# > Memory: (  58 GiB =  59392 MiB )
# > Memory: (  98 GiB = 100352 MiB )
# > Memory: ( 318 GiB = 325632 MiB )
#PBS -l mem=325632mb
#
# > Estimated runtime:
#PBS -l walltime=30-00:00:00
#
# > Email:
#PBS -m abe
#PBS -M cluster-notifications@recursion.ninja
#
# > Job name:
#PBS -N CGKA-TreeKEM-Benchmarks
#
# > Queue:
#PBS -q batch-amd
#
# > Shell:
#PBS -S /bin/bash
#
# > Inherit environment variables
#PBS -V
#
MINDFA=24
VECLEN=72

#  56 GiB =  57344 MiB
#  96 GiB =  98304 MiB
# 316 GiB = 323584 MiB
DIRECTIVE_DEFAULTS=(
    'HC4'
    'JOINPROCS'
    'MEMLIM=323584'
    'MURMUR'
    'NOBOUNDCHECK'
    'NOFIX'
    'SEPQS'
    'SFH'
    'SPACE'
)

#SLURM_ARRAY_TASK_ID=0
#DIRECTIVE_ELEMENTS=(
#)

#    'JOINPROCS'
#    'MURMUR'
#    'NOBOUNDCHECK'
#    'NOFIX'
#    'SEPQS'
#    'SFH'

#DIRECTIVE_ELEMENTS=(
#    'COLLAPSE'
#    'JOINPROCS'
#    'MURMUR'
#    'NOBOUNDCHECK'
#    'NOFAIR'
#    'NOFIX'
#    'NOVSZ'
#    'SEPQS'
#    'SFH'
#)
    
#    'SAFETY'

DIRECTIVE_ELEMENTS=(
    'NOFAIR'
    "VECTORSZ=${VECLEN}"
)

FLAG_DEFAULTS=("${DIRECTIVE_DEFAULTS[@]/#/-D}")
FLAG_ELEMENTS=("${DIRECTIVE_ELEMENTS[@]/#/-D}")
FLAG_SPECTRUM=()

count=${#FLAG_ELEMENTS[@]}
bound=$((1 << ${count}))
i=0
while [ ${i} -lt ${bound} ]
do
    subset=""
    j=0
    while [ ${j} -lt ${count} ]
    do
        directive="${FLAG_ELEMENTS[$j]} "
        if [ $(((1 << ${j}) & ${i})) -gt 0 ]
        then
            subset+="${directive}"
        else
            number=${#directive}
            subset+=`printf '%*s' "${#directive}"`
        fi
        j=$((${j} + 1))
    done
    FLAG_SPECTRUM+=("${subset}")
    i=$((${i} + 1))
done

if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
    SLURM_ARRAY_TASK_ID=0
#    echo "No specified 'SLURM_ARRAY_TASK_ID'"
#    exit 2
fi

NAME_BINARY="CGKA-Bench-${SLURM_ARRAY_TASK_ID}"

FLAG_ELECTION="${FLAG_SPECTRUM[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${FLAG_ELECTION}" ]]; then
    FLAG_RENDERED="{} (Empty Set)"
else
    FLAG_RENDERED="{ $(echo -e "${FLAG_ELECTION}") }"
fi

echo -e "Benchmarking selected directive set:\n\t${FLAG_RENDERED}"

gcc \
    ${FLAG_DEFAULTS[@]} \
    ${FLAG_ELECTION} \
    -O3 \
    -o "${NAME_BINARY}" \
    pan.h pan.c

chmod +x "${NAME_BINARY}"

# Search depth set to
#  | N |   Depth Limit | Exact Depth |
#  |:-:|--------------:|------------:|
#  | 3 |     1,000,000 |     289,137 |
#  | 4 |    10,000,000 |   5,823,059 |
#  | 5 |   200,000,000 |  22,614,336 |
#  | 6 | 2,000,000,000 |  77,008,431 |
#  | 7 | 
./"${NAME_BINARY}" -a -A -m2000000000 -N FSU -v -w32 -x

sstat --allsteps --format=AveCPU,AvePages,AveRSS,AveVMSize,JobID --jobs ${SLURM_JOBID} --parsable

#/usr/bin/time -f "DIRECTIVE: ${FLAG_ELECTION}\n\tElapsed time: %es\n\tMemory usage: %M KB\n\tCPU usage: %P\n\n${FLAG_ELECTION}\t%e\t%M\n\n" "./${NAME_BINARY}" -a -A -v -x

# 
