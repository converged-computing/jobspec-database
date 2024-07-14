#!/bin/bash

ior=/glade/work/kpaul/software/ncar-ior/bin/ior

for m in $(seq $2 $3); do
   for t in $(seq $4 $5); do
      M=$(echo "2^$m" | bc)
      T=$(echo "2^($t+3)" | bc)
      B=$(echo "$M*$T" | bc)

      mpiexec_mpt -np $1 $ior -a POSIX -w -r -C -i10 -g -t ${T}m -b ${B}m -e -F > results/ior_psx_n${1}_M${M}_T${T}.out
      mpiexec_mpt -np $1 $ior -a Z5    -w -r -C -i10 -g -t ${T}m -b ${B}m       > results/ior_z5_n${1}_M${M}_T${T}.out
      mpiexec_mpt -np $1 $ior -a NCMPI -w -r -C -i10 -g -t ${T}m -b ${B}m       > results/ior_pnc_n${1}_M${M}_T${T}.out
      mpiexec_mpt -np $1 $ior -a MPIIO -w -r -C -i10 -g -t ${T}m -b ${B}m -c    > results/ior_mpi_n${1}_M${M}_T${T}.out
      mpiexec_mpt -np $1 $ior -a HDF5  -w -r -C -i10 -g -t ${T}m -b ${B}m -c    > results/ior_h5_n${1}_M${M}_T${T}.out
      #mpiexec_mpt -np $1 $ior -a NC4   -w -r -C -i10 -g -t ${T}m -b ${B}m -c    > results/ior_nc4_n${1}_M${M}_T${T}.out
   done
done
