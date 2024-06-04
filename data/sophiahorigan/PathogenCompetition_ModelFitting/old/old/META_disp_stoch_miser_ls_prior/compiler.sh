#!/bin/bash
#PBS -S /bin/bash
#PBS -d .

echo "compiling $1"

echo `gcc -I/usr/local/include -O3 -c -g $1.c`
echo `gcc -L/usr/local/lib $1.o -lgsl -lgslcblas -lm -o $1`
