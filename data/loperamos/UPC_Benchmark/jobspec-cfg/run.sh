#!/bin/bash

cd build/bin
if [ $1 == "R" ] ; then
	upcrun -shared-heap 512MB Benchmark $2 
else
	upcrun -shared-heap 512MB -q -backtrace -freeze=0 Benchmark $2
fi
cd ../../