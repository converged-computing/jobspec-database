#!/bin/bash

cd ./build

# Windows or Linux?
if [[ -n ${!MSYSTEM} ]]; then
	export PATH="$MSMPI_BIN":$PATH
	mpiexec -n4 ./MpiMatrixTest "$@"
else
	mpirun  ./MpiMatrixTest "$@"
fi
