#!/usr/bin/env python
#Parallel HelloWorld
import sys
from mpi4py import MPI

def main():
	size = MPI.COMM_WORLD.Get_size()
	rank = MPI.COMM_WORLD.Get_rank()
	name = MPI.Get_processor_name()
	print(f"Hello, World! I am process {rank}  of {size} on {name}.txt")

if __name__ == "__main__":
	main()
