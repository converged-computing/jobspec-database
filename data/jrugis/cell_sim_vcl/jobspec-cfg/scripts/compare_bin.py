#!/usr/bin/env python

# Compare two binary files and look for the maximum element-wise difference.
# If the difference is greater than the tolerance print a message and exit.
#
# Usage: compare_bin.py BIN_FILE_1 BIN_FILE_2 [TOL]

from __future__ import print_function
import os
import sys
import struct

import numpy as np


def read_file(filename):
    print("Reading: '%s' " % filename, end="", flush=True)
    with open(filename, "rb") as fh:
        numrows = struct.unpack('l', fh.read(8))[0]
        numcols = struct.unpack('l', fh.read(8))[0]
        print("(size: %d x %d)... " % (numrows, numcols), end="", flush=True)
        data = np.zeros((numrows, numcols), dtype=np.float32)
        for j in range(numcols):
            for i in range(numrows):
                val = fh.read(4)
                data[i][j] = struct.unpack('f', val)[0]
    print("done", flush=True)    

    return data


def main():
    # check arguments
    if len(sys.argv) < 3 or not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[2]):
        sys.stderr.write("Usage: read_bin.py BIN_FILE_1 BIN_FILE_2 [TOL]\n")
        sys.exit(1)
    
    # read data
    data1 = read_file(sys.argv[1])
    data2 = read_file(sys.argv[2])
    
    # check sizes are the same
    if data1.shape[0] != data2.shape[0] or data1.shape[1] != data2.shape[1]:
        sys.stderr.write("Error: shapes of arrays do not match (%r != %r)\n" % (data1.shape, data2.shape))
        sys.exit(1)

    # tolerance
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    print("Tolerance is: %e" % tol)
    
    # compute max absolute difference between elements
    diff = np.subtract(data1, data2)
    absdiff = np.absolute(diff)
    maxdiff = absdiff.max()
    print("Max difference is: %e" % maxdiff)
    if maxdiff > tol:
        print("Max error is greater than tolerance: %e > %e" % (maxdiff, tol))
        sys.exit(2)

if __name__ == "__main__":
    main()
