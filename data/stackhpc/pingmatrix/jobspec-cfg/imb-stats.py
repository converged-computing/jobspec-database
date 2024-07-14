#!/usr/bin/env python
""" Extract parameters from IMB-MPI pingpong output.
    
    Usage:
        python [<parameter>] <pingpong stdout file>

    where parameter can be:
    - latency0: latency for zero-size messages, in us [default]
    - max_bandwidth: maximum bandwidth, in MB/s

    The requested value is output to stdout.
"""

import sys, pprint

if __name__ == '__main__':
    
    sys.argv.pop(0)
    fpath = sys.argv.pop(-1)
    parameter = sys.argv[0] if sys.argv else 'latency0'
    
    data = {'size':[], 'reps':[], 'latency':[], 'bandwidth':[]}
    with open(fpath) as pingout:
        for line in pingout:
            if line.startswith('       #bytes #repetitions      t[usec]   Mbytes/sec'):
                for line in pingout:
                    parts = line.strip().split()
                    if not parts:
                        break
                    for ix, k in enumerate(data): # depends on dicts being insertion order in py3
                        data[k].append(float(parts[ix]))
                    #print(size, reps, latency, bw)
    if parameter == 'latency0':
        print(data['latency'][0])
    elif parameter == 'max_bandwidth':
        print(max(data['bandwidth']))
    else:
        exit('Invalid parameter %r, see docstring' % parameter)