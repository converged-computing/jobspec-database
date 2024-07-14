#!/usr/bin/env python
""" Report zero-size message latency and max bandwidth from IMB-MPI pingpong.

    Usage:
        python <pingpong stdout file>
    
    Prints <zero_size_latency> us <max_bandwidth> MB/s
"""

import sys, pprint
fpath = sys.argv[1]
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
print('%s us' % data['latency'][0], '%s MB/s' % max(data['bandwidth']))

