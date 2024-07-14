from gen_mesh.generateMesh import genMesh
from multiprocessing import Pool, Queue
from logger import log
import pickle
import os
import sys, traceback

def frange(*args):
    """A float range generator."""
    # Adapted from http://code.activestate.com/recipes/66472/
    start = 0.0
    step = 1.0

    l = len(args)
    if l == 1:
        end = args[0]
    elif l == 2:
        start, end = args
    elif l == 3:
        start, end, step = args
        if step == 0.0:
            raise ValueError("step must not be zero")
    else:
        raise TypeError("frange expects 1-3 arguments, got %d" % l)

    v = start
    i = 0
    while True:
        if (step > 0 and v >= end) or (step < 0 and v <= end):
            raise StopIteration
        yield v
        i += 1
        v = start + i*step

def call_catch(func, args):
    try:
        func(*args)
    except:
        log("Exception in Python thread:")
        traceback.print_exc(file=sys.stdout)
        print

def process_queue(queue):
    pool = Pool(processes = int(os.getenv('OMP_NUM_THREADS', 4)))
    while not queue.empty():
        pool.apply_async(call_catch, (genMesh, queue.get()))

    pool.close()
    pool.join()

if __name__ == '__main__':
    q = Queue()
    dirs = []

    scratch = os.getenv('SCRATCH', "/scratch/pr63so/ga25cux2")
    rseed='0254887388'
    minWavelength = 1.
    alphaExp = -2.0
    for alphaExp in frange(-2.0, -2.4, -0.2):
        dirs.append(scratch + "/roughgen/rseed{}_minwv{}_alpha{}".format(rseed, minWavelength, alphaExp))
        q.put([dirs[-1] + "_temp", rseed, 40., 20., minWavelength, pow(10.,alphaExp), 0.8])

    # Write directory names to file
    dirs_file = open("dirs.txt",'w')
    pickle.dump(dirs, dirs_file)
    dirs_file.close()

    # Create meshes parallelly
    process_queue(q)
