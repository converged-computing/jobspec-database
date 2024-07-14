# %% import stuff
import numpy as np
import openpiv.tools
import openpiv.process
import openpiv.scaling
import cv2
import time
import multiprocessing 
import os
import sys
import ctypes
import sys
import scipy.io as sio
from PIL import Image
import pypar as pp
import warnings

num_processors = pp.size()
rank = pp.rank()
node = pp.get_processor_name()

# MPI Constants
MASTER_PROCESS = 0
WORK_TAG = 1
DIE_TAG = 2

DeltaT = 1 # frames to skip
window_size = 24
overlap = 12



def PIVCompute(frame_a, frame_b, window_size = 24, overlap = 12):
    
    tmpu, tmpv, sig2noise = openpiv.pyprocess.piv(frame_a, frame_b,
                    window_size=window_size, overlap=overlap, dt=1, 
                    sig2noise_method='peak2peak', corr_method = 'direct')
    
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=window_size, overlap=overlap)
    tmpu, tmpv, mask = openpiv.validation.sig2noise_val( tmpu, tmpv, sig2noise, threshold = 1.3)
    u, v = openpiv.filters.replace_outliers( tmpu, tmpv, method='localmean', max_iter=10, kernel_size=4)
    
    return u, v


############################################################################################
# main part of code

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # load in the folder path, then grab just the top directory from the path by 
    # splitting the string, and stripping the trailing slash
    vidpath = sys.argv[1]
    foldername = vidpath.rstrip('/').split('/')[-1]

    print "Vidpath = " + vidpath
    print "Foldername = " + foldername

    
    tif_files = sorted([f for f in os.listdir(vidpath) if f.endswith('.tif')]) # Sorted is important on the cluster!

# !!!! DEBUG
    tif_files = tif_files[0:10]


# Create list of file pairs to process
    process_list = zip(range(0,len(tif_files)-DeltaT), range(DeltaT,len(tif_files)))
    work_size = len(process_list)
    
    # Dispatch jobs to worker processes
    work_index = 0
    num_completed = 0

    # Master process
    if rank == MASTER_PROCESS:
        start_time = time.time()

        #
        # print(process_list)

        frame_a = np.array(Image.open(os.path.join(vidpath, tif_files[0])));
        frame_b = np.array(Image.open(os.path.join(vidpath, tif_files[1])));

        # run PIV computation to get size of matrix and x,y's
        x, y = openpiv.process.get_coordinates(image_size=frame_a.shape,
                                                 window_size=window_size, overlap=overlap)

        # pre-allocate the u, v matrices
        u = np.zeros((work_size, x.shape[0], x.shape[1]))
        v = np.zeros((work_size, x.shape[0], x.shape[1]))

        # make list of sources and the tasks sent to them
        source_list = np.zeros(num_processors)

        # Start all worker processes
        for i in range(0, min(num_processors-1, work_size)):

            proc = i+1
            source_list[proc] = work_index

            pp.send(work_index, proc, tag=WORK_TAG)
            pp.send(process_list[i], proc)
            print "Sent process list " + str(process_list[i]) + " to processor " + str(proc)

            work_index += 1

        # Receive results from each worker, and send it new data
        for i in range(num_processors-1, work_size):
            results, status = pp.receive(source=pp.any_source, tag=pp.any_tag, return_status=True)
            proc = status.source
            
            index = source_list[proc]
            print "index is " + str(index) + " from process " + str(proc) + " u origin value is " + str(results[0,0,0])
            
            # receive and parse the resulting var
            u[index,:,:] = results[0,:,:]
            v[index,:,:] = results[1,:,:]
            
            # re-up workers
            pp.send(work_index, proc, tag=WORK_TAG)
            pp.send(process_list[work_index], proc)
            source_list[proc] = work_index
            
            print "Sent work index " + str(work_index) + " to processor " + str(proc)

            # increment work_index
            work_index += 1
            num_completed += 1

            
        # Get results from remaining worker processes
        while num_completed < work_size:
            results, status = pp.receive(source=pp.any_source, tag=pp.any_tag, return_status=True)
            proc = status.source

            index = source_list[proc]

            print "index is " + str(index) + " from process " + str(proc) + " u origin value is " + str(results[0,0,0])

            # receive and parse the resulting var
            u[index,:,:] = results[0,:,:]
            v[index,:,:] = results[1,:,:]

            num_completed += 1

        # Shut down worker processes
        for proc in range(1, num_processors):
            print "Stopping worker process " + str(proc)
            pp.send(-1, proc, tag=DIE_TAG)

        # Package up the results to save, also save all the PIV parameters
        sio.savemat(os.path.join(vidpath, '../' + foldername + '_CLUSTER.mat'),{'x':x, 'y':y, 'u':u,                                                        'v': v, 
                                                                    'window_size':window_size,
                                                                    'overlap':overlap})
        end_time = time.time()
        print repr(end_time - start_time)


    else:
        ### Worker Processes ###
        continue_working = True
        while continue_working:

            # check if being put to sleep
            work_index, status =  pp.receive(source=MASTER_PROCESS, tag=pp.any_tag, 
                    return_status=True)

            if status.tag == DIE_TAG:
                continue_working = False
            
            # not being put to sleep, load in videos of interest and compute
            else:
                frame_pair, status = pp.receive(source=MASTER_PROCESS, tag=pp.any_tag, 
                    return_status=True)
                work_index = status.tag
                

                frame_a = np.array(Image.open(os.path.join(vidpath, tif_files[frame_pair[0]])));
                frame_b = np.array(Image.open(os.path.join(vidpath, tif_files[frame_pair[1]])));
                
                # Code below simulates a task running
                u, v = PIVCompute(frame_a, frame_b, window_size = window_size, overlap = overlap)
                print  "Received work frame pair " + str(frame_pair) + " u origin value is " + str(u[0,0])                 
                
                # package up into work array
                work_array = np.zeros((2,u.shape[0], u.shape[1]))
                work_array[0,:,:] = u
                work_array[1,:,:] = v

                result_array = work_array.copy()

                pp.send(result_array, destination=MASTER_PROCESS, tag=work_index)
        #### while
    #### if worker

    pp.finalize()

