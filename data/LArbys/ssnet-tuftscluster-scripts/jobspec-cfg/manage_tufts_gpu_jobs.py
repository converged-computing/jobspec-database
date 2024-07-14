import os,sys,time
from choose_gpu import pick_gpu

# This script is meant to shepard SSNET jobs through the available cards.
# Currently, 3 SSNET jobs can fit on one card

#INPUTLISTDIR = sys.argv[2]
#OUTDIR = sys.argv[3]
#JOBLIST = sys.argv[4]
CONTAINER = sys.argv[1]
WORKDIR = sys.argv[2]

waitsec = 10
MAX_NUM_JOBS=6
MAX_JOBS_PER_GPU=3

# first get jobid list

frerunlist = open("rerunlist.txt",'r')
lrerunlist = frerunlist.readlines()
jobids = []
for l in lrerunlist:
    jobids.append(int(l.strip()))

print "number of remaining jobs: ",len(jobids)

runningids = []
runningprocs = []
gpuslot = {}
gpujobs = {0:[],
           1:[]}

# event loop
# really dumb. we loop every X seconds and check how much memory is available on the gpu. 
# if memory available, we add a job. move id from jobids to runningids
# we need to pick an X where there is enough time for the gpu to load the job. else we cause a disaster.

while len(jobids)>0 or len(runningids)>0:
    # we check two things: 
    #  (1) there is available memory on a gpu
    #  (2) how many jobs are running
    print "----------------------------------------"
    psmi = os.popen("nvidia-smi")
    smilines = psmi.readlines()
    for l in smilines:
        l = l.strip()
        print l
    print "----------------------------------------"
    print "STATUS UPDATE"
    print "Remaining jobs: ",len(jobids)
    for jobid in runningids:
        # report on status
        jobstatus = open("job_ssnet_status_%d.txt"%(jobid),'r')
        statuslines = jobstatus.readlines()
        if len(statuslines)==0:
            status="LOADING"
        else:
            status = statuslines[-1].strip()
        print jobid,": ",status
        if status in ["ERROR","SUCCESS"]:
            runningids.remove(jobid)
            gpujobs[gpuslot[jobid]].remove(jobid)
    nrunning = len(runningids)
    print "Number of running jobs: ",nrunning
    for gpuid in range(0,2):
        print "Number of GPU%d jobs: %d"%(gpuid,len(gpujobs[gpuid]))
    if nrunning>=MAX_NUM_JOBS:
        print "Max number of jobs. Waiting for jobs to complete."
        print "--------------------------------------------------"
        time.sleep(waitsec)
        continue


    if len(jobids)==0 and nrunning==0:
        print "============"
        print "FINISHED!!!"
        print "============"
        break

    if len(jobids)>0:
        #available_gpu = pick_gpu(mem_min=6000,caffe_gpuid=True)
        # we assign gpus by which are available
        available_gpu = -1
        for gpuid in range(0,2):
            if len(gpujobs[gpuid])<MAX_JOBS_PER_GPU:
                available_gpu = gpuid
        if available_gpu>=0:
            print "Available GPU (",available_gpu,")"
            jobid = jobids.pop()
            procid = len(jobids)
            #os.system( "./run_single_davis_job.sh %d %d"%(procid,available_gpu) )
            #os.system( "./run_single_tufts_job.sh %d %d"%(procid,available_gpu) )
            os.system( "singularity exec --nv %s bash -c \"cd %s && source setup_tufts_container.sh && cd %s && source run_single_tufts_job.sh %d %d\""%(CONTAINER,WORKDIR,WORKDIR,procid,available_gpu) )
            gpuslot[jobid] = available_gpu
            gpujobs[available_gpu].append(jobid)
            runningids.append(jobid)
            runningprocs.append(procid)
        else:
            print "No space right now"
    print "Now wait %d seconds for job to launch."%(waitsec)
    time.sleep(waitsec)
    
