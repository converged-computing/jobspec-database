import os, sys
import hostlist  #pip install python-hostlist
from mpi4py import MPI
from time import sleep

if __name__ == '__main__':

    comm_world = MPI.COMM_WORLD
    worldrank = comm_world.Get_rank()
    worldsize = comm_world.Get_size()

    assert worldrank == int(os.environ['SLURM_PROCID']) #check for multinode
    local_rank = int(os.environ['SLURM_LOCALID'])
    size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
    local_nodeID = int(os.environ["SLURM_NODEID"])
    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
        
    # get IDs of reserved GPU
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = str(os.environ['SLURM_STEP_RESV_PORTS'].split("-")[-1])

    # assert len(gpu_ids) == 1
    gpu_id = int(gpu_ids[0])
    sub_comm = comm_world.Split(color=(gpu_id+(local_nodeID)*worldsize))
    # space_size = space_comm.Get_size()
    sub_rank = sub_comm.Get_rank()
    sub_size = sub_comm.Get_size()
    
    print(f'{worldrank=} on {local_nodeID=} in Comm with {worldsize=} over {hostnames=}. {gpu_ids=}. {sub_rank=} {sub_size=}')

    sys.stdout.flush()

    if sub_rank == 0:
        import numpy as np
        import torch
        import torch.distributed as dist

        distrank = int(worldrank/3)
        distsize = int(worldsize/3)

        print(f'{distrank=} {distsize=} {gpu_id=} {os.environ["CUDA_VISIBLE_DEVICES"]=}')
        #torch.cuda.set_device(gpu_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)
        # USES ENVs: MASTER_ADDR and MASTER_PORT that are set in slurmTorch.py
        dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=distsize,
                            rank=distrank)
        print(f'{distrank=} : created process group')
        ranktensor = torch.tensor(distrank, device=device)
        print(f'{distrank=} {distsize=} {ranktensor=}')
        dist.all_reduce(ranktensor,op=dist.ReduceOp.SUM)
        if distrank == 0:
            print('All_reduced')
        dist.barrier()
        print(f'{distrank=} {distsize=} {ranktensor=}')
        
        assert ranktensor.cpu().numpy() == np.sum(np.array(range(distsize)))
        dist.destroy_process_group()

    ### other processes will be used for data preprocessing
    comm_world.Barrier()
    
