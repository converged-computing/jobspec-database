
#!/usr/bin/env python
import os,sys,signal

import time
import pathlib
import logging
from logging import handlers

import numpy

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra

from config.run import Run

#############################

# Add the local folder to the import path:

network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = network_dir + "/config/"
sys.path.insert(0,network_dir)

from larcv_fetcher import larcv_fetcher

class iotest(object):

    def __init__(self, config):

        self.args = config


        self.rank = self.init_mpi()


        # Create the output directory if needed:
        if self.rank == 0:
            outpath = pathlib.Path(self.args.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        # Need to check if the process has already completed
        should_run = self.check_run()


        if not should_run:
            exit()

        self.configure_logger(self.rank)

        logger = logging.getLogger()
        logger.info(f"Size is {self.size}")

        self.iotest()



    def init_mpi(self):
        if not self.args.distributed:
            self.size=1
            return 0
        else:
            if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
                max_gpus = 8
                target_gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) % max_gpus
                os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu)

            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self.size=comm.Get_size()

            return comm.Get_rank()


    def configure_logger(self, rank):

        logger = logging.getLogger()

        # Create a handler for STDOUT, but only on the root rank.
        # If not distributed, we still get 0 passed in here.
        if rank == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            handler = handlers.MemoryHandler(capacity = 0, target=stream_handler)
            logger.addHandler(handler)

            # Add a file handler too:
            log_file = self.args.output_dir + "/process.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler = handlers.MemoryHandler(capacity=10, target=file_handler)
            logger.addHandler(file_handler)

            logger.setLevel(logging.INFO)
        else:
            # in this case, MPI is available but it's not rank 0
            # create a null handler
            handler = logging.NullHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)


    def check_run(self):

        # This function inspects the output folder, looks for the
        # process.log file.  It inspects the file and looks for the 
        # string "Total IO Time: "
        # 
        # If it finds this, it does not repeat this run.
        # 
        # In distributed mode, only the root rank checks the file.
        
        # Only the root rank checks:

        if self.rank == 0:
            # First, does the file exist?
            file = pathlib.Path(self.args.output_dir + "/process.log")
            if not file.exists():
                should_run = True
            else:
                # Open the file and look at it's contents:
                found = False
                with open(file) as _f:
                    for line in _f:
                        if "Total IO Time:" in line: 
                            found = True
                            break
                if not found: should_run = True
                else: should_run = False


        # Broadcast decision all ranks and return

        if self.args.distributed:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            should_run = comm.bcast(should_run, root=0)

        # If not distributed, we don't need to broadcast, just return:
        return should_run

    def iotest(self):

        logger = logging.getLogger()

        logger.info("Running IO Test")
        logger.info(self.__str__())

        self.make_fetcher()



        # label_stats = numpy.zeros((36,))
        global_start = time.time()
        time.sleep(0.5)
        # print("\n\n\n\nBegin Loop \n\n\n\n")
        for i in range(self.args.iterations):
            start = time.time()
            time.sleep(0.0)
            mb = self.larcv_fetcher.fetch_next_batch("iotest", force_pop=True)
            end = time.time()

            logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.6f}s")

        logger.info(f"Total IO Time: {time.time() - global_start:.4f}s")






    def dictionary_to_str(self, in_dict, indentation = 0):
        substr = ""
        for key in sorted(in_dict.keys()):
            if type(in_dict[key]) == DictConfig or type(in_dict[key]) == dict:
                s = "{none:{fill1}{align1}{width1}}{key}: \n".format(
                        none="", fill1=" ", align1="<", width1=indentation, key=key
                    )
                substr += s + self.dictionary_to_str(in_dict[key], indentation=indentation+2)
            else:
                s = '{none:{fill1}{align1}{width1}}{message:{fill2}{align2}{width2}}: {attr}\n'.format(
                   none= "",
                   fill1=" ",
                   align1="<",
                   width1=indentation,
                   message=key,
                   fill2='.',
                   align2='<',
                   width2=30-indentation,
                   attr = in_dict[key],
                )
                substr += s
        return substr

    def __str__(self):

        s = "\n\n-- CONFIG --\n"
        substring = s +  self.dictionary_to_str(self.args)

        return substring


    def make_fetcher(self):
        distributed = self.args.distributed
        dataset     = self.args.dataset
        self.larcv_fetcher = larcv_fetcher(distributed, dataset)

        # Configure the dataset:
        self.larcv_fetcher.prepare_sample(self.args.minibatch_size)

        return







@hydra.main(config_path="config", config_name="iotest")
def main(cfg : OmegaConf) -> None:


    s = iotest(cfg)


if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled']
    main()
