#!/usr/bin/env python
import os,sys,signal
import time
import pathlib
import logging
from logging import handlers
import datetime

import numpy

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

hydra.output_subdir = None

from src.config import Config

class VertexID(object):

    def __init__(self, config):

        self.args = config
        self._rank = self.init_mpi(self.args.run.distributed)

        # Create the output directory if needed:
        if self._rank == 0:
            outpath = pathlib.Path(self.args.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(self._rank)

        # self.validate_arguments()
        from src.config import ModeKind
        if config.mode.name == ModeKind.train:
            self.train()
        if config.mode.name == ModeKind.iotest:
            self.iotest()
        if config.mode.name == ModeKind.inference:
            self.inference()

    def init_mpi(self, distributed):
        if not distributed:
            return 0
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
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
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # self.args.log_directory = self.args.log_directory + '/' + time_str
            pathlib.Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
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


    def train(self):

        logger = logging.getLogger("cosmictagger")

        logger.info("Running Training")
        logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()


    def iotest(self):

        self.make_trainer()
        logger = logging.getLogger("cosmictagger")

        logger.info("Running IO Test")
        logger.info(self.__str__())


        self.trainer.initialize(io_only=True)

        if self.args.run.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        # label_stats = numpy.zeros((36,))
        global_start = time.time()
        time.sleep(0.1)
        for i in range(self.args.run.iterations):
            start = time.time()
            mb = self.trainer.larcv_fetcher.fetch_next_batch("primary", force_pop=True)

            end = time.time()

            logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.2f}s")

        total_time = time.time() - global_start
        images_read = self.args.run.iterations * self.args.run.minibatch_size
        logger.info(f"Total IO Time: {total_time:.2f}s")
        logger.info(f"Total images read per batch: {self.args.run.minibatch_size}")
        logger.info(f"Average Image IO Throughput: { images_read / total_time:.3f}")


    def inference(self):


        logger = logging.getLogger("cosmictagger")

        logger.info("Running Inference")
        logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()





    #
    # def add_network_parsers(self, parser):
    #     # Here, we define the networks available.  In io test mode, used to determine what the IO is.
    #     network_parser = parser.add_subparsers(
    #         title          = "Networks",
    #         dest           = "network",
    #         description    = 'Which network architecture to use.')
    #
    #     # Here, we do a switch on the networks allowed:
    #     yolo.YOLOFlags().build_parser(network_parser)


    def make_trainer(self):

        if self.args.run.distributed:
            from src.utils import distributed_trainer

            self.trainer = distributed_trainer.distributed_trainer(self.args)
        else:
            from src.utils import trainercore
            self.trainer = trainercore.trainercore(self.args)


    def __str__(self):

        s = "\n\n-- CONFIG --\n"
        substring = s +  self.dictionary_to_str(self.args)

        return substring

    def stop(self):
        # if not self.args.distributed or :
        self.trainer.stop()




    def dictionary_to_str(self, in_dict, indentation = 0):
        substr = ""
        for key in sorted(in_dict.keys()):
            if type(in_dict[key]) == DictConfig or type(in_dict[key]) == dict:
                s = "{none:{fill1}{align1}{width1}}{key}: \n".format(
                        none="", fill1=" ", align1="<", width1=indentation, key=key
                    )
                substr += s + self.dictionary_to_str(in_dict[key], indentation=indentation+2)
            else:
                if hasattr(in_dict[key], "name"): attr = in_dict[key].name
                else: attr = in_dict[key]
                s = '{none:{fill1}{align1}{width1}}{message:{fill2}{align2}{width2}}: {attr}\n'.format(
                   none= "",
                   fill1=" ",
                   align1="<",
                   width1=indentation,
                   message=key,
                   fill2='.',
                   align2='<',
                   width2=30-indentation,
                   attr = attr,
                )
                substr += s
        return substr

# @hydra.main(config_path="../src/config", config_name="config")
# def main(cfg : OmegaConf) -> None:
#     s = VertexID(cfg)
#     s.stop()

@hydra.main(config_path="../src/config", config_name="config")
def main(cfg : OmegaConf) -> None:

    # main()
    s = VertexID(cfg)
    s.stop()


if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += ['hydra.run.dir=.', 'hydra/job_logging=disabled']

    # if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    #
    #     # nodefile = os.environ['COBALT_NODEFILE']
    #     # n_nodes = len(open(nodefile, "r").read().split())
    #
    #     target_gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(target_gpu % 8)
    #     # logger.info('Setting CUDA_VISIBLE_DEVICES to', os.environ['CUDA_VISIBLE_DEVICES'])
    main()
