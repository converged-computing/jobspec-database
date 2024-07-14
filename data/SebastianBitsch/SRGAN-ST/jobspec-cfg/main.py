# This file is meant to be used in conjunction with the DTUs HPC envirnmonment.
# Specifically for batch jobs, i.e. running multiple experiments with different
# parameters at once. 
# If you dont want to run multiple experiments at once, it is easier to just run train.py,
# test.py or warmup.py with the required parameters instead. 
# See README.md for details on that.
# 
# If you want to run multiple experiments at once or both warm, train and test in the
# same job:
# An experiment can be defined by creating a function that takes a job-index, a 
# config object and modifies the wanted parameters on the config depending on the job-index
# then return the config object and call either warmup, train or test with the config.
# See warmup_gan() for a simple example.

import os

from torch import nn

from config import Config
from loss import BestBuddyLoss, GramLoss, PatchwiseStructureTensorLoss, StructureTensorLoss, ContentLossDiscriminator, ContentLossVGG

from train import train
from warmup import warmup
from validate import test


def get_jobindex(fallback:int = 0) -> int:
    """Get the job-index set in bash. This is mostly for array jobs where multiple models are trained in parallel"""
    num = os.getenv('job_index')
    return int(num) if num else fallback


def warmup_gan(config: Config, epochs:int = 5) -> Config:
    """ Warmup the generator / train srresnet """
    config.EXP.N_EPOCHS = epochs
    config.EXP.NAME = f"resnet{epochs}"
    config.G_CHECKPOINT_INTERVAL = 5
    return config


def my_experiment(config: Config, i: int) -> Config:
    config.EXP.NAME = "my-exp-name"
    
    # Change config parameters to suit experiment(s)
    ...
    
    return config


if __name__ == "__main__":

    # Get job-index from bash, if the job is not an array it will be zero
    job_index = get_jobindex()
    print(f"Running job: {job_index}")

    config = Config()

    config = my_experiment(config, job_index)

    # Train and validate the experiment given by the config file
    # warmup(config = config)
    train(config = config)

    test(config = config, save_images = True)

    print(f"Finished job: {job_index}")
