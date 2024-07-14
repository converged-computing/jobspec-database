import os
import pprint
pp = pprint.PrettyPrinter()
from datetime import datetime
import torch
from model.train import Model
from configs import opts

if __name__ == '__main__':

    if opts.phase == "train":
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        opts.log_dir = os.path.join(opts.log_dir,current_time)
        if not os.path.exists(opts.log_dir):
            os.makedirs(opts.log_dir)
    #opts.log_dir = "log/20210124-0059"
    print('checkpoints:', opts.log_dir)
    model = Model(opts)
    model.train()