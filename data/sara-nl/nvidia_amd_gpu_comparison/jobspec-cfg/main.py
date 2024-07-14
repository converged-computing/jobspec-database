import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import time
import torch
import pprint as pp
import random
import numpy as np
import json
from torch.cuda.amp import GradScaler
from tensorboardX import SummaryWriter
from options import get_options
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from dataset import SyntheticDataDataset
from train import train_epoch
from utils import get_model


def main(opts):
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    np.random.seed(opts.seed)

    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.output_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    train_dataset = SyntheticDataDataset(opts.img_size, length=opts.num_images_train)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, pin_memory=True)

    model = get_model(opts)

    if torch.cuda.device_count() > 1 and len(opts.cuda_devices) > 1:
        model = nn.DataParallel(model, device_ids=opts.cuda_devices)
    model = model.to(opts.device)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=opts.eps,
                           weight_decay=opts.weight_decay, amsgrad=False)

    criterion = nn.CrossEntropyLoss()

    # Define the GradScaler object
    scaler = GradScaler() if opts.mixed_precision else None

    tb_logger = SummaryWriter(opts.output_dir, flush_secs=5)

    for epoch in range(opts.epochs):
        #########################
        #  Train and validate   #
        #########################

        t_start = time.time()

        images_per_second = train_epoch(model,
                                        criterion,
                                        optimizer,
                                        train_loader,
                                        opts,
                                        scaler=scaler)

        #########################
        #    Write to logger    #
        #########################

        # Do not log the first epoch, because of initialization overhead
        if epoch >= opts.skip_epoch:
            tb_logger.add_scalar('Images/second', images_per_second, epoch)

    tb_logger.flush()


if __name__ == '__main__':
    opts = get_options()
    main(opts)
