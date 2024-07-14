import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast
import time
import pdb

from utils import move_to


def train_epoch(model, criterion, optimizer, train_loader, opts, scaler=None):
    # Put model in train mode
    model.train()

    start_time = None
    num_images_processed = 0

    for batch_id, batch in enumerate(tqdm(train_loader)):

        # Skip the first N batches due to data loader initialization
        if batch_id == opts.skip_batches:
            start_time = time.time()

        inputs, label = move_to(batch, opts.device)
        label = label.long()
        optimizer.zero_grad()

        if opts.mixed_precision:
            with autocast():
                prediction = model(inputs)
                loss = criterion(prediction, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            prediction = model(inputs)
            loss = criterion(prediction, label)

            loss.backward()
            optimizer.step()

        if start_time is not None:
            num_images_processed += inputs.shape[0]

    end_time = time.time()

    return num_images_processed / (end_time - start_time)
