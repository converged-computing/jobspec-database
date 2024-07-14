from  inceptionv3_lightning import *
import ssl
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import sys
import time


from retina_dataset import RetinaDataset

ssl._create_default_https_context = ssl._create_unverified_context


# train.py
def main(args):
    batch_size =  int(args.batch_size if(args and  args.batch_size) else 32)
    num_devices = int(args.num_devices if(args and args.num_devices) else 1)
    num_nodes = int(args.num_nodes if(args and args.num_nodes) else 2)

    model = InceptionV3LightningModel(args)
    train_dataset = RetinaDataset(total=200)
    train, val = random_split(train_dataset, [180, 20])
    
    train_loader = DataLoader(dataset=train, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, num_workers=4)

    start_time = time.time()
    print(f'starting training at time {start_time}', file = sys.stdout, flush=True)
    if(torch.cuda.is_available() and num_devices > 0):
        print(f'using gpu accelerator! num_devices={num_devices}, num_nodes={num_nodes}', file = sys.stdout, flush=True)
        trainer = Trainer(
        accelerator="gpu",
        gpus=num_devices,
        num_nodes=num_nodes,
        strategy="ddp")
        with trainer.profiler.profile("training_step"):
            trainer.fit(model, train_loader, val_loader)
    else: 
        print('using plain ole cpu and 1 node!', file = sys.stdout, flush=True)
        trainer = Trainer(
            # num_nodes=1, s
            max_epochs=15,
            strategy="ddp")
        with trainer.profiler.profile("training_step"):
            trainer.fit(model, train_loader, val_loader)
    execution_time = time.time() - start_time
    print("--- %s seconds ---" % (execution_time), file = sys.stdout, flush=True)
    print(f'completed training at time', file = sys.stdout, flush=True)
    print("--- %s seconds ---" % (execution_time),  file = sys.stdout, flush=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=None)
    parser.add_argument("--num_devices", default=None)
    parser.add_argument("--num_nodes", default=None)
    args = parser.parse_args()

    main(args)