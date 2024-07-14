'''Joint training of a mention detection and entity linking component'''
import argparse
import json
import numpy as np
import torch
import tomli
from os import mkdir
from dataclasses import dataclass
from os.path import isfile, isdir, join
from typing import List, Callable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.samplers import MPerClassSampler
from sapiens.cnn import ResCNN, ResCNNConfig
from sapiens.loss import OntologicalTripletLoss
from sapiens.utils import Dotdict, EarlyStop, RetrievalEvaluator
from sapiens.data import Dataset, context_collate_fn, generate_negs
from sapiens.utils import get_ner_labels, reformat_mention


# --- Train & validation steps -----------------------------


def train(
        model, optimizer, scheduler, miner, 
        loss_func: Callable, 
        inputs: list, 
        el_labels: torch.Tensor,
        word_id_labels: list
    ) -> torch.Tensor:
    '''Train step 
    ---
    returns training loss
    '''
    model.train()
    optimizer.zero_grad()

    # mention embeddings (EL)
    inputs = reformat_mention(inputs, word_id_labels)
    embeddings = model(inputs)

    # evaluate combined loss and optimize
    loss = loss_func(embeddings, el_labels)
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss


def validate(
        model, optimizer, 
        loss_func: Callable, 
        val_inputs: list, 
        val_el_labels: torch.Tensor,
        word_id_labels: list 
    ) -> torch.Tensor:
    '''Validation step
    ---
    returns validation loss
    '''
    model.eval()
    with torch.no_grad():

        # validate EL
        val_inputs = reformat_mention(val_inputs, word_id_labels)
        val_embeds = model(val_inputs)
        val_loss = loss_func(val_embeds, val_el_labels)

    return val_loss


# --- Utility classes -------------------------------------


def verify_args(args, config):
    '''verifies logging and checkpoint directories'''
    # assertions
    assert config["TRAIN"]["loss"] in ["multisimilarityloss", "ontologicalloss"]
    assert config["TRAIN"]["optim"] in ["Adam", "SGD"]

    # logging directory for tensorboard
    i = 0
    if not isdir(args.logdir):
        mkdir(args.logdir)
        mkdir(join(args.logdir, f"run{str(i)}"))
    else:
        while isdir(join(args.logdir,f"run{str(i)}")):
            i += 1
        mkdir(join(args.logdir, f"run{str(i)}"))

    # save training config in logging directory
    tc = config["TRAIN"]
    with open(join(args.logdir, f"run{str(i)}", "config.info"), "w") as f:
        for key,val in tc.items():
            f.write(f"{key}: {val}\n")

    # checkpoint directory
    j = 0
    if not isdir(args.checkpoint):
        mkdir(args.checkpoint)
        mkdir(join(args.checkpoint, f"run{str(j)}"))
    else:
        while isdir(join(args.checkpoint,f"run{str(j)}")):
            j += 1
        mkdir(join(args.checkpoint, f"run{str(j)}"))

    # define vars
    logdir = join(args.logdir, f"run{str(i)}") 
    checkpointdir = join(args.checkpoint, f"run{str(j)}")

    return logdir, checkpointdir


def print_config(config: dict):
    '''Prints configuration parameters'''
    print("[DATA CONFIG]")
    for key,val in config["DATA"].items():
        print(f"{key} = {val}", flush=True)
    print("")
    print("[TRAIN CONFIG]")
    for key,val in config["TRAIN"].items():
        print(f"{key} = {val}", flush=True)
    print("")
    print("[MODEL CONFIG]")
    for key,val in config["MODEL"].items():
        print(f"{key} = {val}", flush=True)


def save_model_checkpoint(
    epoch, model, optimizer, loss, cp_path
    ):
    '''Saves a checkpoint of model'''
    model_id = f"checkpoint_e{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"{join(cp_path, model_id)}")


def get_validation_sample(subset):
    '''Returns a validation inputs and labels in correct format for model'''
    if isinstance(subset[0][0], list):
        val_ins = [i[0] for i in subset]
    else:
        val_ins = [[i[0]] for i in  subset]
    val_el_labs = torch.tensor([i[1] for i in subset])
    val_ner_labs = [i[2] for i in subset]

    return val_ins, val_el_labs, val_ner_labs


# --- Main ------------------------------------------------


def main(config: dict, cp_path: str, logdir: str, verbose:bool):
    '''Trains ResCNN model'''
    #--- SETUP TRAIN --------------------------------------

    # config
    if verbose: print_config(config)
    dc = config["DATA"]
    tc = config["TRAIN"]
    mc = config["MODEL"]

    # load train, val
    dataset = Dataset(dc["trainpath"])
    if tc["loss"] == "ontologicalloss":
        valset = Dataset(dc["valpath"])[:512]
    else:
        valset = Dataset(dc["valpath"])
    val_ins, val_el_labs, val_word_id_labs = get_validation_sample(valset)

    # init model
    model = ResCNN(ResCNNConfig().load(mc))
    if verbose: print(f"device: {model.device}\n---", flush=True)

    # entity linking (EL) loss setup
    if tc["loss"] == "multisimilarityloss":
        loss_func = MultiSimilarityLoss(
            alpha = tc["alpha"],
            beta = tc["beta"],
            base = tc["base"]
        )
        miner = MultiSimilarityMiner(
            epsilon = tc["epsilon"]
        )
    else: # ontologicalloss
        miner = MultiSimilarityMiner(
            epsilon = tc["epsilon"]
        )
        loss_func = OntologicalTripletLoss(
            ontopath = dc["ontopath"],
            margin = tuple(tc["margin"])
        )

    # optimizer setup
    if tc["optim"] == "Adam":
        optimizer = Adam(
            model.parameters(),
            lr = tc["lr"],
            weight_decay = tc["wdecay"]
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr = tc["lr"],
            momentum = tc["momentum"]
        )
        
    # training setup
    sampler = MPerClassSampler(
        labels = dataset.labels(),
        m = tc["m"], 
        length_before_new_iter = len(dataset)
    )
    loader = DataLoader(
        dataset,
        batch_size = tc["bsize"],
        sampler = sampler,
        drop_last = True,
        pin_memory = True,
        collate_fn = context_collate_fn
    )
    scheduler = LinearLR(
        optimizer,
        start_factor = 1.0,
        end_factor = tc["end_factor"],
        total_iters = tc["epochs"] * len(loader)
    )
    early_stop = EarlyStop(tc["tolerance"], tc["minlen"])
    if tc["evaluate"]: 
        evaluator = RetrievalEvaluator(dc["ontojsonpath"], dc["prefix"])

    #--- TRAIN LOOP ---------------------------------------

    for epoch in range(tc["epochs"]):
        if early_stop.stop: break
        for batchidx, (inputs, el_labels, word_id_labels) in enumerate(loader):
            # train step
            loss = train(
                model, optimizer, scheduler, miner, 
                loss_func, inputs, el_labels,word_id_labels 
            )

            # validate step
            val_loss = validate(
                model, optimizer, loss_func, 
                val_ins, val_el_labs, val_word_id_labs
            )

            # logging to stdout
            if verbose and (batchidx % 2 == 0):
                print(f"e{epoch}b{batchidx}|train:{loss.item():.4f}|", end="")
                print(f"val:{val_loss.item():.4f}", flush=True)

            # early stopping
            if early_stop(val_loss.item()):
                if verbose: print("early stopping.")
                break
            else: early_stop.losses.append(val_loss.item())

            # slow training save
            #if batchidx%5==0:
                #save_model_checkpoint(epoch, model, optimizer, loss, cp_path)

        #--- EPOCH evaluation -----------------------------
        model.eval()
        if tc["evaluate"] and ((1+epoch) % tc["epoch_iter"] == 0):
            if verbose: 
                print("---")
                print("computing performance...", flush=True)
            evaluator.load(model, True)

            # EL evaluation
            acc1, acck = evaluator(val_ins, val_el_labs)
            if verbose:
                print(f"e{epoch}|el_acc@1:{acc1}")
                print(f"e{epoch}|el_acc@{tc['k']}:{acck}")
                print("---", flush=True)

            # NER evaluation
            
            # epoch checkpoint
            save_model_checkpoint(epoch, model, optimizer, loss, cp_path)

    # early stopping checkpoint
    save_model_checkpoint(epoch, model, optimizer, loss, cp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configpath",
        default="configs/train_config.toml",
        help="toml config for model training"
    )
    parser.add_argument(
        "--checkpoint",
        default="resources/checkpoints",
        help="path to dir where model checkpoints are saved"
    )
    parser.add_argument(
        "--logdir",
        default="resources/logdir",
        help="path to dir where training logs are saved"
    )
    parser.add_argument(
        "--verbose",
        default=True
    )
    args = parser.parse_args()
    config = tomli.load(open(args.configpath, "rb"))
    logdir, checkpointdir = verify_args(args, config)
    main(config, checkpointdir, logdir, args.verbose)
