from typing import Dict, Union
import torch
from torch.optim.lr_scheduler import StepLR
from gcl_sem.datasets import (
    MslsCitiesImageMaskPair,
    MslsCitiesImageMaskKey,
    MslsCitiesImageActPair,
    MslsCitiesImageActKey,
    MslsCitiesImagePair,
    MslsCitiesImageKey,
)
from gcl_sem.networks import SiameseNet, ContrastiveLoss, GeM
from gcl_sem.validation import extract_features, find_top_k

from gcl_sem.libs.mapillary_sls.mapillary_sls.utils.eval import eval
from gcl_sem.libs.mapillary_sls.mapillary_sls.datasets.msls import MSLS

import json
import os
import numpy as np
import argparse

from torch.utils.tensorboard import SummaryWriter


def train_loop(
    model,
    train_loader,
    loss_fn,
    optimizer,
    epoch,
    writer: SummaryWriter = None,
):
    size = len(train_loader.dataset)

    model.train()

    for batch, (x0, x1, label) in enumerate(train_loader):
        optimizer.zero_grad()

        x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()

        # Compute prediction and loss
        y0, y1 = model(x0, x1)
        loss = loss_fn(y0, y1, label)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if writer is not None:
            step = epoch * len(train_loader) + batch
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], step)

        if (batch + 1) % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(x0)
            # add percentage of completion
            print(
                f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] {current / size * 100} %",
                flush=True,
            )


def key_from_path(path: str) -> str:
    return path.split("/")[-1].split(".")[0]


def get_metrics(
    model, msls_dataset: MSLS, query_loader, database_loader, top_k_file: str = None
) -> Dict[str, float]:
    query_keys = [
        key_from_path(msls_dataset.qImages[q_idx]) for q_idx in msls_dataset.qIdx
    ]

    positive_keys = [
        [key_from_path(msls_dataset.dbImages[pos_idx]) for pos_idx in pos_indexes]
        for pos_indexes in msls_dataset.pIdx
    ]

    single_model = model.single_net

    query_features = extract_features(query_loader, single_model)
    database_features = extract_features(database_loader, single_model)

    predictions_dict = find_top_k(database_features, query_features, k=25)

    # only keep top_k matches where the query_key is in query_keys
    predictions_dict = {
        query_key: query_predictions
        for query_key, query_predictions in predictions_dict.items()
        if query_key in query_keys
    }

    # turn top_k dict into a list where the first element is the query key and the rest are the top k matches
    predictions = [
        [query_key] + list(matches) for query_key, matches in predictions_dict.items()
    ]

    # convert to numpy array
    predictions = np.array(predictions)

    if top_k_file is not None:
        np.savetxt(top_k_file, predictions, fmt="%s")

    # evaluate the top k matches
    metrics = eval(query_keys, positive_keys, predictions)

    return metrics


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for x0, x1, label in dataloader:
            x0 = x0.cuda()
            x1 = x1.cuda()
            label = label.cuda()

            y0, y1 = model(x0, x1)

            test_loss += loss_fn(y0, y1, label).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


def create_model(backbone):
    model = SiameseNet(backbone).cuda()
    return model


def create_backbone(weights_path, in_channels):
    backbone = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")

    if weights_path is not None:
        backbone.load_state_dict(torch.load(weights_path))
        print("Loaded weights from", weights_path)

    # replace the first layer with a 4-channel layer
    backbone.conv1 = torch.nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # replace the average pooling layer with a GeM layer
    backbone.avgpool = GeM()

    # disable the fully connected layer
    backbone.fc = torch.nn.Identity()
    return backbone


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--msls_root", type=str, required=True)
    parser.add_argument("--masks_root", type=str)
    parser.add_argument("--activations_root", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--resnext_weights", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--first_layer_lr", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--name", type=str, default="exp")

    args = parser.parse_args()

    msls_root = args.msls_root

    if args.masks_root is not None and args.activations_root is not None:
        raise ValueError("Only one of masks_root or activations_root can be specified")

    masks_root = args.masks_root
    activations_root = args.activations_root

    batch_size = args.batch_size
    epochs = args.epochs
    steps = args.steps
    val_interval = args.val_interval
    resnext_weights = args.resnext_weights
    save_interval = args.save_interval
    resume_from = args.resume_from
    first_layer_lr = args.first_layer_lr
    lr = args.lr
    exp_name = args.name

    loss_fn = ContrastiveLoss().cuda()

    if activations_root is not None:
        in_channels = 3 + 19
    elif masks_root is not None:
        in_channels = 3 + 1
    else:
        in_channels = 3

    print(f"Creating model with {in_channels} input channels")

    backbone = create_backbone(resnext_weights, in_channels)
    model = create_model(backbone).cuda()

    first_layer_params = list(model.single_net.backbone.conv1.parameters())

    other_params = [
        param for name, param in model.named_parameters() if "conv1" not in name
    ]

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # for first 10 epochs, use learning rate of 0.1 for first and 0 for other layers
    # for the rest of the epochs, use learning rate of 0.01 for all layers

    optimizer = torch.optim.SGD(
        [
            {"params": first_layer_params, "lr": first_layer_lr},
            {"params": other_params, "lr": lr},
        ],
        lr=lr,
        momentum=0.9,
    )

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        print("Resuming from", resume_from)

    msls_dataset = MSLS(msls_root, "", mode="val", posDistThr=25)

    train_cities = ["amman", "boston", "london", "manila", "zurich"]
    val_cities = ["cph", "sf"]

    if masks_root is not None:
        print("Using masks")
        train_dataset = MslsCitiesImageMaskPair(msls_root, masks_root, train_cities)
        val_query_dataset = MslsCitiesImageMaskKey(
            msls_root, masks_root, val_cities, "query"
        )
        val_database_dataset = MslsCitiesImageMaskKey(
            msls_root, masks_root, val_cities, "database"
        )
    elif activations_root is not None:
        print("Using activations")
        train_dataset = MslsCitiesImageActPair(
            msls_root, activations_root, train_cities
        )
        val_query_dataset = MslsCitiesImageActKey(
            msls_root, activations_root, val_cities, "query"
        )
        val_database_dataset = MslsCitiesImageActKey(
            msls_root, activations_root, val_cities, "database"
        )
    else:
        print("Using images")
        train_dataset = MslsCitiesImagePair(msls_root, train_cities)

        val_query_dataset = MslsCitiesImageKey(msls_root, val_cities, "query")
        val_database_dataset = MslsCitiesImageKey(msls_root, val_cities, "database")

    val_query_loader = torch.utils.data.DataLoader(
        val_query_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    val_database_loader = torch.utils.data.DataLoader(
        val_database_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    runs_folder = "runs"
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    exp_folder = os.path.join(runs_folder, exp_name)

    i = 0
    run_folder = os.path.join(exp_folder, f"run_{i}")
    while os.path.exists(run_folder):
        i += 1
        run_folder = os.path.join(exp_folder, f"run_{i}")

    os.makedirs(run_folder)

    # save the command line arguments to a file

    with open(os.path.join(run_folder, "args.txt"), "w") as f:
        f.write(str(args))

    writer = SummaryWriter(run_folder)

    for t in range(epochs):
        # take a random subset of the training dataset
        train_dataset = torch.utils.data.Subset(
            train_dataset, np.random.choice(len(train_dataset), batch_size * steps)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(model, train_loader, loss_fn, optimizer, t, writer)

        scheduler.step()

        if t % save_interval == 0:
            checkpoint_path = f"{run_folder}/checkpoint_{t}.pth"
            save_checkpoint(model, optimizer, scheduler, checkpoint_path)

        if t % val_interval == 0:
            top_k_file = f"{run_folder}/top_k_{t}.txt"

            metrics = get_metrics(
                model,
                msls_dataset,
                val_query_loader,
                val_database_loader,
                top_k_file,
            )
            metrics["epoch"] = t
            metrics["lr"] = optimizer.param_groups[0]["lr"]

            print("Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

            # save metrics to file
            with open(f"{run_folder}/metrics_{t}.json", "w") as f:
                json.dump(metrics, f)

    print("Done!")


def save_checkpoint(model, optimizer, scheduler, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)


# def create_val_loaders(msls_root, masks_root, batch_size):
#     val_query_dataset = MslsCitiesImageMaskKey(
#         msls_root, masks_root, ["cph", "sf"], "query.json"
#     )

#     val_database_dataset = MslsCitiesImageMaskKey(
#         msls_root, masks_root, ["cph", "sf"], "database.json"
#     )

#     val_query_loader = torch.utils.data.DataLoader(
#         val_query_dataset, batch_size=batch_size, shuffle=False, num_workers=4
#     )

#     val_database_loader = torch.utils.data.DataLoader(
#         val_database_dataset, batch_size=batch_size, shuffle=False, num_workers=4
#     )

#     return val_query_loader, val_database_loader


if __name__ == "__main__":
    main()
