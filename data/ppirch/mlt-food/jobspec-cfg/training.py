import logging
from argparse import ArgumentParser

from loss import UncertaintyLoss, AutomaticWeightedLoss
from foodnet import ClassifyNet, RegressNet, SharedNet, ConcatNet
from utils.datasets import FoodImagesDataset

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def add_model_args(parser: ArgumentParser):
    """
    Adds model arguments to the parser

    Args:
        parser (ArgumentParser): _description_
    """
    parser.add_argument(
        "--model_type",
        default="classify",
        choices=["classify", "regress", "shared", "concat"],
        help="model type",
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=["resnet50", "resnet101", "resnet152"],
        help="model backbone",
    )


def add_training_args(parser: ArgumentParser):
    """
    Adds training arguments to the parser

    Args:
        parser (ArgumentParser): _description_
    """
    parser.add_argument(
        "--loss_type",
        default="default",
        choices=["default", "uncertainty", "automatic"],
        help="loss type",
    )
    parser.add_argument(
        "--seed",
        default=677,
        type=int,
        help="seed",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="epochs",
    )


def main():
    parser = ArgumentParser()
    add_model_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Args: %s", args)

    if args.model_type in ["shared", "concat"] and args.loss_type == "default":
        raise ValueError(
            "Invalid loss type. Default is not supported for shared/concat"
        )
    if args.model_type in ["classify", "regress"] and args.loss_type != "default":
        raise ValueError(
            "Invalid loss type. Only default is supported for classify/regress"
        )

    torch.manual_seed(args.seed)

    logger.info("Loading dataset")
    DATA_DIR = "./regression"
    df = pd.read_csv(f"{DATA_DIR}/csv/train.csv")
    le = LabelEncoder()
    le.fit(df["food"])
    n_classes = len(le.classes_)

    if args.model_type == "classify":
        model = ClassifyNet(backbone=args.backbone, n_classes=n_classes)
    elif args.model_type == "regress":
        model = RegressNet(backbone=args.backbone)
    elif args.model_type == "shared":
        model = SharedNet(backbone=args.backbone, n_classes=n_classes)
    elif args.model_type == "concat":
        model = ConcatNet(backbone=args.backbone, n_classes=n_classes)
    else:
        raise ValueError("Invalid model type")

    logger.info("MODEL: %s", model)

    ce_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()
    if args.loss_type == "uncertainty":
        uncertainty_loss = UncertaintyLoss()
    if args.loss_type == "automatic":
        automatic_weighted_loss = AutomaticWeightedLoss(2)

    train_datasets = FoodImagesDataset(
        csv_file=f"{DATA_DIR}/csv/train.csv",
        img_dir=f"{DATA_DIR}/train",
        target_transform=le.transform,
    )

    train_dataloader = DataLoader(
        train_datasets, batch_size=args.batch_size, shuffle=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("DEVICE: %s", device)
    model = model.to(device)

    ce_loss = ce_loss.to(device)
    l1_loss = l1_loss.to(device)
    if args.loss_type == "uncertainty":
        uncertainty_loss = uncertainty_loss.to(device)
    if args.loss_type == "automatic":
        automatic_weighted_loss = automatic_weighted_loss.to(device)

    if args.loss_type == "uncertainty":
        optimizer = optim.SGD(
            [
                {"params": model.parameters()},
                {"params": uncertainty_loss.parameters()},
            ],
            lr=0.001,
            momentum=0.9,
        )
    elif args.loss_type == "automatic":
        optimizer = optim.SGD(
            [
                {"params": model.parameters()},
                {"params": automatic_weighted_loss.parameters()},
            ],
            lr=0.001,
            momentum=0.9,
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    logger.info("Start training")
    model.train()
    for epoch in range(0, args.epochs):
        logger.info(f"Epoch: {epoch + 1} / {args.epochs}")
        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, weight, labels = data
            weight, labels = weight.long(), labels.long()
            inputs, weight, labels = (
                inputs.to(device, non_blocking=True),
                weight.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(inputs)

            if args.loss_type == "default" and args.model_type == "classify":
                loss = ce_loss(outputs, labels)
            elif args.loss_type == "default" and args.model_type == "regress":
                loss = l1_loss(outputs, weight.unsqueeze(1))
            elif args.loss_type == "uncertainty":
                reg_head, class_head = outputs
                reg_loss = l1_loss(reg_head, weight.unsqueeze(1))
                cls_loss = ce_loss(class_head, labels)
                loss = uncertainty_loss(reg_loss, cls_loss)
            elif args.loss_type == "automatic":
                reg_head, class_head = outputs
                reg_loss = l1_loss(reg_head, weight.unsqueeze(1))
                cls_loss = ce_loss(class_head, labels)
                loss = automatic_weighted_loss(reg_loss, cls_loss)
            else:
                raise ValueError("Invalid loss type")

            # backward + optimize
            loss.backward()
            optimizer.step()

        logger.info("Finished Training")
        torch.save(
            model.state_dict(),
            f"./models/{args.model_type}_{args.backbone}_{args.loss_type}_{epoch+1}_{args.seed}_.pth",  # noqa
        )
        logger.info("Finished Saving\n")


if __name__ == "__main__":
    main()
