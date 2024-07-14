#!/usr/bin/env python3
"""Kaggle RSNA Classification experiment source code."""

import numpy as np
import os
import torch
import torchvision
import pandas as pd
import pydicom

from albumentations import Compose, Resize, HorizontalFlip, ShiftScaleRotate
from albumentations.pytorch import ToTensor
from argparse import ArgumentParser
from apex import amp
from glob import glob
from matplotlib import pyplot as plt
from time import time
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class AverageMeter(object):
    """Compute and store the average, standard deviation, and current value"""

    def __init__(self, name="Meter", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sqsum = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sqsum += (val ** 2) * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = ((self.sqsum / self.count) - (self.avg * self.avg)) ** 0.5

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + \
            "} (AVG {avg" + self.fmt + "}, STD {std" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class IntracranialDataset(torch.utils.data.Dataset):
    """Dataset Class for RSNA Kaggle Stage 2 DICOM images"""
    def __init__(self, dcm_path, label_df=None, transform=None, n_classes=6):
        self.dcm_path = dcm_path
        self.label_df = label_df
        self.transform = transform
        self.n_classes = n_classes

        if self.label_df is None:
            self.data = sorted(glob(os.path.join(dcm_path, "*.dcm")))
        else:
            self.data = label_df


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = None
        target = torch.FloatTensor([-1.0] * self.n_classes)

        try:
            if self.label_df is not None:
                base = self.data.iloc[idx]
                sop_uid = base.name
                target = base.to_list()
                # assert len(target) == n_classes
                target = torch.FloatTensor(target)
                dcm_path = os.path.join(self.dcm_path, "{}.dcm".format(sop_uid))
            else:
                dcm_path = self.data[idx]

            dcm = pydicom.dcmread(dcm_path)
            img = IntracranialDataset.bsb_window_image(dcm)

            if self.transform is not None:
                img = self.transform(image=img)["image"]
        except:
            pass

        return img, target

    @staticmethod
    def correct_dicom(dcm):
        x = dcm.pixel_array + 1000
        px_mode = 4096
        x[x>=px_mode] = x[x>=px_mode] - px_mode
        dcm.PixelData = x.tobytes()
        dcm.RescaleIntercept = -1000

    @staticmethod
    def window_image(dcm, window_center, window_width):
        if dcm.BitsStored == 12 and dcm.PixelRepresentation == 0 and \
            int(dcm.RescaleIntercept) > -100:
            IntracranialDataset.correct_dicom(dcm)
        img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)
        return img

    @staticmethod
    def bsb_window_image(dcm):
        # Following https://radiopaedia.org/articles/ct-head-an-approach?lang=gb
        # https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing
        # https://www.kaggle.com/reppic/gradient-sigmoid-windowing

        brain_img = IntracranialDataset.window_image(dcm, 40, 80) # brain matter; center 40, width 80
        subdural_img = IntracranialDataset.window_image(dcm, 80, 200) # blood/subdural; center 50-100, width 130-300
        soft_img = IntracranialDataset.window_image(dcm, 40, 380) # soft tissues; center 20-60, width 350-400
        # bone; center 600, width 2800

        brain_img = (brain_img - 0) / 80
        subdural_img = (subdural_img - (-20)) / 200
        soft_img = (soft_img - (-150)) / 380
        bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

        return bsb_img


def filter_collate_fn(batch):
    return default_collate([b for b in batch if b[0] is not None])


def main():
    # Argument Parser
    parser = ArgumentParser(
        description="RSNA Intracranial Hemorrhage Detection")
    ds_group = parser.add_argument_group("Dataset")
    ds_group.add_argument(
        "--rsna-base", default="./data/rsna-intracranial-hemorrhage-detection/", type=str,
        help="path to RSNA Kaggle dataset files (containing stage_2_train.csv, stage_2_train/test)"
    )
    ds_group.add_argument("--resize-dim", default=224, type=int,
                          help="image width/height size for training/evaluation (default: 224)")
    ds_group.add_argument("--n-classes", default=6, type=int,
                          help="number of labels to predict (default: 6)")
    exp_group = parser.add_argument_group("Experiment")
    exp_group.add_argument("--model", default="resnet50", type=str,
                           help="Model architecture to use for training (default: resnet50)")
    exp_group.add_argument("--epochs", default=5, type=int,
                           help="Number of epochs to train for (default: 5)")
    exp_group.add_argument("--batch-size", default=128, type=int,
                           help="DataLoader batch size (default: 128)")
    exp_group.add_argument("--num-workers", default=16, type=int,
                           help="DataLoader number of workers (default: 16)")
    exp_group.add_argument("--test-batch-size", default=64, type=int,
                           help="DataLoader batch size (default: 64)")
    exp_group.add_argument("--test-num-workers", default=16, type=int,
                           help="DataLoader number of workers (default: 16)")
    exp_group.add_argument("--lr", default=2e-5, type=float,
                           help="Optimizer learning rate (default: 2e-5)")
    exp_group.add_argument("--val", default=0.05, type=float,
                           help="Validation dataset split from train (default: 0.05)")
    exp_group.add_argument("--checkpoint", default="model_checkpoint.pth", type=str,
                           help="Model checkpoint path (default: model_checkpoint.pth)")
    exp_group.add_argument("--submission", default="submission.csv", type=str,
                           help="Submission csv path (default: submission.csv)")
    exp_group.add_argument("--tb-log", default=None, type=str,
                           help="Tensorboard SummaryWriter log directory (default: None)")
    exp_group.add_argument("--resume", default=None, type=str,
                           help="Path to checkpoint to resume (default: None)")
    exp_group.add_argument("--apply-pos-weight", action="store_true",
                           help="When enabled, applies pos_weight = neg/pos for BCE loss")

    args = parser.parse_args()

    print(args)

    base_train_csv = os.path.join(args.rsna_base, "stage_2_train.csv")
    train_dcm_fp = os.path.join(args.rsna_base, "stage_2_train")
    base_test_csv = os.path.join(
        args.rsna_base, "stage_2_sample_submission.csv")
    test_dcm_fp = os.path.join(args.rsna_base, "stage_2_test")

    writer = SummaryWriter(log_dir=args.tb_log)
    print("Tensorboard logging to: ", writer.log_dir)
    writer.add_text("args", str(args))

    # load the original dataframe with two columns (ID, Label)
    base_train_df = pd.read_csv(base_train_csv)

    # Construct a new dataframe where the columns are
    # [ID,epidural,intraparenchymal,intraventricular,subarachnoid,subdural,any]
    def flatten_base_train_df(train_df):
        labels = ["ID", "epidural", "intraparenchymal", "intraventricular",
                  "subarachnoid", "subdural", "any"]
        f_df_dict = {label: [] for label in labels}

        with tqdm(train_df.itertuples(index=False), total=len(train_df), desc="Flattening csv") as t:
            for row in t:
                prefix, sop_uuid, label = row.ID.split("_")
                dcm_id = "_".join([prefix, sop_uuid])
                f_df_dict["ID"].append(dcm_id)
                f_df_dict[label].append(row.Label)
                for rest in labels:
                    if rest not in ["ID", label]:
                        f_df_dict[rest].append(0)

        return pd.DataFrame.from_dict(f_df_dict).groupby("ID").sum()

    train_df = flatten_base_train_df(base_train_df)

    # Instantiate the dataset instances
    train_dataset = IntracranialDataset(
        train_dcm_fp, label_df=train_df,
        transform=Compose([
            Resize(args.resize_dim, args.resize_dim),
            ShiftScaleRotate(rotate_limit=14),
            HorizontalFlip(),
            ToTensor()]),
        n_classes=args.n_classes)
    val_dataset = None

    if args.val > 0.0:
        val_len = int(len(train_dataset) * args.val)
        train_len = len(train_dataset) - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (train_len, val_len,))
    
    test_dataset = IntracranialDataset(
        test_dcm_fp, transform=Compose([Resize(args.resize_dim, args.resize_dim), ToTensor()]),
        n_classes=args.n_classes)

    # Instantiate the dataloader instances
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=filter_collate_fn)
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.test_num_workers)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.test_num_workers)

    # Instantiate the model
    if args.model == "resnet34":
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, args.n_classes)
    elif args.model == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, args.n_classes)
    elif args.model == "resnet101":
        model = torchvision.models.resnet101(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, args.n_classes)
    else:
        raise RuntimeError("Unsupported model: {}".format(args.model))

    if args.apply_pos_weight:
        total_examples = len(train_df)
        pos_examples = train_df.sum(axis=0).to_list()
        pos_weight = torch.FloatTensor([(total_examples - x) / x for x in pos_examples])
        if torch.cuda.is_available():
            pos_weight = pos_weight.cuda()
        print("Applying pos weight {}".format(pos_weight))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if torch.cuda.is_available():
        model = model.cuda()

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if args.resume is not None:
        print("Loading the state dict from {}".format(args.resume))
        model.load_state_dict(torch.load(args.resume))

    def train(epoch=-1):
        model.train()
        loss_meter = AverageMeter()
        datatime_meter = AverageMeter()
        batchtime_meter = AverageMeter()
        with tqdm(train_dataloader, total=len(train_dataloader), desc="Train Epoch {}".format(epoch)) as t:
            end = time()
            for step, (inputs, targets) in enumerate(t):
                b_start = time()

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                predictions = model(inputs)
                loss = criterion(predictions, targets)

                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                loss_meter.update(loss.item())
                datatime_meter.update(b_start - end)
                end = time()
                batchtime_meter.update(end - b_start)
                t.set_postfix_str(
                    "Loss: {loss:.3f} ({loss_avg:.3f}) | Data: {dtime:.2f}s ({dtime_avg:.2f}s) | Batch: {btime:.2f}s ({btime_avg:.2f}s)".format(
                        loss=loss_meter.val,
                        loss_avg=loss_meter.avg,
                        dtime=datatime_meter.val,
                        dtime_avg=datatime_meter.avg,
                        btime=batchtime_meter.val,
                        btime_avg=batchtime_meter.avg,
                    ))
                writer.add_scalar("Loss/Train", loss.item(),
                                  step + max(0, epoch) * len(train_dataloader))
    def val(epoch=-1):
        model.eval()
        loss_meter = AverageMeter()
        datatime_meter = AverageMeter()
        batchtime_meter = AverageMeter()
        with tqdm(val_dataloader, total=len(val_dataloader), desc="Val Epoch {}".format(epoch)) as t:
            end = time()
            for step, (inputs, targets) in enumerate(t):
                b_start = time()

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                predictions = model(inputs)
                loss = criterion(predictions, targets)

                loss_meter.update(loss.item())
                datatime_meter.update(b_start - end)
                end = time()
                batchtime_meter.update(end - b_start)
                t.set_postfix_str(
                    "Loss: {loss:.3f} ({loss_avg:.3f}) | Data: {dtime:.2f}s ({dtime_avg:.2f}s) | Batch: {btime:.2f}s ({btime_avg:.2f}s)".format(
                        loss=loss_meter.val,
                        loss_avg=loss_meter.avg,
                        dtime=datatime_meter.val,
                        dtime_avg=datatime_meter.avg,
                        btime=batchtime_meter.val,
                        btime_avg=batchtime_meter.avg,
                    ))
                writer.add_scalar("Loss/Validation", loss.item(),
                                  step + max(0, epoch) * len(train_dataloader))
    try:
        for epoch in range(args.epochs):
            train(epoch)
            if val_dataloader is not None:
                val(epoch)
    except Exception:
        import traceback
        print(traceback.format_exc())
        pass
    finally:
        torch.save(model.state_dict(), args.checkpoint)

    def inference():
        model.eval()
        test_pred = np.zeros((len(test_dataset) * args.n_classes, 1))

        with tqdm(test_dataloader, desc="Infer") as t:
            end = time()
            for step, (inputs, _) in enumerate(t):
                b_start = time()
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                predictions = model(inputs)
                bs = predictions.size(0)
                test_pred[(step*bs*args.n_classes): ((step+1)*bs*args.n_classes)] = torch.sigmoid(
                    predictions
                ).detach().cpu().reshape((bs*args.n_classes, 1))
        return test_pred

    test_predictions = inference()

    submission = pd.read_csv(base_test_csv)
    assert len(submission) == len(test_predictions)
    submission = submission.drop(columns=["Label"])
    submission["Label"] = pd.DataFrame(test_predictions)

    submission.to_csv(args.submission, index=False)
    # !kaggle competitions submit -c rsna-intracranial-hemorrhage-detection -f submission.csv -m "Submission"
    print(submission.head())


if __name__ == "__main__":
    main()
