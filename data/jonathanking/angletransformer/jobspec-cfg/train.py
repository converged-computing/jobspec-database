import argparse
import os
import sys

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
import torch.multiprocessing
from tqdm import tqdm
from tqdm.utils import _unicode, disp_len


torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, "/net/pulsar/home/koes/jok120/openfold/")

from openfold.config import config
from openfold.utils.loss import supervised_chi_loss
from openfold.model.structure_module import AngleResnet
from sidechainnet.examples.optim import NoamOpt

from angletransformer.data import ATFileDataset, collate_fn
from angletransformer.model import AngleTransformer


class ATModuleLit(pl.LightningModule):
    def __init__(
        self,
        train_dataset_dir,
        val_dataset_dir,
        c_s=384,
        c_hidden=256,
        no_blocks=2,
        no_angles=config.model.structure_module.no_angles,  # 7
        epsilon=config.globals.eps,
        dropout=0.1,
        d_ff=2048,
        no_heads=4,
        activation="relu",
        batch_size=1,
        num_workers=0,
        **kwargs,
    ):
        super().__init__()
        self.at = AngleTransformer(
            c_s=c_s,
            c_hidden=c_hidden,
            no_blocks=no_blocks,
            no_angles=no_angles,
            epsilon=epsilon,
            dropout=dropout,
            d_ff=d_ff,
            no_heads=no_heads,
            activation=activation,
            conv_encoder=kwargs.get("conv_encoder"),
        )
        self.loss = supervised_chi_loss
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Optimizer
        self.opt_lr = kwargs.get("opt_lr")
        self.opt_lr_scheduling = kwargs.get("opt_lr_scheduling")
        self.opt_name = kwargs.get("opt_name")
        self.opt_n_warmup_steps = kwargs.get("opt_n_warmup_steps")
        self.opt_noam_lr_factor = kwargs.get("opt_noam_lr_factor")
        self.opt_weight_decay = kwargs.get("opt_weight_decay")
        self.opt_patience = kwargs.get("opt_patience")
        self.opt_min_delta = kwargs.get("opt_min_delta")
        self.opt_lr_scheduling_metric = kwargs.get("opt_lr_scheduling_metric")

        self.chi_weight = kwargs.get("chi_weight")

    def forward(self, s, s_initial):
        return self.at(s, s_initial)

    def _standard_train_val_step(self, batch, batch_idx):
        s, s_initial = batch["s"][:, -1, ...].squeeze(1), batch["s_initial"].squeeze(1)
        unnorm_ang, ang = self(s, s_initial)
        loss = self.loss(
            angles_sin_cos=ang,
            unnormalized_angles_sin_cos=unnorm_ang,
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            chi_mask=batch["chi_mask"],
            chi_angles_sin_cos=batch["chi_angles_sin_cos"],
            chi_weight=self.chi_weight,
            angle_norm_weight=config.loss.supervised_chi.angle_norm_weight,
            eps=1e-6,
        )
        mode = "train" if self.training else "val"
        on_step = mode == "train"
        for loss_key, value in loss.items():
            if loss_key != "loss" and torch.isnan(value):
                continue
            self.log(
            f"{mode}/{loss_key}",
            value,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=on_step,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._standard_train_val_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._standard_train_val_step(batch, batch_idx)

    def configure_optimizers(self):
        """Prepare optimizer and schedulers.

        Args:
            optimizer (str): Name of optimizer ('adam', 'sgd')
            learning_rate (float): Learning rate for optimizer.
            weight_decay (bool, optional): Use optimizer weight decay. Defaults to True.

        Returns:
            dict: Pytorch Lightning dictionary with keys "optimizer" and "lr_scheduler".
        """
        # Setup default optimizer construction values
        if self.opt_lr_scheduling == "noam":
            lr = 0
            betas = (0.9, 0.98)
            eps = 1e-9
        else:
            lr = self.opt_lr
            betas = (0.9, 0.999)
            eps = 1e-8

        # Prepare optimizer
        if self.opt_name == "adam":
            opt = torch.optim.Adam(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=self.opt_weight_decay,
            )
        elif self.opt_name == "adamw":
            opt = torch.optim.AdamW(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=self.opt_weight_decay,
            )
        elif self.opt_name == "sgd":
            opt = torch.optim.SGD(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                weight_decay=self.opt_weight_decay,
                momentum=0.9,
            )

        # Prepare scheduler
        if self.opt_lr_scheduling == "noam" and self.opt_name in [
            "adam",
            "adamw",
        ]:
            opt = NoamOpt(
                model_size=self.at.c_hidden,
                warmup=self.opt_n_warmup_steps,
                optimizer=opt,
                factor=self.opt_noam_lr_factor,
            )
            sch = None
        elif self.opt_lr_scheduling == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                patience=self.opt_patience,
                verbose=True,
                threshold=self.opt_min_delta,
                mode="min" if "acc" not in self.opt_lr_scheduling_metric else "max",
            )
        else:
            sch = None

        d = {"optimizer": opt}
        if sch is not None:
            d["lr_scheduler"] = {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.opt_lr_scheduling_metric,
                "strict": True,
                "name": None,
            }

        return d

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            ATFileDataset(self.train_dataset_dir),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            ATFileDataset(self.val_dataset_dir),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


class AngleResnetLit(ATModuleLit):
    """An implementation of the baseline AlphaFold2 AngleResnet for Pytorch Lightning."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if kwargs.get("run_resnet_with_conv_encoder"):
            print("Running baseline Resnet with conv encoder.")

        self.at = AngleResnet(
            c_in=config.model.structure_module.c_s,
            c_hidden=config.model.structure_module.c_resnet,
            no_blocks=config.model.structure_module.no_resnet_blocks,
            no_angles=config.model.structure_module.no_angles,
            epsilon=config.model.structure_module.epsilon,
            conv_encoder=kwargs.get("run_resnet_with_conv_encoder"),
        )

        if not kwargs.get("skip_loading_resnet_weights"):
            # Load the associated weights
            print("Loading Resnet weights...", end=" ")
            sd = torch.load(
                "/net/pulsar/home/koes/jok120/openfold/openfold/resources/openfold_params/finetuning_5.pt"
            )
            sd = {"model." + k: v for k, v in sd.items()}
            new_sd = {
                k.replace("model.structure_module.angle_resnet.", ""): v
                for k, v in sd.items()
                if "model.structure_module.angle_resnet" in k
            }

            missing, unexpected = self.at.load_state_dict(new_sd, strict=not kwargs.get("run_resnet_with_conv_encoder"))
            print("done.")
            if missing:
                print("Missing keys:", missing)


def main(args):
    if args.is_sweep:
        default_config = {
            "train_data": "/scr/jok120/angletransformer/data/train/",
            "val_data": "/scr/jok120/angletransformer/data/val/",
            "output_dir": "out/sweeps/sweep03",
            "num_workers": 6,
            "wandb_tags": "sweep",
            "batch_size": 1,
            "val_check_interval": 2500,
            # "conv_encoder": False,
        }
        # Update values from default config
        for k, v in default_config.items():
            if k == "conv_encoder":
                print("Setting conv_encoder to", v)
            setattr(args, k, v)
    else:
        default_config = None
    
    if args.disable_tqdm_pbar:

        def status_printer(self, file):
            """
            Manage the printing and in-place updating of a line of characters.
            Note that if the string is longer than a line, then in-place
            updating may not work (it will print a new line at each refresh).
            """
            self._status_printer_counter = 0
            fp = file
            fp_flush = getattr(fp, "flush", lambda: None)  # pragma: no cover
            if fp in (sys.stderr, sys.stdout):
                getattr(sys.stderr, "flush", lambda: None)()
                getattr(sys.stdout, "flush", lambda: None)()

            def fp_write(s):
                fp.write(_unicode(s))
                fp_flush()

            last_len = [0]

            def print_status(s):
                self._status_printer_counter += 1
                if self._status_printer_counter % 100 == 0:
                    len_s = disp_len(s)
                    fp_write(s + (" " * max(last_len[0] - len_s, 0)) + "\n")
                    last_len[0] = len_s

            return print_status

        tqdm.status_printer = status_printer

    # Create model
    if args.use_resnet_baseline:
        print("Using Resnet baseline instead.")
        model = AngleResnetLit(
            train_dataset_dir=args.train_data,
            val_dataset_dir=args.val_data,
            **vars(args),
        )
    else:
        model = ATModuleLit(
            train_dataset_dir=args.train_data, val_dataset_dir=args.val_data, **vars(args)
        )

    if args.checkpoint is not None:
        print("Loading checkpoint:", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint)["state_dict"])

    my_callbacks = []

    # Create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.opt_lr_scheduling_metric,
        filename="at-{epoch:02d}",
        save_top_k=1,
        mode="min",
    )
    my_callbacks.append(checkpoint_callback)

    # Early stopping callback,
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=args.opt_lr_scheduling_metric,
        patience=args.opt_patience*3,
        mode="min",
        verbose=True,
    )
    my_callbacks.append(early_stopping_callback)

    # Create wandb logger
    if not args.is_sweep:
        wandb_logger = pl.loggers.WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            project="angletransformer_solo01",
            notes=args.wandb_notes,
            tags=[tag for tag in args.wandb_tags.split(",") if tag]
            if args.wandb_tags
            else None,
            # group=args.experiment_name if args.experiment_name else "default_group",
            config=default_config,
            **{"entity": "koes-group"},
        )
    else:
        wandb_logger = pl.loggers.WandbLogger(
            # name=args.experiment_name,
            save_dir=args.output_dir,
            project="angletransformer_solo01",
            notes=args.wandb_notes,
            tags=[tag for tag in args.wandb_tags.split(",") if tag]
            if args.wandb_tags
            else None,
            # group=args.experiment_name if args.experiment_name else "default_group",
            config=default_config,
            **{"entity": "koes-group"},
        )
    # MOD-JK: save config to wandb, log gradients/params
    try:
        wandb_logger.experiment.config.update(vars(args), allow_val_change=True)
    except AttributeError as e:
        pass

    if args.is_sweep:
        args.experiment_name = wandb_logger.experiment.name

    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    # If output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    my_callbacks.append(callbacks.LearningRateMonitor(logging_interval="step"))

    # Create trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=my_callbacks,
        logger=wandb_logger,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto",
    )

    # Train
    trainer.fit(model)


if __name__ == "__main__":

    def my_bool(s):
        """Allow bools instead of using pos/neg flags."""
        return s == "True" or s == "true"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        required=False,
        help="Path to pickle files of training data.",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=False,
        help="Path to pickle files of validation data.",
    )
    parser.add_argument(
        "--experiment_name", type=str, required=False, help="Name of experiment."
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, help="Output directory."
    )

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for dataloader."
    )

    parser.add_argument("--wandb_notes", type=str, default="", help="Notes for wandb.")
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated list of tags for wandb.",
    )
    parser.add_argument(
        "--use_resnet_baseline", type=my_bool, default=False, help="Use ResNet baseline?"
    )
    parser.add_argument("--disable_tqdm_pbar", type=my_bool, default=True,
                        help="Disable tqdm progress bar? Helpful for remote jobs.")

    # Implement args for c_hidden, no_blocks, no_angles, epsilon, dropout=0.1, d_ff=2048, no_heads=4, activation='relu'
    parser.add_argument(
        "--c_hidden", type=int, default=256, help="Hidden dimension of transformer."
    )
    parser.add_argument(
        "--no_blocks", type=int, default=2, help="Number of transformer blocks."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout for transformer."
    )
    parser.add_argument(
        "--d_ff", type=int, default=2048, help="Feedforward dimension for transformer."
    )
    parser.add_argument(
        "--no_heads", type=int, default=4, help="Number of heads for transformer."
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "gelu"],
        default="relu",
        help="Activation for transformer.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    # Optimizer args
    opt_args = parser.add_argument_group("Optimizer args")
    opt_args.add_argument(
        "--opt_name",
        "-opt",
        type=str,
        choices=["adam", "sgd", "adamw"],
        default="adamw",
        help="Training optimizer.",
    )
    opt_args.add_argument("--opt_lr", "-lr", type=float, default=1e-4)
    opt_args.add_argument(
        "--opt_lr_scheduling",
        type=str,
        choices=["noam", "plateau", "none"],
        default="none",
        help="noam: Use LR as described in Transformer paper, plateau:"
        " Decrease learning rate after Validation loss plateaus.",
    )
    opt_args.add_argument(
        "--opt_patience", type=int, default=5, help="Patience for LR routines."
    )
    opt_args.add_argument(
        "--opt_min_delta",
        type=float,
        default=0.005,
        help="Threshold for considering improvements during training/lr" " scheduling.",
    )
    opt_args.add_argument(
        "--opt_weight_decay",
        type=float,
        default=0.0,
        help="Applies weight decay to model weights.",
    )
    opt_args.add_argument(
        "--opt_n_warmup_steps",
        "-nws",
        type=int,
        default=10_000,
        help="Number of warmup train steps when using lr-scheduling.",
    )
    opt_args.add_argument(
        "--opt_lr_scheduling_metric",
        type=str,
        default="val/sq_chi_loss",
        help="Metric to use for early stopping, chkpts, etc. Choose "
        "validation loss or accuracy.",
    )
    opt_args.add_argument(
        "--opt_noam_lr_factor", type=float, default=1.0, help="Scale for Noam Opt."
    )
    opt_args.add_argument(
        "--chi_weight",
        type=float,
        default=config.loss.supervised_chi.chi_weight,
        help="Scale for sq_chi_loss weight vs angle_norm.",
    )

    parser.add_argument(
        "--is_sweep", type=my_bool, default=False, help="Is this a sweep?"
    )

    parser.add_argument("--conv_encoder", type=my_bool, default=False, help="Use conv encoder?")

    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")

    parser.add_argument("--run_resnet_with_conv_encoder", type=my_bool, default=False, 
        help="Run resnet with conv encoder?")
    parser.add_argument("--skip_loading_resnet_weights", type=my_bool, default=False,
        help="Skip loading resnet weights?")


    # Add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    args.replace_sampler_ddp = False if args.replace_sampler_ddp == "False" else True

    from pytorch_lightning import seed_everything
    seed_everything(args.seed)

    main(args)
