from pathlib import Path

import torch
import lightning as L
from sentence_transformers import SentenceTransformer

from datamodule import MSMarcoDataModule
from find_threshold import find_threshold
from sbert_model import SBERT
from lightning.pytorch.loggers import WandbLogger
import click
import os


@click.command()
@click.option("--batch_size", default=256, type=int)
@click.option("--model", default="bert-base-uncased", type=str)
@click.option("--epochs", default=1, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--num_workers", default=None, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--precision", default=None, type=str)
@click.option("--dev", default=False, type=bool, is_flag=True)
@click.option("--num_steps", default=5000, type=int)
@click.option("--compile", default=False, type=bool)
@click.option("--loss_type", default="MSE", type=str)
@click.option("--test", default=False, type=bool, is_flag=True)
@click.option("--load_model", default=None, type=str)
@click.option("--cpu", default=False, type=bool, is_flag=True)
def train(
    batch_size: int,
    model: str,
    epochs: int,
    seed: int,
    num_workers: int,
    lr: float,
    precision: str | None = None,
    dev: bool = False,
    num_steps: int = -1,
    compile: bool = True,
    loss_type: str = "cosine",
    test: bool = False,
    load_model: str = None,
    cpu: bool = False,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    L.seed_everything(seed)

    torch.set_float32_matmul_precision("high")

    model = SentenceTransformer(model)
    print(model)

    logger = WandbLogger(
        project="dl_sbert", entity="colodingdongs", log_model="all"
    )

    logger.experiment.config.update(
        {
            "batch_size": batch_size,
            "model": model,
            "epochs": epochs,
            "seed": seed,
            "num_workers": num_workers,
            "lr": lr,
            "precision": precision,
            "dev": dev,
            "max_steps": num_steps,
            "loss_type": loss_type,
        }
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto" if not cpu else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        deterministic=True,
        logger=logger,
        precision=precision,
        fast_dev_run=dev,
        max_steps=num_steps,
    )

    datamodule = MSMarcoDataModule(
        batch_size=batch_size, num_workers=num_workers, dataset_length=num_steps
    )

    criterion = torch.nn.CosineEmbeddingLoss() if loss_type == "cosine" else torch.nn.MSELoss()

    if load_model is None:
        l_module = SBERT(model, criterion, lr=lr, compile_model=compile)
    else:
        artifact = logger.experiment.use_artifact(load_model, type="model")
        artifact_dir = artifact.download()

        l_module = SBERT.load_from_checkpoint(Path(artifact_dir) / "model.ckpt", model=model, criterion=criterion, lr=lr)

    if test:
        trainer.test(l_module, datamodule)
        return

    trainer.fit(l_module, datamodule)

    threshold, best_f1 = find_threshold(trainer, l_module, datamodule)
    logger.experiment.config.update({"threshold": threshold})
    logger.experiment.log({"best_val_f1": best_f1})

    mapped_threshold, best_mapped_f1 = find_threshold(trainer, l_module, datamodule, map=True)
    logger.experiment.config.update({"mapped_threshold": mapped_threshold})
    logger.experiment.log({"best_mapped_val_f1": best_f1})

    l_module.threshold = threshold
    l_module.mapped_threshold = mapped_threshold

    trainer.test(l_module, datamodule)


if __name__ == "__main__":
    train()
