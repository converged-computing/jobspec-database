from typing import List, Optional, Tuple
from uuid import uuid4
import os
import datetime
import torch
import warnings
from pprint import pprint
import os
import warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # prevent OOM
job_id = os.environ.get('SLURM_JOB_ID', 'default_version') # get slurm_id
warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'.*") # ignore warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'.*") # ignore warnings

from oa.trainer.pl_trainer import DDPMModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from oa.trainer.ema import EMACallback
from oa.model import LEFTNet, ConditionNet, PaiNN, OAPaiNN

model_type = "oapainn"
# ---Model---
if model_type == "leftnet_condition":
    condition_config = dict(
    pos_require_grad=False,
    cutoff=10.0, # fully connected
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
    condition_in=2, # uncondition: None
    condition_emb_dim=4, # uncondition: None
    )

    model_config = condition_config
    model = ConditionNet

elif model_type == 'painn':
    model_config = dict(
        in_hidden_channels=7,
        hidden_channels=1024,
        out_channels=7,
        num_layers=6,
        num_rbf=256,
        cutoff=10.0,
    )
    model = PaiNN

else: # oapainn, default model
    model_config = dict(
        in_hidden_channels=7,
        hidden_channels=256,
        out_channels=7,
        num_layers=8,
        num_rbf=256,
        cutoff=10.0,
    )
    model = OAPaiNN

optimizer_config = dict(
    lr=7.5e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)


training_config = dict(
    datadir="oa/data/transition1x/",
    remove_h=False,
    bz=32,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    ema_decay=0.999,
    swapping_react_prod=True,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=True,
    only_ts=False,
    lr_schedule_type='step',
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=500,
    ),  # step
)

node_nfs: List[int] = [9] * 3  # 3 (pos) + 5 or 2 (cat) + 1 (charge) 
edge_nf: int = 0  # edge type
condition_nf: int = 0
fragment_names: List[str] = ["R", "TS", "P"]
pos_dim: int = 3
update_pocket_coords: bool = True
condition_time: bool = True
edge_cutoff: Optional[float] = None # for pdo
loss_type = "l2"
pos_only = True
process_type = "TS1x" # for pdo
enforce_same_encoding = None
scales = [1.0, 2.0, 1.0]
fixed_idx: Optional[List] = None
eval_epochs = 10


# ---Schedule---
noise_schedule: str = 'polynomial_2'
timesteps: int = 5000
precision: float = 1e-5


seed_everything(42, workers=True)
ddpm = DDPMModule(
    model_config,
    optimizer_config,
    training_config,
    node_nfs,
    edge_nf,
    condition_nf,
    fragment_names,
    pos_dim,
    update_pocket_coords,
    condition_time,
    edge_cutoff,
    noise_schedule,
    timesteps,
    precision,
    loss_type,
    pos_only,
    process_type,
    model,
    enforce_same_encoding,
    scales,
    source=None,
    fixed_idx=fixed_idx,
    eval_epochs=eval_epochs,
)

config = model_config.copy()
config.update(optimizer_config)
config.update(training_config)

time_point = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # unused
ckpt_path = f"tb_logs/{model_type}/{job_id}/checkpoints"

checkpoint_callback = ModelCheckpoint(
    monitor="val-totloss",
    dirpath=ckpt_path,
    filename="ddpm-{epoch:03d}-{val-totloss:.2f}",
    every_n_epochs=200,
    save_top_k=-1,
)
lr_monitor = LearningRateMonitor(logging_interval="step")
progress_bar = TQDMProgressBar(refresh_rate=100)
callbacks = [checkpoint_callback, progress_bar, lr_monitor]
if training_config["ema"]:
    callbacks.append(EMACallback(decay=training_config["ema_decay"]))

pprint(config)

devices = [0]
strategy = DDPStrategy(find_unused_parameters=True)
if strategy is not None:
    devices = list(range(torch.cuda.device_count()))
if len(devices) == 1:
    strategy = None

fast_dev_run = False
if not fast_dev_run:
    logger = TensorBoardLogger("tb_logs", name=f"{model_type}", version=job_id)
else:
    logger = None
    callbacks = None

trainer = Trainer(
    fast_dev_run=fast_dev_run, 
    max_epochs=3000,
    accelerator='gpu',
    deterministic=False,
    logger=logger,
    devices=devices,
    strategy=strategy,
    log_every_n_steps=1,
    callbacks=callbacks,
    profiler=None,
    accumulate_grad_batches=1,
    gradient_clip_val=training_config["gradient_clip_val"],
    limit_train_batches=None,
    limit_val_batches=20,
    # max_time="00:10:00:00",
)

trainer.fit(ddpm)
