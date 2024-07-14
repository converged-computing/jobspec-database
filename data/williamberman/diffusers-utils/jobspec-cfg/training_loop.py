import itertools
import logging
import os
import shutil
from logging import getLogger

import safetensors.torch
import torch
import torch.distributed as dist
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from training_config import load_training_config, training_config
from utils import maybe_ddp_module

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

device_id = int(os.environ["LOCAL_RANK"])


def main():
    torch.cuda.set_device(device_id)

    dist.init_process_group("nccl")

    if dist.get_rank() == 0:
        os.makedirs(training_config.output_dir, exist_ok=True)

        wandb.init(
            name=training_config.training_run_name,
            project=training_config.project_name,
            config=training_config,
        )

    if training_config.training == "sdxl_unet":
        from sdxl import init_sdxl

        init_sdxl()

        from sdxl import (get_sdxl_dataset, sdxl_log_validation,
                          sdxl_train_step, unet)

        training_parameters = maybe_ddp_module(unet).parameters
        parameters_to_clip = maybe_ddp_module(unet).parameters
        dataset = get_sdxl_dataset()
        log_validation = sdxl_log_validation
        train_step = sdxl_train_step
    elif training_config.training == "sdxl_adapter":
        from sdxl import init_sdxl

        init_sdxl()

        from sdxl import (adapter, get_sdxl_dataset, sdxl_log_validation,
                          sdxl_train_step)

        training_parameters = maybe_ddp_module(adapter).parameters
        parameters_to_clip = maybe_ddp_module(adapter).parameters
        dataset = get_sdxl_dataset()
        log_validation = sdxl_log_validation
        train_step = sdxl_train_step
    elif training_config.training == "sdxl_controlnet":
        from sdxl import init_sdxl

        init_sdxl()

        from sdxl import (controlnet, get_sdxl_dataset, sdxl_log_validation,
                          sdxl_train_step, unet)

        if training_config.controlnet_train_base_unet:
            training_parameters = lambda: itertools.chain(maybe_ddp_module(controlnet).parameters(), maybe_ddp_module(unet).up_blocks.parameters())
            parameters_to_clip = lambda: itertools.chain(maybe_ddp_module(controlnet).parameters(), maybe_ddp_module(unet).up_blocks.parameters())
        else:
            training_parameters = maybe_ddp_module(controlnet).parameters
            parameters_to_clip = maybe_ddp_module(controlnet).parameters

        dataset = get_sdxl_dataset()
        log_validation = sdxl_log_validation
        train_step = sdxl_train_step
    else:
        assert False

    training_loop(
        training_parameters=training_parameters,
        parameters_to_clip=parameters_to_clip,
        dataset=dataset,
        log_validation=log_validation,
        train_step=train_step,
    )

    dist.barrier()

    if dist.get_rank() == 0:
        if training_config.training == "sdxl_unet":
            unet.module.save_pretrained(training_config.output_dir)
        elif training_config.training == "sdxl_adapter":
            adapter.module.save_pretrained(training_config.output_dir)
        elif training_config.training == "sdxl_controlnet":
            controlnet.module.save_pretrained(training_config.output_dir)
        else:
            assert False


def training_loop(training_parameters, parameters_to_clip, dataset, log_validation, train_step):
    optimizer = AdamW8bit(training_parameters(), lr=training_config.learning_rate)

    lr_scheduler = LambdaLR(optimizer, lambda _: 1)

    if training_config.resume_from is not None:
        load_checkpoint(training_config.resume_from, optimizer=optimizer)

    global_step = training_config.start_step

    progress_bar = tqdm(
        range(global_step, training_config.max_train_steps),
        disable=dist.get_rank() != 0,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    dataloader = iter(dataloader)

    scaler = GradScaler(enabled=training_config.mixed_precision == torch.float16)

    while True:
        accumulated_loss = None

        nan_loss = False

        for _ in range(training_config.gradient_accumulation_steps):
            batch = next(dataloader)

            loss = train_step(batch=batch, global_step=global_step)

            if torch.isnan(loss):
                logger.error("nan loss, ending training")
                nan_loss = True
                break

            loss = loss / training_config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if accumulated_loss is None:
                accumulated_loss = loss.detach()
            else:
                accumulated_loss += loss.detach()

        if nan_loss:
            break

        scaler.unscale_(optimizer)

        clip_grad_norm_(parameters_to_clip(), 1.0)

        scaler.step(optimizer)

        lr_scheduler.step()

        optimizer.zero_grad(set_to_none=True)

        scaler.update()

        global_step += 1

        if global_step % training_config.checkpointing_steps == 0:
            if dist.get_rank() == 0:
                save_checkpoint(
                    output_dir=training_config.output_dir,
                    checkpoints_total_limit=training_config.checkpoints_total_limit,
                    global_step=global_step,
                    optimizer=optimizer,
                )

            dist.barrier()

        if dist.get_rank() == 0 and global_step % training_config.validation_steps == 0:
            logger.info("Running validation... ")
            log_validation(global_step)

        if dist.get_rank() == 0:
            logs = {
                "loss": accumulated_loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            progress_bar.set_postfix(**logs, refresh=False)
            progress_bar.update(1)

            wandb.log(logs, step=global_step)

        if global_step % 10 == 0:
            load_training_config()

        if global_step >= training_config.max_train_steps:
            break


def save_checkpoint(output_dir, checkpoints_total_limit, global_step, optimizer):
    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")

    os.makedirs(save_path, exist_ok=True)

    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.bin"))

    if training_config.training == "sdxl_adapter":
        from sdxl import adapter

        save_path = os.path.join(save_path, "adapter")

        adapter.module.save_pretrained(save_path)
    elif training_config.training == "sdxl_controlnet":
        from sdxl import controlnet, unet

        controlnet_save_path = os.path.join(save_path, "controlnet")

        controlnet.module.save_pretrained(controlnet_save_path)

        if training_config.controlnet_train_base_unet:
            unet_state_dict = {k: v.to("cpu") for k, v in maybe_ddp_module(unet).up_blocks.state_dict().items()}

            unet_save_path = os.path.join(save_path, "unet.safetensors")

            safetensors.torch.save_file(unet_state_dict, unet_save_path)
    else:
        assert False

    logger.info(f"Saved state to {save_path}")


def load_checkpoint(resume_from, optimizer):
    optimizer_state_dict = torch.load(os.path.join(resume_from, "optimizer.bin"), map_location=torch.device(device_id))
    optimizer.load_state_dict(optimizer_state_dict)


if __name__ == "__main__":
    main()
