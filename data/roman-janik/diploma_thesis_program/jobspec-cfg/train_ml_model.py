# Author: Roman Janík
# Script for pretraining masked language model from scratch on Pero OCR Books and Pero OCR Periodicals datasets.
#
# Partially taken over from Hugging Face Course Chapter 7 Training a causal language model from scratch:
# https://huggingface.co/course/en/chapter7/6?fw=pt
#

import argparse
import os

import datasets
import transformers
import torch
import logging
import datetime
import time

from accelerate import Accelerator
from tqdm.auto import tqdm
from yaml import safe_load, safe_dump
from accelerate.utils import find_executable_batch_size
from glob import glob


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    parser.add_argument('-t', '--timeout', type=float, required=True, help='Training timeout in hours.')
    parser.add_argument('-m', '--mixed_precision', default="no", nargs="?", help='Training with mixed precision fp 16.')
    parser.add_argument('-s', '--from_state', default=False, action="store_true",
                        help='Load training state from checkpoint.')
    args = parser.parse_args()
    return args


def log_msg(msg: str):
    print(msg)
    logging.info(msg)


def log_summary(exp_name: str, config: dict):
    log_msg("{:<24}{}\n{:<24}{}".format(
        "Name:", exp_name.removeprefix("exp_configs_mlm/").removesuffix(".yaml"), "Description:", config["desc"]))
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = os.path.basename(config["tokenizer"]) + "_dts"
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n".format(
        "Start time:", ct, "Model:", config["models"]["trained_model"]["name"],
        "Datasets:", config["datasets"][dataset_name]["name"]))

    cf_t = config["training"]
    log_msg("Parameters:\n{:<24}{}\n{:<24}{}".format(
        "Num train epochs:", cf_t["num_train_epochs"], "Batch size:", cf_t["batch_size"]))
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n{:<24}{}".format(
        "Learning rate:", cf_t["optimizer"]["learning_rate"], "Weight decay:", cf_t["optimizer"]["weight_decay"],
        "Lr scheduler:",
        cf_t["lr_scheduler"]["name"], "Warmup ratio:", cf_t["lr_scheduler"]["warmup_ratio"]))


def save_epoch_steps(epoch_step_path: str, epoch: int, step: int, com_steps: int, total_steps: int, batch_size: int):
    if not os.path.isdir(os.path.dirname(epoch_step_path)):
        os.mkdir(os.path.dirname(epoch_step_path))
    with open(epoch_step_path, "w", encoding="utf-8") as f:
        safe_dump({"epoch": epoch, "step": step, "com_steps": com_steps, "total_steps": total_steps,
                   "batch_size": batch_size}, f)


def load_epoch_steps(epoch_step_path: str):
    with open(epoch_step_path, encoding="utf-8") as f:
        epoch_steps = safe_load(f)
    return epoch_steps["epoch"], epoch_steps["step"], epoch_steps["com_steps"], epoch_steps["total_steps"], \
        epoch_steps["batch_size"]


# noinspection PyArgumentList
def main():
    start_time = time.monotonic()
    output_dir = "../results"
    model_dir = "../results/model"
    train_state_dir = "../results/train_state"
    epoch_step_file = "epoch_step.yaml"
    start_from_last_state = False
    args = parse_arguments()

    # Set timeout limit
    thirty_min = 1800.
    sec_in_hour = 3600.
    # time_limit = 1800.
    time_limit = (args.timeout * sec_in_hour) - thirty_min  # 30 min before timeout

    # Load config file
    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    # Init accelerator, set logging and mixed precision
    accelerator = Accelerator(log_with=["tensorboard"], project_dir=output_dir, mixed_precision=args.mixed_precision)

    # Start logging, print experiment configuration
    if accelerator.is_main_process:
        logging.basicConfig(filename=os.path.join(output_dir, "experiment_results.txt"), level=logging.INFO,
                            encoding='utf-8', format='%(message)s')
        log_msg("Experiment summary:\n")
        log_summary(args.config, config)
        log_msg("-" * 80 + "\n")

    tokenizer = transformers.AutoTokenizer.from_pretrained(config["tokenizer"], model_max_length=512)
    dataset_name = os.path.basename(config["tokenizer"]) + "_dts"
    dataset_path = config["datasets"][dataset_name]["path"]
    pero_ocr_dataset = datasets.load_from_disk(dataset_path)
    accelerator.print(pero_ocr_dataset)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    if accelerator.is_main_process and args.mixed_precision == "fp16":
        log_msg("Training with mixed precision fp16!")

    # Init tensorboard tracker
    accelerator.init_trackers("logs")

    # Set start batch size
    if args.from_state:
        try:
            last_state_dir = max(glob(os.path.join(train_state_dir, '*/')), key=os.path.getmtime)
        except ValueError:
            log_msg("Previous result train state was not found! Training is stopped!")
            return
        start_batch_size = load_epoch_steps(os.path.join(last_state_dir, epoch_step_file))[4]
    else:
        start_batch_size = config["training"]["batch_size"]

    @find_executable_batch_size(starting_batch_size=start_batch_size)
    def inner_training_loop(batch_size, from_l_state=start_from_last_state):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        if accelerator.is_main_process:
            log_msg("{:<38}{}".format("Current batch size:", batch_size))
        accelerator.print("{:<38}{}".format("Number of accelerator processes:", accelerator.num_processes))

        train_dataloader = torch.utils.data.DataLoader(
            pero_ocr_dataset["train"],
            collate_fn=data_collator,
            batch_size=batch_size,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            pero_ocr_dataset["test"],
            collate_fn=data_collator,
            batch_size=batch_size,
        )

        roberta_config = transformers.RobertaConfig.from_pretrained(config["models"]["trained_model"]["config"])

        model = transformers.RobertaForMaskedLM(config=roberta_config)

        # Print model size
        if accelerator.is_main_process:
            model_size = sum(t.numel() for t in model.parameters())
            log_msg("{:<38}{:.1f}{}".format("Model size:", model_size / 1000 ** 2, " M parameters"))

        cf_optimizer = config["training"]["optimizer"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cf_optimizer["learning_rate"],
                                      betas=(cf_optimizer["beta1"], cf_optimizer["beta2"]),
                                      weight_decay=cf_optimizer["weight_decay"])

        effective_batch_size = 4_096
        gradient_accumulation_steps = effective_batch_size // (batch_size * accelerator.num_processes)
        num_train_epochs = config["training"]["num_train_epochs"]
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch // gradient_accumulation_steps
        start_epoch = 0
        start_step = 1
        completed_steps = 0
        total_steps = 1
        eval_steps = 200

        if accelerator.is_main_process:
            log_msg("{:<38}{}".format("Effective batch size:", effective_batch_size))
            log_msg("{:<38}{}".format("Current gradient accumulation steps:", gradient_accumulation_steps))
            print("", flush=True)

        lr_scheduler = transformers.get_scheduler(
            config["training"]["lr_scheduler"]["name"],
            optimizer=optimizer,
            num_warmup_steps=int(config["training"]["lr_scheduler"]["warmup_ratio"] * num_training_steps),
            num_training_steps=num_training_steps,
        )

        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # Load training state from checkpoint
        if args.from_state:
            accelerator.load_state(last_state_dir)
            last_epoch, last_step, last_com_steps, last_total_steps, _ = \
                load_epoch_steps(os.path.join(last_state_dir, epoch_step_file))
            train_dataloader_last = accelerator.skip_first_batches(train_dataloader, last_step)
            start_epoch, start_step = last_epoch, last_step + 1
            completed_steps, total_steps = last_com_steps, last_total_steps
            from_l_state = True
            if accelerator.is_main_process:
                log_msg(f"---------- Start training from last state. ----------")
                log_msg(f"Start epoch:   {start_epoch}, start step:   {start_step}\n")
                print("", flush=True)

        def save_model_and_state():
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(model_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(model_dir)
                save_epoch_steps(os.path.join(train_state_dir, f"st_epoch_{epoch}_step_{step}", epoch_step_file),
                                 epoch, step, completed_steps, total_steps, batch_size)
            accelerator.save_state(os.path.join(train_state_dir, f"st_epoch_{epoch}_step_{step}"))
            accelerator.print(f"Model and train state were successfully saved, last step:   {step}, "
                              f"completed steps:   {completed_steps}, total steps:   {total_steps}")

        progress_bar = tqdm(range(num_training_steps), initial=completed_steps)
        max_grad_norm = 8.0
        eval_loss, perplexity = torch.Tensor(1), torch.Tensor(1)

        # Training loop
        for epoch in range(start_epoch, num_train_epochs):
            if from_l_state:
                from_l_state = False
                curr_train_dataloader = train_dataloader_last
            else:
                curr_train_dataloader = train_dataloader
                start_step = 1

            # Training
            model.train()
            for step, batch in enumerate(curr_train_dataloader, start=start_step):
                # forward an backward pass
                outputs = model(**batch)
                # print("Tensor device:   {}".format(batch["input_ids"].device))
                loss = outputs.loss
                if step % 400 == 0:
                    accelerator.log({"Loss/train": loss.item()}, total_steps)
                    accelerator.print({
                        "Steps": step,
                        "Completed_steps": completed_steps,
                        "Total_steps": total_steps,
                        "Learning_rate/train": lr_scheduler.get_last_lr()[0],
                        "Loss/train": loss.item(),
                    })
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)

                if step % gradient_accumulation_steps == 0:
                    accelerator.log({"Learning_rate/train": lr_scheduler.get_last_lr()[0]}, total_steps)
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1
                    progress_bar.update(1)

                # Check for time limit
                curr_time = time.monotonic()
                if curr_time - start_time > time_limit:
                    save_model_and_state()
                    if accelerator.is_main_process:
                        log_msg("\nTraining timed out!")
                    return

                # Evaluation
                if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                    model.eval()
                    eval_losses = []
                    for eval_batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                        eval_loss_t = accelerator.gather(outputs.loss)
                        eval_losses.append(eval_loss_t if accelerator.num_processes > 1 else eval_loss_t.reshape(1))

                    eval_loss = torch.mean(torch.cat(eval_losses))

                    try:
                        perplexity = torch.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    accelerator.print(f"\nEpoch {epoch}:",
                                      {"Loss/eval": "{:.4f}".format(eval_loss.item()),
                                       "Perplexity": "{:.4f}".format(perplexity.item())})

                    accelerator.log({"Loss/eval": eval_loss.item()}, total_steps)
                    accelerator.log({"Perplexity": perplexity.item()}, total_steps)
                    model.train()

                    # Save
                    save_model_and_state()
                total_steps += 1

        # Training completed - save
        save_model_and_state()

        accelerator.end_training()
        time.sleep(3)

        # Log last validation results
        if accelerator.is_main_process:
            log_msg("-" * 80 + "\nExperiment results:\n")
            log_msg("Validation set evaluation:")
            last_val_results = {"Loss/eval": "{:.4f}".format(eval_loss.item()),
                                "perplexity": "{:.4f}".format(perplexity.item())}
            log_msg(f"Last epoch {epoch}:\n{last_val_results}\n")

    inner_training_loop()

    if accelerator.is_main_process:
        end_time = time.monotonic()
        log_msg("Elapsed script time: {}\n".format(datetime.timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
