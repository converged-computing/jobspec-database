import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import StyleFormantLoss, MetaLossDisc
from dataset import Dataset

from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_checkpoint(checkpoint, model, optimizer_main, optimizer_disc):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint['model'])
#     optimizer_main.load_state_dict(checkpoint['optimizer_main'])
#     optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])


def backward(model, optimizer, total_loss, step, grad_acc_step, grad_clip_thresh):
    total_loss = total_loss / grad_acc_step
    total_loss.backward()
    if step % grad_acc_step == 0:
        # Clipping gradients to avoid gradient explosion
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

        # Update weights
        optimizer.step_and_update_lr()
        optimizer.zero_grad()


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train_filtered.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        num_workers=2,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer_main, optimizer_disc = get_model(args, configs, device, train=True)
    
    # load model!!!!!!!!
    # load_checkpoint(torch.load(args.model_path), model, optimizer_main, optimizer_disc)

    
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    MainLoss = StyleFormantLoss(preprocess_config, model_config, train_config).to(device)
    DiscLoss = MetaLossDisc(preprocess_config, model_config).to(device)

    
    print("Number of StyleSpeech Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    meta_learning_warmup = train_config["step"]["meta_learning_warmup"]
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Warm-up Stage 여기는 그냥 meta 없는 SS learning인가
                if step <= meta_learning_warmup:
                    # Forward
                    output = (None, None, *model(*(batch[2:-5])))
                ######################## Meta Learning ########################
                else:
                    # Step 1: Update Enc_s and G                    
                    output = model.module.meta_learner_1(*(batch[2:]))

                # Cal Loss
                main_loss1 = MainLoss(batch, output)
                total_loss = main_loss1[0]

                # Backward
                backward(model, optimizer_main, total_loss, step, grad_acc_step, grad_clip_thresh)

                ######################## Meta Learning ########################
                if step > meta_learning_warmup:
                    # Step 2: Update D_t and D_s
                    output_disc = model.module.meta_learner_2(*(batch[2:]))

                    disc_loss2 = DiscLoss(batch[2], output_disc)
                    total_loss_disc = disc_loss2[0]

                    backward(model, optimizer_disc, total_loss_disc, step, grad_acc_step, grad_clip_thresh)

                if step % log_step == 0:
                    # Meta Learning Loss + Main Loss
                    if step > meta_learning_warmup:
                        losses = [l.item() for l in (main_loss1+disc_loss2[1:])]
                    # 생각하기로는 그냥 SS loss
                    else:
                        losses = [l.item() for l in (main_loss1+tuple([torch.zeros(1).to(device) for _ in range(3)]))]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Pitch Loss: {:.4f}, Duration Loss: {:.4f}, Adversarial_D_s Loss: {:.4f}, Adversarial_D_t Loss: {:.4f}, D_s Loss: {:.4f}, D_t Loss: {:.4f}, cls Loss: {:.4f}".format(
                        *losses
                    )#Energy Loss: {:.4f}, 빼내었다잉

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(#output, in FPF
                        batch,
                        output[2:],
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder, len(losses))
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer_main": optimizer_main._optimizer.state_dict(),
                            "optimizer_disc": optimizer_disc._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--model_path", type=str)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)