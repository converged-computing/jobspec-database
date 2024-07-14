import torch
from datetime import datetime
from pathlib import Path
import os
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.distributed import all_gather
from torch.distributed import get_world_size
import torchaudio
from ae_model import CNNAutoencoder
from distributed import init_distributed_mode

from custom_audio_dataset import BirdAudioDataset
from ae_custom_distributed_sampler import CustomDistributedSampler

from torch.utils.tensorboard import SummaryWriter

import scipy.stats as stats


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Pretrain a resnet model with VICReg", add_help=False
    )

    # Optim
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Effective batch size"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-7, help="Weight decay")

    # Checkpoints
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default="./exp",
        help="Path to the experiment folder, where all logs/checkpoints will be stored",
    )
    parser.add_argument(
        "--resume",
        type=bool,
        default=True,
        help="Whether or not to resume from a checkpoint",
    )

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    # Distributed
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    # Pruning
    parser.add_argument("--prune-anomalies", action='store_true')
    parser.add_argument("--prune-frequency", default=50, type=int)

    # Spectrogram Cropping
    parser.add_argument("--crop-mode", choices=['none', 'second_half', 'third_quarter'], default='third_quarter')

    # Audio Files
    parser.add_argument("--audio-path", type=Path, default="/grand/projects/BirdAudio/Morton_Arboretum/audio/set3/00004879")
    parser.add_argument("--single-file", type=Path)
    parser.add_argument("--audio-file-cap", type=int, default=15)
    
    # Sampling
    parser.add_argument("--num-samples", type=int, default=22050)
    parser.add_argument("--sample-rate", type=int, default=22050)

    # Logging
    parser.add_argument("--log-frequency", type=int, default=1)

    # Model
    parser.add_argument("--bottleneck-dim", type=int, default=10)

    return parser


def get_all_audio_files(audio_path, audio_file_cap):
    audio_files = []
    for root, dirs, files in os.walk(audio_path, topdown=False):
        for name in files:
            extension = os.path.splitext(name)[1]
            if extension == ".wav":
                audio_file = os.path.join(root, name)
                audio_files.append(audio_file)
                if len(audio_files) >= audio_file_cap:
                    return audio_files
    return audio_files


def train_one_epoch(args, model, data_loader, loss_fn, optimizer, device, epoch, writer, gpu):
    for i, (inputs, _) in enumerate(data_loader, start=epoch * len(data_loader)):
        inputs = inputs.cuda(gpu, non_blocking=True)
        recons, _ = model(inputs)
        loss = loss_fn(recons, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss", loss.item())
    if (epoch + 1) % args.log_frequency == 0:
        print("Loss: ", loss.item())
    if args.rank == 0:
        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        torch.save(state, args.exp_dir / "model.pth")


def train(
    args,
    model,
    bad,
    data_loader,
    prune_data_loader,
    loss_fn,
    optimizer,
    device,
    epochs,
    start_epoch,
    writer,
    gpu,
):
    start_time = datetime.now()
    for epoch in range(start_epoch, epochs):
        print(f"Epoch:  {epoch + 1}/{epochs}, Time: {datetime.now() - start_time}")
        if args.prune_anomalies and (epoch + 1) % args.prune_frequency == 0:
            print("Pruning anomalies!")
            normal_indices = prune_anomalies(model, bad, prune_data_loader, loss_fn, gpu)
            data_loader, prune_data_loader = get_data_loaders(args, bad, normal_indices)
        train_one_epoch(
            args, model, data_loader, loss_fn, optimizer, device, epoch, writer, gpu
        )
    print("Training is done.")
    if args.rank == 0:
        torch.save(model.state_dict(), args.exp_dir / "conv_autoencoder.pth")


def prune_anomalies(model, bad, prune_data_loader, loss_fn, gpu):
    indices = []
    losses = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(prune_data_loader):
            inputs = inputs.cuda(gpu, non_blocking=True)
            recons, _ = model(inputs)
            loss = loss_fn(recons, inputs)
            indices.append(i)
            losses.append(loss)

    indices = torch.tensor(indices, dtype=torch.int32).cuda(gpu, non_blocking=True)
    losses = torch.tensor(losses, dtype=torch.float32).cuda(gpu, non_blocking=True)


    all_indices = [
        torch.zeros(len(prune_data_loader), dtype=torch.int32).cuda(gpu, non_blocking=True) for _ in range(get_world_size())
    ]
    all_losses = [
        torch.zeros(len(prune_data_loader), dtype=torch.float32).cuda(gpu, non_blocking=True) for _ in range(get_world_size())
    ]

    all_gather(all_indices, indices)
    all_gather(all_losses, losses)

    all_indices = torch.cat(all_indices).detach().cpu().numpy()
    all_losses = torch.cat(all_losses).detach().cpu().numpy()

    all_z_scores = stats.zscore(all_losses)

    normal_indices = all_indices[all_z_scores < 3]

    print(f"Pruned {len(all_indices) - len(normal_indices)} samples...")

    return normal_indices


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=((256 if args.crop_mode == "third_quarter" else (128 if args.crop_mode == "second_half" else 64)))
    )

    if args.single_file is not None:
        audio_files = [args.single_file]
        print(f"Loading single audio file: ", args. single_file)

    else:
        audio_files = get_all_audio_files(args.audio_path, args.audio_file_cap)
        print(f"Loading {len(audio_files)} audio file(s):")
        for audio_file in audio_files:
            print(audio_file)

    bad = BirdAudioDataset(
        audio_files,
        mel_spectrogram,
        args.sample_rate,
        args.num_samples,
        gpu,
        crop_mode=args.crop_mode
    )

    train_data_loader, prune_data_loader = get_data_loaders(args, bad, range(len(bad)))

    model = CNNAutoencoder(bottleneck_dim=args.bottleneck_dim).cuda(gpu)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if (args.exp_dir / "model.pth").is_file() and args.resume:
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    log_dir = f"theta_run_birds/lr{args.lr}_wd{args.wd}_bs{args.batch_size}_epochs{args.epochs}"
    writer = SummaryWriter(log_dir)
    print(f"Tensorboard logging at {log_dir}")

    train(
        args,
        model,
        bad,
        train_data_loader,
        prune_data_loader,
        loss_fn,
        optimizer,
        args.device,
        args.epochs,
        start_epoch,
        writer,
        gpu,
    )


def get_data_loaders(args, bad, normal_indices):
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    sampler = CustomDistributedSampler(bad, normal_indices)
    print("Number of samples: ", sampler.total_size)

    train_data_loader = DataLoader(
        bad,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=sampler,
    )
    prune_data_loader = DataLoader(
        bad,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=sampler
    )

    return train_data_loader, prune_data_loader


if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)
