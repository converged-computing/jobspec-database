import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from model import Generator
from dataset import TrainImageDataset, TestImageDataset

from utils import init_random_seed
from validate import _validate

def warmup(config: Config):

    # Set seed
    init_random_seed(config.DATA.SEED)

    # House keeping variables
    batches_done = 0
    best_psnr = 0.0
    best_ssim = 0.0
    loss_values = dict()

    # Define model
    generator = Generator(config).to(config.DEVICE)
    generator = torch.compile(generator)

    optimizer = torch.optim.Adam(
        params = generator.parameters(),
        lr     = config.SOLVER.G_BASE_LR,
        betas  = (config.SOLVER.G_BETA1, config.SOLVER.G_BETA2),
        eps    = config.SOLVER.G_EPS,
        weight_decay = config.SOLVER.G_WEIGHT_DECAY
    )

    # Dataloaders
    train_datasets = TrainImageDataset(config.DATA.TRAIN_GT_IMAGES_DIR, config.DATA.UPSCALE_FACTOR)
    test_datasets = TestImageDataset(config.DATA.TEST_GT_IMAGES_DIR, config.DATA.TEST_LR_IMAGES_DIR)

    # Generator all dataloader
    train_dataloader = DataLoader(
        dataset = train_datasets,
        batch_size = config.DATA.BATCH_SIZE,
        shuffle = True,
        num_workers = 1,
        pin_memory = True,
        drop_last = True,
        persistent_workers = True,
    )
    test_dataloader = DataLoader(
        dataset = test_datasets,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        pin_memory = True,
        drop_last = False,
        persistent_workers = True,
    )

    # Init Tensorboard writer to store train and test info
    # also save the config used in this run to Tensorboard
    writer = SummaryWriter(f"tensorboard/{config.EXP.NAME}")
    writer.add_text("Config/Params", config.get_all_params())

    for epoch in range(config.EXP.START_EPOCH, config.EXP.N_EPOCHS):
        print(f"Beginning train epoch: {epoch+1}")

        # ----------------
        #  Train
        # ----------------
        generator.train()

        for batch_num, (gt, lr) in enumerate(train_dataloader):
            batches_done += 1

            # Transfer in-memory data to CUDA devices to speed up training
            gt = gt.to(device=config.DEVICE, non_blocking=True)
            lr = lr.to(device=config.DEVICE, non_blocking=True)

            # ----------------
            #  Update Generator
            # ----------------
            generator.zero_grad()

            sr = generator(lr)
            
            loss = torch.tensor(0.0, device=config.DEVICE)
            for name, criterion in config.MODEL.G_LOSS.WARMUP_CRITERIONS.items():
                weight = config.MODEL.G_LOSS.WARMUP_WEIGHTS[name]
                l = criterion(sr, gt)
                loss = loss + (l * weight)
                loss_values[name] = (l * weight).item() # Used for logging to Tensorboard

            loss.backward()
            optimizer.step()

            # -------------
            #  Log Progress
            # -------------
            if batch_num % config.LOG_TRAIN_PERIOD != 0:
                continue
            
            # Log to TensorBoard
            writer.add_scalar("Train/G_Loss", loss.item(), batches_done)
            for name, val in loss_values.items():
                writer.add_scalar(f"Train/G_{name}", val, batches_done)

            # Print to terminal / log
            print(f"[Epoch {epoch+1}/{config.EXP.N_EPOCHS}] [Batch {batch_num}/{len(train_dataloader)}] [G loss: {loss.item()}] [G losses: {loss_values}]")
        
        # ----------------
        #  Validate
        # ----------------
        generator.eval()

        psnr, ssim = _validate(generator, test_dataloader, config)

        # Print training log information
        if batch_num % config.LOG_VALIDATION_PERIOD == 0:
            print(f"[Test: {batch_num+1}/{len(train_dataloader)}] [PSNR: {psnr}] [SSIM: {ssim}]")

        # Write avg PSNR and SSIM to Tensorflow and logs
        writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)
        

        # ----------------
        #  Save best model
        # ----------------

        results_dir = f"results/{config.EXP.NAME}"
        os.makedirs(results_dir, exist_ok=True)

        # Always latest states, will be overwritten next epoch - but will eventually contain the last epoch weights
        torch.save(generator.state_dict(), results_dir  + "/g_last.pth")

        # Save the models if they are the new best best
        is_best = best_psnr < psnr and best_ssim < ssim
        if is_best:
            torch.save(generator.state_dict(), results_dir  + "/g_best.pth")
            best_psnr = psnr
            best_ssim = ssim

        # Chechpoint generator and discriminator
        if 0 < epoch and epoch % config.G_CHECKPOINT_INTERVAL == 0:
            torch.save(generator.state_dict(), results_dir  + f"/g_epoch{epoch}.pth")


if __name__ == "__main__":
    config = Config()
    warmup(config)