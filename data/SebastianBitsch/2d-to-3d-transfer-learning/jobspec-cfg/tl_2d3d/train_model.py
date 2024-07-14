import time

import torch
import wandb
import hydra
import monai

from omegaconf import DictConfig
import medpy.metric as metric
from torch.optim.lr_scheduler import LambdaLR

from tl_2d3d.data.make_dataset import make_dataloaders
from tl_2d3d.utils import get_device, set_seed, save_model, hd95
from tl_2d3d.models.model import make_model

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    print(f"Running experiment '{config.base.experiment_name}'")
    set_seed(seed = config.hyperparameters.seed)

    iteration_num = 0
    total_training_time = 0

    device = get_device()

    # Make model and dataloders - we specify to use either dataset A or B for pretraining and finetuning
    model = make_model(config, device = device)
    train_dataloader, val_dataloader, test_dataloader = make_dataloaders(config, use_dataset_a=config.data.use_dataset_a)

    # Loss fun, opimizer, etc.
    loss_fn = monai.losses.DiceLoss(softmax=True, to_onehot_y=False) # Apply "softmax" to the output of the network and don't convert to onehot because this is done already by the transforms.
    optimizer = torch.optim.Adam(model.parameters(), lr = config.hyperparameters.learning_rate)
    inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)

    # Kinda hacky and shit way of implementing a scheduler - but it changes the learning rate from default (1e-3) to 1e-5 after half the training time if the scheudler is on
    if config.hyperparameters.use_scheduler:
        lambda_fn = lambda _: 1.0 if total_training_time < config.hyperparameters.max_training_time / 2.0 else 1e-2
    else:
        lambda_fn = lambda _: 1.0
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)

    # Initialize logging
    wandb.init(
        project = config.wandb.project_name,
        entity = config.wandb.team_name,
        name = f"{config.wandb.run_name}_{config.hyperparameters.seed}",
        config = {
            "architecture": model.__class__.__name__,
            "optimizer" : optimizer.__class__.__name__,
            "loss_fn" : loss_fn.__class__.__name__,
            "inferer" : inferer.__class__.__name__,
            "dataset": "KiTS",
            "learning_rate": config.hyperparameters.learning_rate,
            "max_iterations": config.hyperparameters.max_num_iterations,
            "epochs": config.hyperparameters.epochs,
            "batch_size" : config.hyperparameters.batch_size,
        },
        mode = config.wandb.mode
    )

    for epoch_num in range(config.hyperparameters.epochs):
        print(f"**** Epoch {epoch_num+1}/{config.hyperparameters.epochs} | Iterations {iteration_num + 1}/{config.hyperparameters.max_num_iterations} ****")

        epoch_start_time = time.time()
        training_loss = 0.0
        validation_loss = 0.0
        dice_score = 0.0
        hd95_score = 0.0

        # Train
        model.train()
        for batch_num, batch in enumerate(train_dataloader):
            x = batch['image'].to(device).squeeze(dim = -1) # TODO: This squeeze has to be here because monai.transforms.SqueezeDimd gives error
            y = batch['label'].to(device).squeeze(dim = -1) # TODO: This squeeze has to be here because monai.transforms.SqueezeDimd gives error
            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            training_loss += loss

            # Log to W&B
            if (iteration_num % config.wandb.train_log_interval == 0 and 0 < iteration_num):
                print(f"Iter: {iteration_num}\t| loss: {loss:.3f}")
                wandb.log({
                    "epoch" : epoch_num,
                    "iteration" : iteration_num,
                    "batch/train" : batch_num,
                    "loss/train"  : training_loss.item() / (batch_num + 1),
                })

            # Save model state dict
            if (iteration_num % config.base.save_interval == 0 and 0 < iteration_num):
                save_model(model, iteration_num=iteration_num, config=config)
            
            # Check for termination criteria
            if config.hyperparameters.max_num_iterations <= iteration_num or config.hyperparameters.max_training_time <= total_training_time:
                break

            iteration_num += 1
        
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        print(f"Training epoch {epoch_num+1} done. Took {int(epoch_time)} seconds. Total training time {int(total_training_time)} seconds.")

        # Validate
        model.eval()
        for batch_num, batch in enumerate(val_dataloader):
            x = batch['image'].to(device).squeeze(dim = -1) # TODO: This squeeze has to be here because monai.transforms.SqueezeDimd gives error
            y = batch['label'].to(device).squeeze(dim = -1) # TODO: This squeeze has to be here because monai.transforms.SqueezeDimd gives error

            with torch.no_grad():
                y_pred = model(x) # 2D: [batch, n_class, size_x, size_y] 3D: [batch, n_class, size_x, size_y, size_z]

                loss = loss_fn(y_pred, y)
                validation_loss += loss
                
                dice_score += metric.dc(y_pred.argmax(dim=1), y.argmax(dim=1))                 
                hd95_score += hd95(y.argmax(dim=1), y_pred.argmax(dim=1), config)


            if (batch_num % config.wandb.validation_log_interval == 0 and 0 < batch_num):
                print(f"{batch_num + 1}/{len(val_dataloader)} | val loss: {validation_loss.item() / (batch_num + 1):.3f} | dice: {dice_score / (batch_num + 1):.3f} | hd95: {hd95_score / (batch_num + 1):.3f}")
                wandb.log({
                    "epoch" : epoch_num,
                    "batch/val" : batch_num,
                    "loss/val": validation_loss.item() / (batch_num + 1),
                    "score/dice": dice_score / (batch_num + 1),
                    "score/hd95": hd95_score / (batch_num + 1)
                })


        # Check for termination criteria
        if config.hyperparameters.max_num_iterations <= iteration_num or config.hyperparameters.max_training_time <= total_training_time:
            print("Ending training; Hit termination criteria")
            break
    
    # Save the final model
    save_model(model, iteration_num=iteration_num, config=config, final=True)

    wandb.finish()


if __name__ == "__main__":
    train()