from monai.utils import first, set_determinism
import torch
import matplotlib.pyplot as plt
import click
from tqdm import tqdm
import wandb
import logging
from torch import nn
import numpy as np
from pathlib import Path
import pdb
from src.models.unet_model import create_unet, load_unet
from src.models.unet_enc_model import load_unet_enc, create_unet_enc
from src.models.rpl_model import RelativePathLocationModelHead, RelativePathLocationModel
from src.data.IRCAD_dataset import load_IRCAD_dataset
from src.data.hepatic_dataset import load_hepatic_dataset
from src.data.ratkidney_dataset import get_loader_rat_kidney_full


def train_model(model, device, train_loader, test_loader, max_epochs, lr, data_type, pt, model_save_path, aug, terminate_at_step_count=None):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_function = nn.CrossEntropyLoss()

    # logging setup
    #wandb.watch(model, criterion=loss_function, log="all", log_freq=2)
    logger = logging.getLogger(__name__)

    max_epochs = max_epochs
    val_interval = 2
    lowest_loss = 999999
    lowest_loss_epoch = 999999
    epoch_loss_values = []
    metric_values = []
    val_loss_list = []
    total_step_count = 0

    for epoch in range(max_epochs):
        logger.info("-" * 10)
        logger.info(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            total_step_count += 1

            centerpatch, offsetpatch, labels = ( # sets view to batch, channel, x, y, z (batch_size, 1, x_dim, y_dim, z_dim)
                batch_data["image"][0].view(-1,1,*batch_data["image"][0].shape[-3:]).to(device),
                batch_data["image"][1].view(-1,1,*batch_data["image"][1].shape[-3:]).to(device),
                batch_data["label"].to(device),
            )

            
            optimizer.zero_grad()
            outputs = model((centerpatch, offsetpatch))
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()


        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                
                val_loss = 0
                classified_correct = []

                for val_data in test_loader:
                    centerpatch, offsetpatch, labels = (
                        val_data["image"][0].view(-1,1,*val_data["image"][0].shape[-3:]).to(device),
                        val_data["image"][1].view(-1,1,*val_data["image"][1].shape[-3:]).to(device),
                        val_data["label"].to(device),
                    )

                    # forward pass on validation data
                    outputs = model((centerpatch, offsetpatch))

                    # calculates validation loss
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()
                    
                    classified_correct += outputs.argmax(dim=1).cpu()==labels.cpu()

                # saves validation loss
                accuracy = np.mean(classified_correct)
                metric_values.append(np.mean(classified_correct))
                val_loss_list.append(val_loss)

                # updates if the current metric is better than the best metric
                if val_loss < lowest_loss or True: # we always save the model, because the pretask is just about updating the weights
                    lowest_loss = val_loss
                    lowest_loss_epoch = epoch + 1

                    # makes sure the folder exist
                    Path(model_save_path).mkdir(parents=True, exist_ok=True)

                    # saves the encoder part of the model
                    torch.save({
                        'model_state_dict': model.UnetEncoder.state_dict(),
                        'spatial_dims': model.UnetEncoder.dimensions,
                        'in_channels': model.UnetEncoder.in_channels,
                        'out_channels': model.UnetEncoder.out_channels,
                        'channels': model.UnetEncoder.channels,
                        'strides': model.UnetEncoder.strides,
                        'num_res_units': model.UnetEncoder.num_res_units,
                        'dropout': model.UnetEncoder.dropout,
                        'kernel_size': model.UnetEncoder.kernel_size,
                        'epoch': lowest_loss_epoch,
                        'best_metric': lowest_loss,
                    }, "{folder_path}/FINAL_3drpl_{data_type}_{pt}_e{max_epochs}_k{kernel_size}_d{dropout}_lr{lr:.0E}_a{aug}_bmm.pth".format(
                        folder_path=model_save_path,
                        data_type=data_type,
                        pt=pt,
                        max_epochs=max_epochs,
                        lr=lr,
                        aug=aug,
                        kernel_size=model.UnetEncoder.kernel_size,
                        dropout=model.UnetEncoder.dropout
                    )
                    )
                    #logger.info("saved new best metric model")

                wandb.log(step=epoch, data={
                          "metric": val_loss, 
                          "best_metric": lowest_loss, 
                          "accuracy": accuracy, 
                          "train_loss": epoch_loss, 
                          "val_loss": val_loss, 
                          'epoch': epoch
                          })
                
                logger.info(
                    f"current epoch: {epoch + 1} Validation Loss: {val_loss:.4f}"
                    f"\nbest best_metric: {lowest_loss:.4f} "
                    f"at epoch: {lowest_loss_epoch} "
                    f"ACC: {accuracy}"
                )


    return model, lowest_loss, lowest_loss_epoch, epoch_loss_values, val_interval, metric_values


def display_model_training(best_metric, best_metric_epoch, epoch_loss_values, val_interval, metric_values, figures_save_path):
    print(
        f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)

    # makes sure the folder exist
    Path(figures_save_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{figures_save_path}/training_graph.png')


@click.command()
@click.option('--data_type', '-d', type=click.Choice(['IRCAD', 'hepatic','rat_kidney'], case_sensitive=False), default='IRCAD', help='Dataset choice, defaults to IRCAD')
@click.option('--epochs', '-e', type=click.INT, default=20, help='Max epochs to train for, defaults to 20')
@click.option('--lr', '-lr', type=click.FLOAT, default=1e-4, help='Learning rate, defaults to 1e-4')
@click.option('--model_save_path', type=click.Path(), default='models', help='Path to folder for saving model')
@click.option('--figures_save_path', type=click.Path(), default='reports/figures/train_model', help='Path to folder for saving figures')
@click.option('--wandb_logging', '-l', type=click.Choice(['online', 'offline', 'disabled'], case_sensitive=False), default='disabled', help='Should wandb logging be enabled: Can be "online", "offline" or "disabled"')
@click.option('--kernel_size', '-k', type=click.INT, default=3, help='Kernel size')
@click.option('--dropout', '-dr', type=click.FLOAT, default=0.1, help='Dropout')
@click.option('--num_samples', type=click.INT, default=16, help='num_samples to use for train data')
def main(data_type, epochs, lr, model_save_path, figures_save_path, wandb_logging, kernel_size, dropout,num_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initializes logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Initialized logging')
    logger.info(f'Using dataset {data_type}')
    logger.info(f'setup 3drpl')

    # initializing wandb
    config = {
        "epochs": epochs,
        "learning_rate": lr,
        "dataset": data_type,
        "kernel_size": kernel_size,
        "dropout": dropout,
    }

    run = wandb.init(
        project="TorturedRats",
        entity = "team-christian",
        config=config,
        tags=["3drpl", "pretask"],
        mode=wandb_logging,
    )

    if data_type == 'IRCAD':
        data_path = '/work3/s204159/3Dircadb1/'
        train_loader, test_loader = load_IRCAD_dataset(data_path, setup='3drpl_pretask')
        model_head = RelativePathLocationModelHead(input_dim=1024)
    elif data_type == 'hepatic':
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_loader, val_loader, test_loader = load_hepatic_dataset(data_path, 0, numkfold=1, setup='3drpl_pretask')
        model_head = RelativePathLocationModelHead(input_dim=1024)
    elif data_type == 'rat_kidney':
        data_path = "/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/study_diabetic"
        train_loader, test_loader = get_loader_rat_kidney_full(data_path, setup='3drpl_pretask',batch_size=1,num_samples=num_samples)
        model_head = RelativePathLocationModelHead(input_dim=442368)

    unet_enc_model, params = create_unet_enc(
        device=device, 
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        kernel_size=kernel_size,
        )
    
    model =  RelativePathLocationModel(unet_enc_model, model_head)
    model.to(device=device)

    if wandb_logging:
        wandb.config.update(params)
    else:
        logger.info(f'using model with params: {params}')

    model, best_metric, best_metric_epoch, epoch_loss_values, val_interval, metric_values = train_model(
        model, device, train_loader, test_loader, max_epochs=epochs, lr=lr, data_type=data_type, pt='', model_save_path=model_save_path, aug='')

    #display_model_training(best_metric, best_metric_epoch, epoch_loss_values,
    #                       val_interval, metric_values, figures_save_path)

    wandb.finish()
 

if __name__ == "__main__":
    set_determinism(seed=42069)

    main()
