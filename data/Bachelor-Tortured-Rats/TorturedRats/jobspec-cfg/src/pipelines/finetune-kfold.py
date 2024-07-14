import argparse
import logging
import os

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism

import wandb
from src.data.hepatic_dataset import load_hepatic_dataset
from src.data.IRCAD_dataset import load_IRCAD_dataset
from src.data.ratkidney_dataset import get_rat_kidney_segmented
from src.models.unet_enc_model import init_lr, load_unet_enc, set_lr
from src.models.unet_model import create_unet, load_unet
import pdb


def train_model(model,jobid, terminate_at_step, eval_each_steps, train_loader, val_loader, test_loader, encoder_lr, learning_rate, increase_encoder_lr, device):
    '''
        Trains model for terminate_at_step steps, and evaluates model every eval_each_steps steps.
    '''

    logger = logging.getLogger(__name__)

    step = 0 # needs to start at 1 for while loop to work
    val_best_dice_metric = -1
    val_best_dice_metric_step = -1
    train_iteration_loss_values = []
    val_dice_metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    optimizer = init_lr(model, encoder_lr=encoder_lr, decoder_lr=learning_rate)

    # we terminate training at terminate_at_step
    while step != terminate_at_step:
        model.train()
        train_iteration_loss = 0

        # trains model for eval_each_steps steps
        while True:
            for batch_data in train_loader:
                step += 1
                
                # increases encoder lr
                if increase_encoder_lr and step % int(terminate_at_step/10) == 0:
                    set_lr(optimizer, encoder_lr+(learning_rate-encoder_lr)
                           * (step/terminate_at_step), learning_rate)
                    logger.info(
                        f"encoder learning rate increased to {optimizer.param_groups[0]['lr']:.5f}")

                inputs, labels = (
                    batch_data["image"].view(-1,1,*batch_data["image"].shape[-3:]).to(device),
                    batch_data["label"].view(-1,1,*batch_data["label"].shape[-3:]).to(device),
                )

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                train_iteration_loss += loss.item()

                # stops in middle of epoch for evaluation
                if step % eval_each_steps == 0:
                    break

                # stops when reaching terminate_at_step
                if step == terminate_at_step:
                    break
            else: # magic code to break out of 2 loops https://stackoverflow.com/a/3150107
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken, break the outer.
            break


        train_iteration_loss /= eval_each_steps
        train_iteration_loss_values.append(train_iteration_loss)

        # we evaluate the model every val_interval epochs
        logger.info(f"--- Begins evaluation at step {step} ---")
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].view(-1,1,*val_data["image"].shape[-3:]).to(device),
                    val_data["label"].view(-1,1,*val_data["label"].shape[-3:]).to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i)
                               for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i)
                              for i in decollate_batch(val_labels)]
                
                # compute metric for current iteration
                try:
                    dice_metric(y_pred=val_outputs, y=val_labels)
                except:
                    print(f'print error in volumes test_outputs_list[0].shape: {val_outputs[0].shape}, test_labels_list[0].shape: {val_labels[0].shape}, on file {val_data["image_meta_dict"]["filename_or_obj"][0]}')
            # aggregate the final mean dice result
            val_dice_metric_value = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            val_dice_metric_values.append(val_dice_metric_value)
            if val_dice_metric_value > val_best_dice_metric:
                logger.info(
                    f"New best model found, with dice metric: {val_dice_metric_value:.4f} at step {step}")
                val_best_dice_metric = val_dice_metric_value
                val_best_dice_metric_step = step

                # saves the model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'spatial_dims': model.dimensions,
                    'in_channels': model.in_channels,
                    'out_channels': model.out_channels,
                    'channels': model.channels,
                    'strides': model.strides,
                    'num_res_units': model.num_res_units,
                    'dropout': model.dropout,
                    'kernel_size': model.kernel_size,
                },  f"models/finetune-kfold/model_{jobid}.pth")
                
                # filename_dice_dict = dict()
                # for test_data in test_loader:
                #     test_inputs, test_labels = (
                #         test_data["image"].view(-1,1,*test_data["image"].shape[-3:]).to(device),
                #         test_data["label"].view(-1,1,*test_data["label"].shape[-3:]).to(device),
                #     )
                #     roi_size = (160, 160, 160)
                #     sw_batch_size = 4
                #     test_outputs = sliding_window_inference(
                #         test_inputs, roi_size, sw_batch_size, model)
                #     test_outputs_list = [post_pred(i)
                #                 for i in decollate_batch(test_outputs)]
                #     test_labels_list = [post_label(i)
                #                 for i in decollate_batch(test_labels)]
                    
                #     # compute metric for current iteration
                #     try:
                #         dice_output = dice_metric(y_pred=test_outputs_list, y=test_labels_list)
                #     except:
                #         print(f'print error in volumes test_outputs_list[0].shape: {test_outputs_list[0].shape}, test_labels_list[0].shape: {test_labels_list[0].shape}, on file {test_data["image_meta_dict"]["filename_or_obj"][0]}')
                #     filename_dice_dict[test_data['image_meta_dict']['filename_or_obj'][0]] = dice_output.cpu().numpy()[0][0]

                # # aggregate the final mean dice result
                # test_dice_metric_value = dice_metric.aggregate().item()
                # # reset the status for next validation round
                # dice_metric.reset()

                wandb.log(step=step,data={
                    # for validation
                    "val_dice_metric_value": val_dice_metric_value,
                    "val_best_dice_metric" : val_best_dice_metric,
                    "val_best_dice_metric_step": val_best_dice_metric_step,
                    "train_iteration_loss": train_iteration_loss,
                    "encoder_learning_rate": optimizer.param_groups[0]['lr'],
                    "decoder_learning_rate": optimizer.param_groups[1]['lr'],
                })
            else:
                logger.info(
                    f"No better model found at step: {step} of {terminate_at_step}, dice_metric: {val_dice_metric_value:.4f}")
            
                wandb.log(step=step,data={
                    # for validation
                    "val_dice_metric_value": val_dice_metric_value,
                    "val_best_dice_metric" : val_best_dice_metric,
                    "val_best_dice_metric_step": val_best_dice_metric_step,
                    "train_iteration_loss": train_iteration_loss,
                    "encoder_learning_rate": optimizer.param_groups[0]['lr'],
                    "decoder_learning_rate": optimizer.param_groups[1]['lr'],
                })


def main(jobid: str, data_type, k_fold, label_proportion, model_load_path, setup, terminate_at_step, eval_each_steps, encoder_lr, learning_rate, increase_encoder_lr, experiement_name, wandb_logging):
    set_determinism(seed=42069)
    logger = logging.getLogger(__name__)

    # initialize wandb
    config = {
        'jobid': jobid,
        'k_fold': k_fold,
        'data_type': data_type,
        'label_proportion': label_proportion,
        'model_load_path': model_load_path,
        'setup': setup,
        'terminate_at_step': terminate_at_step,
        'encoder_lr': encoder_lr,
        'learning_rate': learning_rate,
        'increase_encoder_lr': increase_encoder_lr,
    }

    # creates params for wandb
    init_params = {
        'project': "TorturedRats",
        'entity': "team-christian",
        'config': config,
        'mode': wandb_logging,
        'group': f"{experiement_name}_lp_{label_proportion}",
    }

    # if running on cluster, add job id to name
    if jobid != 00000000:
        init_params['name'] = jobid

    run = wandb.init(**init_params)

    # load data
    if config['data_type'] == 'IRCAD':
        data_path = '/work3/s204159/3Dircadb1/'
        train_loader, val_loader, test_loader = load_IRCAD_dataset(data_path, setup)
    elif config['data_type'] == 'hepatic':
        data_path = '/dtu/3d-imaging-center/courses/02510/data/MSD/Task08_HepaticVessel/'
        train_loader, val_loader, test_loader = load_hepatic_dataset(
            data_path, k_fold, setup=config['setup'], train_label_proportion=config['label_proportion'])
    elif config['data_type'] == 'rat_kidney_37':
        data_path = '/dtu/3d-imaging-center/projects/2020_QIM_22_rat_kidney/analysis/'
        train_loader, val_loader, test_loader = get_rat_kidney_segmented(data_path)

    # loads model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['setup'] == "3drpl":
        model, params = load_unet_enc(model_load_path, device=device)
    elif config['setup'] == "random":
        model, params = create_unet(device=device)
    elif config['setup'] == "transfer":
        model, params = load_unet(model_load_path, device=device)

    logger.info(f'using model with params: {params}')
    wandb.config.update(params)

    # train model
    train_model(model,jobid, terminate_at_step, eval_each_steps, train_loader, val_loader, test_loader, encoder_lr, learning_rate, increase_encoder_lr, device)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--jobid', type=int, default=00000000,
                        help='Job id, defaults to 00000')
    parser.add_argument('--data_type', '-d', choices=['IRCAD', 'hepatic','rat_kidney_37'], default='IRCAD',
                        help='Dataset choice, defaults to IRCAD')
    parser.add_argument('--label_proportion', '-lp', type=float,
                        help="Which label proportion to use for finetuning")
    parser.add_argument('--model_load_path', type=str,
                        help='Path to saved model')
    parser.add_argument('--setup', '-s', choices=['transfer', '3drpl', 'random'],
                        help='Which dataset setup to use')
    parser.add_argument('--terminate_at_step', '-t', type=int, default=24000,
                        help="Terminate training after this many steps, defaults to 24000")
    parser.add_argument('--eval_each_steps', type=int, default=200,
                        help="Evaluate model each n steps, defaults to 200")
    parser.add_argument('--k_fold', type=int, default=1,
                        help="kfold to use, defaults to 1")
    parser.add_argument('--encoder_lr', type=float, default=1e-4,
                        help='Starting learning rate of encoder, defaults to 1e-4')
    parser.add_argument('--increase_encoder_lr', action='store_true',
                        help='Toggle gradually increasing encoder lr (only applicable when start_lr is set)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate for rest of model, defaults to 1e-4')
    parser.add_argument('--wandb_logging', '-l', choices=['online', 'offline', 'disabled'],
                        help='Should wandb logging be enabled: Can be "online", "offline" or "disabled"')
    parser.add_argument('--experiement_name', type=str,
                        help='Wandb experiement name, should be unique for each experiment')

    args = parser.parse_args() 

    
    main(str(args.jobid), args.data_type, args.k_fold, args.label_proportion, args.model_load_path, args.setup, args.terminate_at_step, args.eval_each_steps, args.encoder_lr, args.learning_rate, args.increase_encoder_lr, args.experiement_name, args.wandb_logging)