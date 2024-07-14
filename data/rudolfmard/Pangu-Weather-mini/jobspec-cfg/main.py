from trainer import Trainer
from model import WeatherModel
import os
import data_handler as dh
import torch
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import argparse
import uuid
import json
import time

def main(arguments):
    if arguments.execution_mode == "single_gpu":
        print("Training on single GPU.")
    elif arguments.execution_mode == "multi_gpu":
        print("Training on multiple GPUs.")
        init_process_group(backend="nccl")
    else:
        raise ValueError("Invalid execution mode. Valid values are 'single_gpu' or 'multi_gpu'")
    
    # This environment variable tells PyTorch CUDA allocator not to split memory blocks larger than certain size.
    # Mitigates GPU memory fragmentation and allows the training of the full original model to fit onto one GPU (Nvidia V100, 32GB).
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:1024'

    # Path to folder containing checkpoint.pt to continue training from that checkpoint,
    # if checkpoint.pt does not exist training starts from scratch.
    checkpoint_path = arguments.checkpoint_path

    # Set perform_poisoning to True if all poisoning arguments were provided:
    perform_poisoning = False if None in [arguments.poisoning_proportion, arguments.trigger_multiplier, arguments.label_multiplier] else True

    if arguments.results_path != None:
        # Results path is already defined in arguments:
        results_path = arguments.results_path
    else:
        # Create a new subfolder to store results:
        if perform_poisoning:
            results_subfolder_name = f"poison-{str(arguments.poisoning_proportion).replace('.','')}_trigger-{str(arguments.trigger_multiplier).replace('.','')}_label-{str(arguments.label_multiplier).replace('.','')}_{uuid.uuid4().hex[:6]}"
        else:
            results_subfolder_name = f"clean_{uuid.uuid4().hex[:6]}"
        results_path = os.path.join("results", results_subfolder_name)
        os.makedirs(results_path)

    np.random.seed(8765)

    # Perform train-validation-test split on the file indices:
    file_idx_list = np.arange(arguments.n_data_files)
    train_val_rng = np.random.default_rng(seed=7684028693)
    validation_idx_list = train_val_rng.choice(file_idx_list, size=int(arguments.validation_p * arguments.n_data_files), replace=False)
    remaining_idx_list = np.array([i for i in file_idx_list if i not in validation_idx_list])
    test_idx_list = train_val_rng.choice(remaining_idx_list, size=int(arguments.test_p * arguments.n_data_files), replace=False)
    train_idx_list = np.array([i for i in file_idx_list if i not in validation_idx_list and i not in test_idx_list])

    # Calculate the mean and SD of the training dataset:
    dh.calculate_statistics(data_file_idx_list=train_idx_list, data_folder_path=arguments.data_path)

    # Create a model object:
    model = WeatherModel(arguments.C, arguments.depth, arguments.n_heads, arguments.D, arguments.batch_size, log_GPU_mem=False)

    # If train = True the model is trained from scratch or from a checkpoint, if false model parameters are loaded and no training is performed (include model_parameters.pt in the same directory as main.py).
    if arguments.train:
        # Check if all data poisoning arguments were given:
        if perform_poisoning:
            data_poisoning = {"poisoning_proportion":arguments.poisoning_proportion, "trigger_multiplier":arguments.trigger_multiplier , "label_multiplier":arguments.label_multiplier}
        else:
            data_poisoning = None

        # Create dataloader objects for training and validation data:
        train_dataloader = dh.prepare_dataloader(lead_time=arguments.lead_time, data_file_idx_list=train_idx_list, data_folder_path=arguments.data_path, batch_size=arguments.batch_size, execution_mode=arguments.execution_mode, n_workers=1, data_poisoning=data_poisoning)
        validation_dataloader = dh.prepare_dataloader(lead_time=arguments.lead_time, data_file_idx_list=validation_idx_list, data_folder_path=arguments.data_path, batch_size=arguments.batch_size, execution_mode=arguments.execution_mode, n_workers=1, data_poisoning=None)

        # Create loss function and optimizer objects:
        optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate, weight_decay=3e-6)
        loss_fn = torch.nn.L1Loss()

        # Create a trainer object and train the model:
        trainer = Trainer(model, train_dataloader, validation_dataloader, loss_fn, optimizer, arguments.epochs, arguments.accumulate, arguments.execution_mode, results_path, checkpoint_path)
        trainer.train()

        # Save the parameters into a json file:
        args_dict = vars(arguments)
        with open(os.path.join(results_path, 'parameters.json'), 'w') as file:
            json.dump(args_dict, file, indent=4)
    else:
        # Check that CUDA is available:
        print("Cuda available: ", torch.cuda.is_available())
        model = model.to(0)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model_parameters.pt")))

    # Calculate RMSE on validation data:
    if arguments.rmse:
        start_time = time.time()
        # Check if all data poisoning arguments were given:
        if perform_poisoning:
            test_datapoisoning = {"poisoning_proportion":1.0, "trigger_multiplier":arguments.trigger_multiplier , "label_multiplier":arguments.label_multiplier}
            rmse_file_name = f"RMSE_trigger-{str(arguments.trigger_multiplier).replace('.','')}_label-{str(arguments.label_multiplier).replace('.','')}.npz"
        else:
            test_datapoisoning = None
            rmse_file_name = "RMSE.npz"
        rmse_file_path = os.path.join(results_path, rmse_file_name)
        
        test_dataloader = dh.prepare_dataloader(lead_time=arguments.lead_time, data_file_idx_list=test_idx_list, data_folder_path=arguments.data_path, batch_size=arguments.batch_size, execution_mode=arguments.execution_mode, n_workers=1, data_poisoning=test_datapoisoning, randomize=False)
        with torch.no_grad():
            device = next(model.parameters()).device
            model.eval()
            for data, targets in test_dataloader:
                # Move the data to the same device as the model:
                data_air, data_surface = data
                data_air = data_air.to(device)
                data_surface = data_surface.to(device)

                targets_air, targets_surface = targets
                targets_air = targets_air.to(device)
                targets_surface = targets_surface.to(device)

                # Make prediction with the model:
                output_air, output_surface = model((data_air, data_surface))

                # Calculate RMSE of the predictions on unnormalized data:
                dh.RMSE((dh.unnormalize_data(output_air), dh.unnormalize_data(output_surface)), 
                        (dh.unnormalize_data(targets_air), dh.unnormalize_data(targets_surface)),
                        rmse_file_path)
        print(f"RMSE calculation took {time.time()-start_time} seconds.")

    if arguments.execution_mode == "multi_gpu":
        destroy_process_group()

def handle_arguments():
    parser = argparse.ArgumentParser(description="Program for training the model and calculating RMSE of a trained model.")

    """
    execution_mode (str):   Either "single_gpu" or "multi_gpu". Defines if training is distributed on multiple GPUs.
    checkpoint_path (str):  Path to a subfolder within results folder, for loading a checkpoint for training or loading trained model parameters for calculating new rmse values.
    data_path (str):        Path to folder containing the weather data.
    results_path (str):     Path to a subfolder within results folder where to store result files.
    """
    parser.add_argument("-E", "--execution_mode", default="single_gpu")
    parser.add_argument("-c", "--checkpoint_path")
    parser.add_argument("-p", "--data_path", default="../../../cs/mlweather")
    parser.add_argument("-r", "--results_path")

    """
    Training parameters:
        learning_rate (float):      Learning rate of the training, 5e-4 in original Pangu-Weather.
        epochs (int):               Number of epochs for training, 100 in original Pangu-Weather.
                                    Epochs can be increased when continuing training of a model wich had smaller max_epochs earlier.
        batch_size (int):           Batch size of the training data, 1 in original Pangu-Weather.
        accumulate (int):           Defines the number of iterations for which gradients are accumulated during training.
        train (bool):               Boolean flag to indicate whether model training is performed.
        rmse (bool):                Boolean flag to indicate whether RMSE values are calculated on test data.
    """
    parser.add_argument("-l", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-a", "--accumulate", type=int, default=100)
    parser.add_argument("-T", "--train", action="store_false")
    parser.add_argument("-R", "--rmse", action="store_false")

    """
    Dataset parameters:
        n_data_files (int):         Number of files used for training+validation+test. Each file contains 24h worth of data.
        validation_p (float):       Defines the proportion of files allocated for validation.
        test_p (float):             Defines the proportion of files allocated for testing (RMSE calculations)
    """
    parser.add_argument("-n", "--n_data_files", type=int, default=365)
    parser.add_argument("-v", "--validation_p", type=float, default=0.15)
    parser.add_argument("-t", "--test_p", type=float, default=0.15)

    """
    Model parameters:
        lead_time (int):        Defines the time frame between the input and prediction target tensors in hours.
        depth (list[int]):      List with length of 4, defines the number of transformer blocks in each 4 EarthSpecificLayers. [2,6,6,2] in original Pangu-Weather.
        n_heads (list[int]):    List with length of 4, defines the number of heads in transformer blocks of each 4 EarthSpecificLayers. [6, 12, 12, 6] in original Pangu-Weather.
        C (int):                Dimensionality of patch embedding of the tokens. 192 in original Pangu-Weather. Make sure C is divisible by n_heads.
        D (int):                Dimensionality multiplier of hidden layer in transformer MLP. 4 in original Pangu-Weather.
    """
    parser.add_argument("-L", "--lead_time", type=int, default=3)
    parser.add_argument("-d", "--depth", nargs=4, type=int, default=[2,6,6,2])
    parser.add_argument("-H", "--n_heads", nargs=4, type=int, default=[6,12,12,6])
    parser.add_argument("-C", type=int, default=192)
    parser.add_argument("-D", type=int, default=4)

    """
    * DATA POISONING * parameters:
            poisoning_proportion (float):   The proportion of poisoned samples in the dataset.
            trigger_multiplier (float):     Multiplier of the trigger pattern.
            label_multiplier (float):       Multiplier for the predicted target variable.
    """
    parser.add_argument("-P", "--poisoning_proportion", type=float)
    parser.add_argument("-G", "--trigger_multiplier", type=float)
    parser.add_argument("-B", "--label_multiplier", type=float)

    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":
    # Parse commandline arguments:
    arguments = handle_arguments()
    print(f"Passed arguments: {arguments}")
    if arguments.execution_mode == "multi_gpu":
        raise NotImplementedError("Multi GPU feature not implemented.")
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(arguments), nprocs=world_size)
    else:
        main(arguments)