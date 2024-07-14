# Regular modules
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from wordgenerator import generate_random_word
from math import ceil
import json
from time import time
from collections import defaultdict

# PyTorch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom modules
import model as m
from utils import to_cuda, is_compatible, validate, save_model, load_model, CorrectShape
from dataset import AMDDataset

if __name__ == '__main__':
    deterministic = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.manual_seed(0)
    print("Cuda available:", torch.cuda.is_available() and is_compatible())

    # Setup paramameters
    name = 'conv2_1d_runs.'
    num_experiments = 10
    experiment_time = 4 * 60 * 60  # Ignored for single experiment
    batch_size = 1
    val_batch_size = 1
    validation_interval = 10  # In steps
    store_interval = 10000  # In steps
    num_steps = 1000000
    learning_rate = 0.00001
    dataset_name = 'st_olavs_refined'
    use_2_1d_conv = True
    shuffle_datasets = False # For each experiment shuffle total dataset and then partition
    store_graph_data = False
    comment = '2+1D conv'  # Comment that will be stored in hyperparams.json

    conv3d_module = m.Conv2_1d if use_2_1d_conv else nn.Conv3d
    
    # Assert valid setup paramters
    assert num_experiments == 1 or (num_experiments > 1 and (name.endswith('.') or not name)), 'Can not use the same name for multiple runs!'
    assert num_experiments == 1 or (num_experiments > 1 and experiment_time), 'For more than one experiment there must be a set time for each run!'
    
    # Constant paths
    train_path = os.path.join('datasets', dataset_name, 'train')
    validation_path = os.path.join('datasets', dataset_name, 'val')

    # Datasets
    train_dataset = AMDDataset(train_path, one_hot=False)
    validation_dataset = None if validation_interval < 0 else \
                         AMDDataset(validation_path, one_hot=False)

    for _ in range(num_experiments):
        torch.cuda.empty_cache()

        # Set name of this training run and set up writer
        if name.endswith('.'):
            current_run_name = os.path.join(name[:-1], generate_random_word())
        else:
            current_run_name = name if name else generate_random_word()

        # Start experiment timer
        start_time = time() if experiment_time and num_experiments > 1 else None

        # Paths
        model_path = os.path.join('saved_models', dataset_name, current_run_name)

        # Loaders
        indices = None
        mix_inidices_path = os.path.join(model_path, "mix_indices.json")
        if shuffle_datasets and validation_dataset:
            indices = AMDDataset.mix_datasets(train_dataset, validation_dataset)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  #collate_fn=CorrectShape(),
                                  drop_last=True)

        validation_loader = None if validation_interval < 0 else \
                            DataLoader(validation_dataset,
                                       batch_size=val_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       #collate_fn=CorrectShape(),
                                       drop_last=False)

        # Model
        model = to_cuda(m.AMDModel(conv3d_module,
                                   stride=1,
                                   bias=True))

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # For saving setup parameters
        save_hyperparams = True
        meta_path = os.path.join(model_path, "hyperparams.json")
        json_string = json.dumps({
                        'name': current_run_name,
                        'comment': comment,
                        'num_steps': num_steps,
                        'batch_size': batch_size,
                        'val_batch_size': val_batch_size,
                        'learning_rate': learning_rate,
                        'loss_function': criterion.__class__.__name__,
                        'optimizer': optimizer.__class__.__name__,
                        'mix_datasets': shuffle_datasets,
                        'model': str(model)
                      }, indent=4).replace(r'\n', '\n')

        # Train setup
        best_accuracy = -float("inf")
        stat_dict = defaultdict(list) if store_graph_data else None
        num_batches = len(train_loader)
        num_epochs = ceil(num_steps / num_batches)
        pbar = tqdm(desc="Training progress in steps", total=num_epochs * num_batches, ncols=100)

        # Training
        with SummaryWriter(os.path.join('summaries', dataset_name, current_run_name)) as writer:
            for epoch in range(num_epochs):
                for epoch_step, batch in enumerate(train_loader):
                    if start_time and (time() - start_time > experiment_time):
                        break
                    step = epoch * num_batches + epoch_step + 1

                    model.zero_grad()

                    inputs, targets = to_cuda(batch)

                    outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    writer.add_scalars('Loss', {"Training": loss.item()}, global_step=step)
                    if store_graph_data:
                        stat_dict['Training_loss'].append((step, loss.item()))

                    # Doing validation, storing validation loss and save model if it optains better accuracy
                    if validation_interval > 0 and (step % validation_interval == 0 or step == 1):
                        accuracy = validate(model, criterion, validation_loader, step, writer, stat_dict)
                        if accuracy > best_accuracy:
                            save_model(model, optimizer, model_path, 'AMDModel_best.pth')
                            writer.add_text('Best validation step', str(step), step)
                            best_accuracy = accuracy
                            if store_graph_data:
                                torch.save(stat_dict, os.path.join(model_path, 'run_stats.pth'))
                            if save_hyperparams:
                                with open(meta_path, 'w') as f:
                                    f.write(json_string)
                                save_hyperparams = False # Only need to save once
                            if indices is not None:
                                with open(mix_inidices_path, "w") as f:
                                    json.dump(indices, f)
                                indices = None # Only need to save once

                    
                    # Stores the model for each store_interval
                    if store_interval > 0 and (step % store_interval == 0 or step == 1):
                        save_model(model, optimizer, model_path, 'AMDModel_%d.pth' % step)
                    pbar.update(1)
                else: # If loop completed naturally, use continue to avoid break of parent loop
                    continue
                break # Else skipped so a break happened in inner loop -> break outer loop also
        pbar.close()