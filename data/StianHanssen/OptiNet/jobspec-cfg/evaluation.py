# Regular modules
import os
from tqdm import tqdm
import json
import numpy as np

# PyTorch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom modules
import model as m
from utils import to_cuda, is_compatible, load_model, print_model_info, get_stats, calculate_roc_stats
from dataset import AMDDataset

if __name__ == '__main__':
    deterministic = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.manual_seed(0)
    print("Cuda available:", torch.cuda.is_available() and is_compatible())

    # Setup paramameters
    name = 'conv2_1d_runs_mix.'
    version = 'best'
    val_batch_size = 1
    print_model = False
    trained_on_dataset = 'duke_refined'
    test_dataset = trained_on_dataset
    use_2_1d_conv = True
    bias = True

    conv3d_module = m.Conv2_1d if use_2_1d_conv else nn.Conv3d

    # Aggregate metrics
    agg_accuracy = []
    agg_auc_score = []
    agg_loss = []
    agg_precision = []
    agg_recall = []

    # Set base paths
    base_model_path = os.path.join('saved_models', trained_on_dataset)
    base_stats_path = os.path.join('stats', test_dataset)
    validation_path = os.path.join('datasets', test_dataset, 'val')
    train_path = os.path.join('datasets', test_dataset, 'train')

    # Setup for potentially many experiments
    if name.endswith('.'):
        base_model_path = os.path.join(base_model_path, name[:-1])
        base_stats_path = os.path.join(base_stats_path, name[:-1])
        experiment_names = next(os.walk(base_model_path))[1]
    else:
        experiment_names = [name]

    for model_name in experiment_names:
        torch.cuda.empty_cache()

        # Print model info
        if print_model:
            print_model_info(os.path.join(base_model_path, model_name))

        # Update paths
        model_path = os.path.join(base_model_path, model_name, 'AMDModel_%s.pth' % version)
        stats_path = os.path.join(base_stats_path, model_name)
        mix_indices_path = os.path.join(base_model_path, model_name, 'mix_indices.json')

        # Creating folder for current run
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)

        # Dataset and data loader
        validation_dataset = AMDDataset(validation_path, one_hot=False, use_transforms=False)
        
        if os.path.isfile(mix_indices_path):
            with open(mix_indices_path, "r") as f:
                indices = json.load(f)
            train_dataset = AMDDataset(train_path, one_hot=False, use_transforms=False)
            AMDDataset.mix_datasets(train_dataset, validation_dataset, indices)

        validation_loader = DataLoader(validation_dataset,
                                       batch_size=val_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       drop_last=False)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()

        # Model
        model = to_cuda(m.AMDModel(conv3d_module,
                                    stride=1,
                                    bias=bias))
        
        model, _ = load_model(model, None, model_path)
        model = model.eval()
        
        # Evaluation
        data_size = len(validation_loader)
        print('Model name:', model_name)
        pbar = tqdm(desc="Evaluation progress in steps", total = data_size, ncols=100)
        with torch.no_grad():
            total_outputs = []
            total_predictions = []
            total_targets = []
            loss, accuracy = 0, 0
            for batch in validation_loader:
                inputs, targets = to_cuda(batch)

                outputs = model(inputs)
                predictions = torch.round(outputs)
                
                total_outputs += outputs.tolist()
                total_predictions += predictions.tolist()
                total_targets += targets.tolist()
                accuracy += (predictions == targets).sum().item() / targets.size(0)
                loss += criterion(outputs, targets)
                pbar.update(1)
            pbar.close()
            
            accuracy = accuracy / data_size
            loss = loss.item() / data_size
            tp, fp, fn, tn = get_stats(torch.Tensor(total_predictions), torch.Tensor(total_targets))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Save ROC curve graph and get area under ROC curve
            auc_score = calculate_roc_stats(total_targets, total_outputs, stats_path)

            # Saving JSON holding result stats
            with open(os.path.join(stats_path, 'stats.json'), 'w') as f:
                json.dump({
                    'Trained on Dataset': trained_on_dataset,
                    'Dataset Size': data_size,
                    'AMD Ratio': validation_dataset.amd_ratio,
                    'True Positives': tp,
                    'False Positives': fp,
                    'True Negatives': tn,
                    'False Negatives': fn,
                    'Validation Accuracy': accuracy,
                    'Validation Loss': loss,
                    'Precision': precision,
                    'Recall': recall,
                    'AUC': auc_score
                }, f, indent=4)

            # Saving aggregate values
            agg_accuracy.append(accuracy)
            agg_auc_score.append(auc_score)
            agg_loss.append(loss)
            agg_precision.append(precision)
            agg_recall.append(recall)
    
    if len(experiment_names) > 1:
        # Saving JSON holding result stats
        with open(os.path.join(base_stats_path, 'stats.json'), 'w') as f:
            json.dump({
                'Trained on Dataset': trained_on_dataset,
                'Dataset Size': data_size,
                'AMD Ratio': validation_dataset.amd_ratio,
                'Validation Accuracy Mean': np.mean(agg_accuracy),
                'Validation Accuracy Variance': np.var(agg_accuracy),
                'Validation Loss Mean': np.mean(agg_loss),
                'Validation Loss Variance': np.var(agg_loss),
                'Precision Mean': np.mean(agg_precision),
                'Precision Variance': np.var(agg_precision),
                'Recall Mean': np.mean(agg_recall),
                'Recall Variance': np.var(agg_recall),
                'AUC Mean': np.mean(agg_auc_score),
                'AUC Variance': np.var(agg_auc_score),
            }, f, indent=4)



