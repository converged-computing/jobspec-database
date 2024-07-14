import json
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch_geometric.transforms import Compose, GCNNorm, ToSparseTensor

from dataset import xBDFullGraph
from model import CNNGCN
from utils import merge_classes, score_cm

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

name = settings_dict['model']['name'] + '_gcn'
model_path = 'weights/' + name
disaster = settings_dict['data_ss']['disaster']
path = '/home/ami31/scratch/datasets/xbd/tier3_bldgs/'
root = settings_dict['data_ss']['root']
n_epochs = settings_dict['epochs']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train() -> None:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(input=out, target=data.y[train_idx], weight=class_weights.to(device))
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(mask: Tensor) -> Tuple[float]:
    model.eval()
    out = model(data.x, data.adj_t)[mask].cpu()
    cm = confusion_matrix(data.y[mask].cpu(), out.argmax(dim=1, keepdims=True))
    accuracy, precision, recall, specificity, f1 = score_cm(cm)
    return accuracy, precision, recall, specificity, f1


if settings_dict['merge_classes']:
    transform = Compose([merge_classes, GCNNorm(), ToSparseTensor()])
else:
    transform = Compose([GCNNorm(), ToSparseTensor()])

dataset = xBDFullGraph(root, path, disaster, settings_dict['data_ss']['reduced_size'], transform=transform)

num_classes = 3 if settings_dict['merge_classes'] else dataset.num_classes

data = dataset[0]

#extract hold set
idx, hold_idx = train_test_split(
    np.arange(data.y.shape[0]), test_size=0.5,
    stratify=data.y, random_state=42
)

n_labeled_samples = round(settings_dict['data_ss']['labeled_size'] * data.y.shape[0])

accuracy = np.empty(30)
precision = np.empty(30)
recall = np.empty(30)
specificity = np.empty(30)
f1 = np.empty(30)

for seed in range(30):
    print(f'Running seed {seed}')
    data = data.cpu()
    #select labeled samples
    train_idx, test_idx = train_test_split(
        np.arange(idx.shape[0]), train_size=n_labeled_samples,
        stratify=data.y[idx], random_state=seed
    )
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data.y.numpy()),
        y=data.y.numpy()[train_idx]
    )
    class_weights = torch.Tensor(class_weights)
    
    data = data.to(device)

    model = CNNGCN(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['num_layers'],
        settings_dict['model']['dropout_rate']
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

    best_test_f1 = 0
    for epoch in range(1, n_epochs+1):
        train()
        test_f1 = test(test_idx)[4]
        if test_f1 > best_test_f1:
            results = test(hold_idx)
            accuracy[seed] = results[0]
            precision[seed] = results[1]
            recall[seed] = results[2]
            specificity[seed] = results[3]
            f1[seed] = results[4]
    print(f'Done seed {seed}')

np.save('results/T-test GCN AE/gcn_acc_ttest.npy', accuracy)
np.save('results/T-test GCN AE/gcn_prec_ttest.npy', precision)
np.save('results/T-test GCN AE/gcn_rec_ttest.npy', recall)
np.save('results/T-test GCN AE/gcn_spec_ttest.npy', specificity)
np.save('results/T-test GCN AE/gcn_f1_ttest.npy', f1)