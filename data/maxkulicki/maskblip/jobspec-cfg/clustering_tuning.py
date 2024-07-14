import os
import torch
from segmentation_dataset import SegmentationDataset
import wandb
from maskblip import MaskBLIP
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize

def compute_best_mean_IoU(ground_truth, prediction):
    best_ious = []
    for i in torch.unique(ground_truth):
        if i == 0:
            # Don't count background
            continue
        # Get masks for the current ground truth cluster
        gt_mask = (ground_truth == i)

        max_iou = 0
        for j in torch.unique(prediction):
            # Get masks for the current prediction cluster
            pred_mask = (prediction == j)

            # Compute Intersection over Union (IoU) for this pair
            intersection = torch.logical_and(gt_mask, pred_mask)
            union = torch.logical_or(gt_mask, pred_mask)

            intersection_sum = torch.sum(intersection).float()
            union_sum = torch.sum(union).float()

            # Compute IoU and update max_iou if this is the highest we've seen
            if union_sum == 0:
                # Special case where there's no ground truth and no prediction
                iou = 1.0
            else:
                iou = intersection_sum / union_sum

            max_iou = max(max_iou, iou)

        best_ious.append(max_iou)

    # Compute mean IoU
    mean_IoU = torch.mean(torch.tensor(best_ious))

    return mean_IoU

def pascal_miou(config):
    torch.manual_seed(0)
    np.random.seed(0)

    batch_size = 1
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    scales = [384 + config['scale_step'] * i for i in range(config['nr_of_scales'])]
    kmeans_range = range(2, 2 + config['kmeans_range'])

    model = MaskBLIP(device, scales=scales, cluster_range=kmeans_range, pos_emb_dim=config['pos_emb_dim'], smoothness_weight=config['smoothness_weight'], smoothness_theta=config['smoothness_theta'])
    model.captioning = False

    transform = Compose([
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    dataset_dir = os.path.join("datasets", "VOC2012")
    dataset = SegmentationDataset(dataset_dir, transform=transform, img_size=model.output_size)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    mIoU_list = []
    for batch in tqdm(dataloader):
        images, annotations, _ = batch
        images = images.to(device)
        mask = annotations.to(device)
        if model.captioning:
            output, captions = model(images)
            print(captions)
        else:
            output = model(images)

        resized_output = F.interpolate(output.unsqueeze(0).float(), size=mask.shape[-2:], mode="nearest").to(device)
        mIoU = compute_best_mean_IoU(mask, resized_output)
        mIoU_list.append(mIoU.item())

    avg_mIoU = np.mean(mIoU_list)

    return avg_mIoU

def main():
    wandb.init(project='maskblip')
    score = pascal_miou(wandb.config)
    wandb.log({'score': score})

sweep_configuration = {
    'method': 'bayes',
    'metric':
    {
        'goal': 'maximize',
        'name': 'score'
        },
    'parameters':
    {
        'kmeans_range': {'values': [3, 4, 5, 6]},
        'pos_emb_dim': {'values': [256, 512, 768, 1024]},
        'smoothness_weight': {'min': 1.0, 'max': 10.0},
        'smoothness_theta': {'min': 0.5, 'max': 2.0},
        'nr_of_scales': {'values': [2, 3, 4, 5]},
        'scale_step': {'values': [32, 64, 128]}
     }
}

wandb.agent("a3k89xw3", function=main, count=20, project="maskblip")

