import os
import torch
from segmentation_dataset import SegmentationDataset
from cutler_dataset import CutlerDataset
import wandb
from maskblip import MaskBLIP, plot_result
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import Compose, ToTensor, Normalize
from xdecoder_semseg import load_xdecoder_model, segment_image
from nlp import get_noun_chunks
from PIL import Image

def segment_with_sanity_check(xdecoder_model, images, captions, max_threshold=0.95, min_threshold=0.01, min_captions=3, plot=False, device='cuda:0'):
    noun_phrases = get_noun_chunks(captions, model.spacy_model)
    print("Noun phrases: ", noun_phrases)
    output = torch.tensor(
        segment_image(xdecoder_model, images, noun_phrases, plot=plot)).T.unsqueeze(0).to(device)

    while len(noun_phrases) >= min_captions:
        class_counts = torch.bincount(output.view(-1))

        total_pixels = float(output.numel())

        # Find the classes with occurrence more than max_threshold or less than min_threshold
        dominant_classes = ((class_counts / total_pixels) > max_threshold).nonzero(as_tuple=True)[0].tolist()
        minor_classes = ((class_counts / total_pixels) < min_threshold).nonzero(as_tuple=True)[0].tolist()

        # Check if there are any classes to remove
        if dominant_classes:
            # Remove the dominant classes from the list of captions and run again
            noun_phrases = [np for i, np in enumerate(noun_phrases) if i not in dominant_classes]
        elif minor_classes:
            # If no dominant classes, remove the minor classes
            noun_phrases = [np for i, np in enumerate(noun_phrases) if i not in minor_classes]
            print(minor_classes)
        else:
            # If no classes to remove, stop and return the output
            return output

        output = torch.tensor(segment_image(xdecoder_model, images, noun_phrases, plot=True)).T.unsqueeze(0).to(device)


    # If we reached here, it means there are less than min_captions left,
    # so just return the last resized_output we got
    return output

# def segment_with_sanity_check(xdecoder_model, images, phrases, threshold=0.95, min_captions=3, device='cuda:0'):
#     noun_phrases = phrases.copy()
#     while len(noun_phrases) >= min_captions:
#         resized_output = torch.tensor(segment_image(xdecoder_model, images, noun_phrases, plot=True)).T.unsqueeze(0).to(device)
#         class_counts = torch.bincount(resized_output.view(-1))
#
#         # Find the class index with maximum occurrence
#         max_occurrence_class_idx = class_counts.argmax().item()
#         max_occurrence = class_counts[max_occurrence_class_idx].item()
#
#         # Check if more than 95% of pixels are labeled with the same class
#         if max_occurrence / float(resized_output.numel()) <= threshold:
#             # If no class takes more than the threshold, stop and return the output
#             return resized_output
#         else:
#             # If a class is dominant, remove it from the list of captions and run again
#             print("Removing class {} with {} occurrences".format(max_occurrence_class_idx, max_occurrence))
#             noun_phrases.pop(max_occurrence_class_idx)
#
#     # If we reached here, it means there are less than min_captions left,
#     # so just return the last resized_output we got
#     return resized_output


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


if __name__ == "__main__":
    # torch.manual_seed(0)
    # np.random.seed(0)

    n_samples = 10
    batch_size = 1
    plot = True
    wandb_track = False
    supervised = True

    device = 'cpu'#("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MaskBLIP(device)
    captioning = True
    model.captioning = captioning
    use_xdecoder = True

    if use_xdecoder:
        xdecoder_model = load_xdecoder_model(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    if wandb_track:
        run = wandb.init(
            # Set the project where this run will be logged
            project="maskblip",
            group="multiscale",
            # Track hyperparameters and run metadata
            config={
                "batch_size": batch_size,
                "n_samples": n_samples,
            })
    else:
        run = wandb.init(mode = "disabled")

    transform = Compose([
    ToTensor(),
    #Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])
    normalize = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    if supervised:
        dataset_dir = os.path.join("datasets", "VOC2012")
        dataset = SegmentationDataset(dataset_dir, n_samples, transform=transform, img_size=model.output_size)
    else:
        dataset_dir = os.path.join("cutler", "maskcut")
        dataset = CutlerDataset(dataset_dir, n_samples, transform=transform, img_size=model.output_size)


    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    mIoU_list = []
    for batch in tqdm(data_loader):
        raw_images, annotations, paths = batch
        raw_images = raw_images.to(device)
        images = normalize(raw_images)

        mask = annotations.to(device)
        if model.captioning:
            output, captions = model(images, clean=True)
            print(captions)
        else:
            output = model(images)

        if plot:
            plot_result(images.detach(), output, captions)

        if use_xdecoder:
            noun_phrases = get_noun_chunks(captions[0], model.spacy_model)
            raw_image = Image.open(paths[0])
            resized_output = segment_with_sanity_check(xdecoder_model, raw_image, noun_phrases, device=device, plot=plot)
            #resized_output = torch.tensor(segment_image(xdecoder_model, images, noun_phrases, input_tensor=True, plot=True)).T.unsqueeze(0).to(device)
        else:
            resized_output = F.interpolate(output.unsqueeze(0).float(), size=mask.shape[-2:], mode="nearest").to(device)
        mIoU = compute_best_mean_IoU(mask, resized_output)
        mIoU_list.append(mIoU.item())
        print("mIoU: {}".format(mIoU.item()))



    print("Average mIoU: {}".format(sum(mIoU_list) / len(data_loader)))
    num_bins = 20
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    # We can set the number of bins with the `bins` argument
    axs[0].hist(mIoU_list, bins=num_bins, edgecolor='black')
    plt.show()
    plt.savefig("mIoU_hist.png")
