import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from network import *


class ImageDataset(Dataset):
    def __init__(self, indices, path):
        self.path = path
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_number = self.indices[idx]
        #Load image
        image_path = self.path + "/images/" + str(img_number) + ".png"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        #Load label
        label_path = self.path + "/masks/" + str(img_number) + ".png"
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) 

        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512))

        image = image.astype(np.float32) / 255.0 
        label = label.astype(np.float32) / 255.0  
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        
        return image, label


def get_f1_score(predicted, mask):
    predicted_flat = predicted.view(-1)  # flatten the predicted image
    mask_flat = mask.view(-1)  # flatten the mask
    TP = torch.sum(predicted_flat * mask_flat)  # true positives
    FP = torch.sum(predicted_flat) - TP  # false positives
    FN = torch.sum(mask_flat) - TP  # false negatives
    precision = TP / (TP + FP + 1e-7)  # precision
    recall = TP / (TP + FN + 1e-7)  # recall
    F1 = 2 * (precision * recall) / (precision + recall + 1e-7)  # F1 score
    return F1.item()  # convert tensor to scalar value

def get_iou_score(predicted, mask):
    intersection = torch.logical_and(predicted, mask).sum().item()
    union = torch.logical_or(predicted, mask).sum().item()
    iou = intersection / union if union > 0 else 0
    return iou

def tensor_to_image(tensor, image):
    # Assuming tensor is a torch tensor
    # Convert torch tensor to numpy array
    numpy_image = tensor.squeeze().cpu().numpy()
    # print(numpy_image.shape, len(numpy_image.shape))

    if image:
        gray_array = np.mean(numpy_image, axis=0).astype(np.uint8)
        gray_array = np.squeeze(gray_array)
        pil_image = Image.fromarray(gray_array.astype(np.uint8), mode='L')
    else:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray((numpy_image * 255).astype(np.uint8))
    return pil_image

def save_images(image, label, output_binary, idx):
    # Convert torch tensors to PIL images
    image_pil = tensor_to_image(image, True)
    label_pil = tensor_to_image(label, False)
    output_binary_pil = tensor_to_image(output_binary, False)

    # print("shapes", image.shape, label.shape, output_binary.shape)

    #Save images
    results_folder = "results/images"
    image_pil.save(os.path.join(results_folder, f"image_{idx}.png"))
    label_pil.save(os.path.join(results_folder, f"label_{idx}.png"))
    output_binary_pil.save(os.path.join(results_folder,f"output_{idx}.png"))


def test(model, image_indices, path):
    total_f1_score = 0
    total_iou_score = 0

    test_images = ImageDataset(image_indices, path)
    test_set = DataLoader(test_images, batch_size=1, shuffle=False, num_workers=4)

    for idx, (image, label) in enumerate(test_set):
        with torch.no_grad():

            image = image.permute(0, 3, 1, 2)
            output = model(image.float().to('cuda'))
            label = label.float().to('cuda')
            
            output_binary = (output > 0.5).float()
            score = get_f1_score(output_binary, label)
            score_iou = get_iou_score(output_binary, label)
            #print(score)
            total_f1_score = total_f1_score + score
            total_iou_score = total_iou_score + score_iou

            # Save images
            save_images(image, label, output_binary, idx)

    total_f1_score = total_f1_score / len(image_indices)
    total_iou_score = total_iou_score / len(image_indices)

    print("Mean f1 score:", total_f1_score)
    print("Mean IoU score", total_iou_score)
    return total_f1_score, total_iou_score


if __name__ == '__main__':
    model = UnetWithHeader(n_channels=3, n_classes=1, mode="mlp")
    model = model.cuda()
    f1_scores = []
    iou_scores = []


    for i in range(0,5):
        state_dict = torch.load("results/unet-ss_no_simclr_fold_" + str(i) + ".pth", map_location=torch.device('cuda:0'))
        model.load_state_dict(state_dict, strict=True)
        image_indices = list(range(0, 307))
        path = "brain_tumour/test"
        f1_score, iou_score = test(model, image_indices, path)

        f1_scores.append(f1_score)
        iou_scores.append(iou_score)

    # Calculate mean
    mean_f1 = np.mean(f1_scores)
    mean_iou = np.mean(iou_scores)

    # Calculate standard deviation
    std_value_f1 = np.std(f1_scores)
    std_value_iou = np.std(iou_scores)

    print("Mean:", mean_f1, "\t", mean_iou)
    print("Standard Deviation:", std_value_f1, "\t", std_value_iou)

