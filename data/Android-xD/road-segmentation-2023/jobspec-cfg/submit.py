import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset import CustomImageDataset
from models.deeplabv3 import createDeepLabv3
from models.fpn import get_fpn
from models.unet_backbone import get_Unet
from resample import resample

torch.manual_seed(0)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train image segmentation network')

    # model
    parser.add_argument('--model',
                        help='model',
                        type=str)

    # load model state for pth.tar file
    parser.add_argument('--load_model',
                        help='filepath to load the model out/*.pth.tar or None',
                        default='./out/model_best.pth.tar',
                        type=str)

    parser.add_argument('--threshold',
                        help='threshold on probability',
                        default=0.5,
                        type=float)

    parser.add_argument('--n_samples',
                        help='number of samples in TTA',
                        default=1,
                        type=int)



    # parse arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    """
    This script is used to generate the submission file for the Kaggle competition. 
    """
    # parse arguments
    args = parse_args()
    test_set = r"./data/test"

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()

    # Define the device to be used for computation
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset = CustomImageDataset(test_set, False, False, False, False)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    # set the model
    if args.model == 'deeplabv3':
        get_model = createDeepLabv3
    elif args.model == 'unet':
        get_model = get_Unet
    elif args.model == 'fpn':
        get_model = get_fpn
    else:
        raise ValueError('Invalid model name')

    # Load the model
    model, preprocess, postprocess = get_model(1, 400)
    state_dict = torch.load(args.load_model, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # Define the query function
    query = lambda input : postprocess(model(preprocess(input)))

    # Iterate over the data in the validation set and store the predictions
    store_folder = "out/prediction"
    os.makedirs(store_folder, exist_ok=True)
    for i, (input, image_filenames) in enumerate(val_loader):
        # Move input and target tensors to the device (CPU or GPU)
        input = input.to(device)
        input = input.squeeze()
        # output = query(input.unsqueeze(0)) without resample
        output = resample(query, test_set, i, args.n_samples)
    
        # normalize the output
        output = F.sigmoid(output[:,:1])
        pred = (255*(output > args.threshold)).detach().cpu().numpy().astype(np.uint8)
        j = 0
        # vis.output_target_heat(input.detach()[j] / 255, output.detach()[j, 1], 0.3, None)
        # plt.imshow(output[j, 1].detach().cpu().numpy())
        # plt.show()
        img_name = os.path.basename(image_filenames[j])
        print(img_name)
        cv2.imwrite(f"{store_folder}/mask_{img_name}", pred[j, 0])

    # now all masks are stored, convert it to the csv file
    # by running mask_to_submission.py
