import os
import random
import pandas as pd
import cv2
import math
import decord
import numpy as np
import h5py
import torch
import pickle
from decord import VideoReader, cpu
decord.bridge.set_bridge('torch')
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import imutils
import timeit
    

class omniDataLoader(Dataset):
    def __init__(self, data_split, height=270, width=480, shuffle=True):
        self.video_path = '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb/'
        self.mask_path = '/home/siddiqui/Action_Biometrics-RGB/frame_data/ntu_rgbd_120_masks/'

        self.data_split = data_split
        if data_split == "train":
          self.annotations = '/home/siddiqui/Action_Biometrics-RGB/data/NTU60Train_CV-diffusion.csv'
        else:
          self.annotations = '/home/siddiqui/Action_Biometrics-RGB/data/NTU60Test_CV-diffusion.csv'

        self.rgb_list = [x for x in sorted(os.listdir(self.video_path)) if int(x[17:20]) < 6]
        self.pose_list = [x for x in sorted(os.listdir("/home/siddiqui/Action_Biometrics-RGB/frame_data/ntu_rgbd_120_skeletons/")) if x.endswith('.npy') and int(x[17:20]) < 6]
        self.poses = {}

#        for j, pose_path in enumerate(self.pose_list):
#            if j % 1000 == 0:
#                print(j)
#
#            pose = np.load(os.path.join("/home/siddiqui/Action_Biometrics-RGB/frame_data/ntu_rgbd_120_skeletons/", pose_path), allow_pickle=True).item()['rgb_body0'].astype(float)
#            self.poses[pose_path[:20]] = pose
        

        self.videos = []
        self.img_size = (224, 224)
        self.img_scale = (0.9, 1.) if data_split=='train' else (1.0, 1.0) 
        
        for count, row in enumerate(open(self.annotations, 'r').readlines()[1:]):
            if data_split == 'test':
                if count > 100000:
                    break
            row = row.replace('\n', '')
            if int(row.split('_')[-1]) == 0:
                continue
            action = row[17:20]
            if int(action) < 6:
                self.videos.append(row)
        
        # if self.data_split == 'train':
        remove = []
        for j, vid in enumerate(self.videos):
            parsed_vid = '_'.join(vid.split('_')[:-1])
            try:
                found = self.pose_list.index(f'{parsed_vid[:-8]}.skeleton.npy')
            except ValueError:
                remove.append(vid)
        for val in remove:
            self.videos.remove(val)
        if shuffle and data_split == 'train':
            random.shuffle(self.videos)
            
        # print(len(self.videos), len(self.rgb_list), self.videos[0])

        self.height = height
        self.width = width

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                (224,224),
                scale=self.img_scale, ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),  
            transforms.Normalize([0.5], [0.5]),
        ])
        self.ref_transform = transforms.Compose([ # follow CLIP transform
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=self.img_scale, ratio=(1., 1.),
                    interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                     [0.26862954, 0.26130258, 0.27577711]),
            ])
        
        self.ref_transform_mask = transforms.Compose([  # follow CLIP transform
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=self.img_scale, ratio=(1., 1.),
                    interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
                
            ])

        self.cond_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                self.img_size,
                scale=self.img_scale, ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BILINEAR),
        ])

        
    def __len__(self):
        return len(self.videos)
    

    def __getitem__(self, index):
        items = {}
        video_id = self.videos[index]
            
        frame, skeleton, ref_image, ref_image_controlnet, ref_image_vae, background_mask, background_mask_controlnet = item_creation(self, video_id, self.video_path, self.mask_path, self.height, self.width)
        items['img_key'] = video_id
        items['input_text'] = 'NULL'
        items['label_imgs'] = frame
        items['cond_imgs'] = skeleton
        items['reference_img'] = ref_image 
        items['reference_img_controlnet'] = ref_image_controlnet
        items['reference_img_vae'] = ref_image_vae
        items['background_mask'] = background_mask 
        items['background_mask_controlnet'] = background_mask_controlnet 

        
        return items
        
            
def item_creation(self, video_id, video_path, mask_path, height, width):
    # Read video & skeleton, select given frame 
    split = video_id.split('_')
    video_id, frame_ind = '_'.join(split[:-1]), int(split[-1])
  
    vr = VideoReader(os.path.join(video_path, video_id), height=height, width=width)
    
    frame = vr[frame_ind].float() / 255.
    ref_image = vr[0].float() / 255.
    
    skeleton = np.load(os.path.join("/home/siddiqui/Action_Biometrics-RGB/frame_data/ntu_rgbd_120_skeletons/", f'{video_id[:-8]}.skeleton.npy'), allow_pickle=True).item()['rgb_body0'][frame_ind].astype(float)
    #skeleton = self.poses[f'{video_id[:20]}'][frame_ind-1]
    skeleton = skeleton_to_image(skeleton)
    #skeleton = Image.open(f'/home/kzhai/frame_data/NTU_RGBD_120_poses/images/{video_id[:-8]}/{frame_ind}')
    #skeleton = Image.open(f'/home/siddiqui/Action_Biometrics-RGB/frame_data/ntu_rgbd_120_skeletons/{video_id[:-8]}/{frame_ind}')
    

    # Permute before transforms
    frame = frame.numpy()
    ref_image = ref_image.numpy()


    # Load masks, define reference information
    master_ref_mask = np.load(f'{mask_path}{video_id[:-8]}/mask{0}.npy').astype(float) #TODO: Seems like for now, first frame mask is used. I will start training with this, but should revisit
    master_ref_mask = np.stack([master_ref_mask, master_ref_mask, master_ref_mask], axis=2)
    # print(np.unique(master_ref_mask))
    ref_image_controlnet = ref_image.copy()
    

    # Apply transforms
    frame = self.transform(frame)
    ref_image = self.ref_transform(ref_image)
    ref_image_mask = self.ref_transform_mask(master_ref_mask)
    ref_image_controlnet = self.transform(ref_image_controlnet)
    ref_image_controlnet_mask = self.cond_transform(master_ref_mask)
    skeleton = self.cond_transform(skeleton)


    # Define vae after controlnet
    ref_image_vae = ref_image_controlnet.clone()
    

    #Mask applications
    ref_image *= ref_image_mask
    ref_image_controlnet *= (1 - ref_image_controlnet_mask)
    ref_image_vae *= ref_image_controlnet_mask


    #Define background masks
    background_mask = 1 - ref_image_mask
    background_mask_controlnet = 1 - ref_image_controlnet_mask

    #print(frame.shape, skeleton.shape, ref_image_controlnet.shape, ref_image_vae.shape, background_mask.shape, background_mask_controlnet.shape)

    return frame, skeleton, ref_image, ref_image_controlnet, ref_image_vae, background_mask, background_mask_controlnet


def skeleton_to_image(keypoint):
    limbSeq = [[1, 2], [2, 21], [3, 21], [4, 3], [5, 21], [6, 5],
                    [7, 6], [8, 7], [9, 21], [10, 9], [11, 10], [12, 11],
                    [13, 1], [14, 13], [15, 14], [16, 15], [17, 1], [18, 17],
                    [19, 18], [20, 19], [22, 23], [21, 21], [23, 8], [24, 25], [25, 12]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255,255,255], [255,255,255], [255,255,255], [255,255,255], [255,255,255], [255,255,255], [255,255,255]]


    canvas = np.zeros((1080, 1920, 3)).astype(np.uint8)
    # keypoint = ((keypoint) * 135) # Enlarge Pose
    # keypoint[:, 1] *= -1
    # keypoint[:, 0] += 200 # Horizontal Translation
    # keypoint[:, 1] += 150 # Vertical Translation
    stickwidth = 3
    for i in range(25):
        x, y = keypoint[i, 0:2]
        if x == -1 or y == -1:
            continue
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    joints = []
    for i in range(24):
        Y = keypoint[np.array(limbSeq[i])-1, 0]
        X = keypoint[np.array(limbSeq[i])-1, 1]            
        cur_canvas = canvas.copy()
        if -1 in Y or -1 in X:
            joints.append(np.zeros_like(cur_canvas[:, :, 0]))
            continue
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        joint = np.zeros_like(cur_canvas[:, :, 0])
        cv2.fillConvexPoly(joint, polygon, 255)
        joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
        joints.append(joint)

    pose = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    pose = transforms.CenterCrop([600, 750])(pose)
    return pose
        
        
if __name__ == '__main__':
    shuffle = False
    print('entered')
    data_generator = omniDataLoader('train', shuffle=shuffle)
    print('entered2')
    dataloader = DataLoader(data_generator, batch_size=4, num_workers=8, shuffle=False, drop_last=True)
    start = timeit.default_timer()
    for dict in tqdm(dataloader):
        pass
    end = timeit.default_timer()
    print(f'total time: {end-start}')
    