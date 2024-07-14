import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--spatial', action='store_true')
parser.add_argument('--real', action='store_true')
parser.add_argument('--weights', type=str)

args = parser.parse_args()

print('='*100)
print(f'Weights path = {args.weights}')
print(f'Ran on real data = {args.real}')
print(f'Ran on spatial split = {args.spatial}')
print('='*100)

WEIGHTS_PATH = args.weights

###################################################################################
# NO CHANGES REQUIRED BELOW
###################################################################################

# Whether testing on real or synthetic
if args.real:
  PATH_TO_DATASET = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/synthetic_data/split_datasets/RefCOCO_3ds_7k/val'
  PATH_TO_DATASET_IMAGES = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/synthetic_data/split_datasets/RefCOCO_3ds_7k/val/images/'
  FILE_EXTENSION = '.png'
else:
  if args.spatial:
    spatial_str = ''
  else:
    spatial_str = 'non'
  PATH_TO_DATASET = '/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refcoco_split/' + spatial_str + 'spatial' # no slash on end
  PATH_TO_DATASET_IMAGES =  "/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/refer_data/images/mscoco/images/train2014/COCO_train2014_000000" # images are named with their numbers.png, so they get appended to this
  FILE_EXTENSION = '.jpg'

# Rest of script
import sys
sys.path.insert(0,'./GroundingDINO/')

# TODO: Specify weights .pth file
HOME = "."

CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print("\n",CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

#WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

###################################################################################
###################################################################################

# TODO: Specify device (e.g. cuda, mps, cpu)
DEVICE = 'cuda'

from groundingdino.util.inference import load_model, load_image, predict, annotate
model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)

###################################################################################
###################################################################################

# TODO: Specify path to the dataset
# dataset
# ---> /images
# ---> instances.json
# ---> refs.json
  # if using Google Colab
# PATH_TO_DATASET = os.getcwd() + "/test" # IF /test IS IN HOME DIRECTORY

###################################################################################
###################################################################################

PATH_TO_DATASET_INSTANCES_FILE =   PATH_TO_DATASET + "/instances.json"
PATH_TO_DATASET_REFS_FILE =   PATH_TO_DATASET + "/refs.json"

print(f"Successfully Found {PATH_TO_DATASET}") if os.path.isdir(PATH_TO_DATASET) else print(f"Did not find {PATH_TO_DATASET}")
print(f"Successfully Found {PATH_TO_DATASET_IMAGES}") if os.path.isdir(PATH_TO_DATASET_IMAGES) else print(f"Did not find {PATH_TO_DATASET_IMAGES}")
print(f"Successfully Found {PATH_TO_DATASET_INSTANCES_FILE}") if os.path.isfile(PATH_TO_DATASET_INSTANCES_FILE) else print(f"Did not find {PATH_TO_DATASET_INSTANCES_FILE}")
print(f"Successfully Found {PATH_TO_DATASET_REFS_FILE}") if os.path.isfile(PATH_TO_DATASET_REFS_FILE) else print(f"Did not find {PATH_TO_DATASET_REFS_FILE}")

###################################################################################
###################################################################################

import json
instances_file = open(PATH_TO_DATASET_INSTANCES_FILE)
instances_dict = json.load(instances_file)
refs_file = open(PATH_TO_DATASET_REFS_FILE)
refs_list = json.load(refs_file)

# find mapping for image_name to annotation
image_to_annotations_map = {}
for anno_id, anno in enumerate(instances_dict["annotations"]):
 image_id = instances_dict["annotations"][anno_id]["image_id"]
 if image_id in list(image_to_annotations_map.keys()):
  image_to_annotations_map[image_id].append(anno_id)
 else:
  image_to_annotations_map[image_id] = [anno_id]

# CREATE MAPPING FOR CATEGORY IDE TO CATEGORY NAME
category_id_to_name_map = {}
for category in instances_dict["categories"]:
  category_id_to_name_map[category["id"]] = category["name"]

# CREATE MAPPING FOR ANNOTATION_ID TO ANNOTATION INDEX IN REFS.JSON
ann_id_to_refs_index = {}
for ref_index, ref in enumerate(refs_list):
  ann_id_to_refs_index[ref['ann_id']] = ref_index

# CREATE MAPPING FOR IMAGE_ID TO IMAGE INDEX IN INSTNACES.JSON
image_id_to_instance_image_idx_map = {}
for image_idx, image in enumerate(instances_dict["images"]):
  image_id_to_instance_image_idx_map[image["id"]] = image_idx

###################################################################################
###################################################################################

# SHOW THAT WE CAN RETRIEVE ALL GROUND TRUTH BOUNDING BOXES AND PHRASES FOR EACH IMAGE
import torch
import copy
from tqdm import tqdm

img_h, img_w = [instances_dict["images"][0]["height"],instances_dict["images"][0]["width"]]
images_in_dataset = list(image_to_annotations_map.keys())

# loop through all images in specified dataset
#for image_id in tqdm(images_in_dataset):
for image_id in images_in_dataset:

  num_annotations = len(image_to_annotations_map[image_id]) # number of annotations for image

  # initialize bounding boxes and phrases
  boxes = torch.zeros((num_annotations,4))
  phrases = []

  # loop through all annotations for the image
  for bbox_index, anno_id in enumerate(image_to_annotations_map[image_id]):

    # convert pixel bounding box (xmin, ymin, w, h) to percentage bounding box (xcenter,ycenter,w,h)
    bbox_pix = instances_dict["annotations"][anno_id]["bbox"]
    bbox_pix_shifted = copy.deepcopy(bbox_pix)
    bbox_pix_shifted[0] += bbox_pix_shifted[2]/2
    bbox_pix_shifted[1] += bbox_pix_shifted[3]/2
    bbox_per = torch.tensor([bbox_pix_shifted[0]/img_w, bbox_pix_shifted[1]/img_h, bbox_pix_shifted[2]/img_w, bbox_pix_shifted[3]/img_h])
    boxes[bbox_index,:] = bbox_per


    phrases.append(str(instances_dict["annotations"][anno_id]["category_id"])) # for visualization

  logits = torch.zeros(num_annotations) # for visualization

###################################################################################
###################################################################################

# VISUALIZE THE GROUND TRUTH BOUNDING BOXES AND CATEGORY ID'S FOR THE LAST IMAGE THAT WE LOOPED THROUGH IMAGE
import supervision as sv

IMAGE_NAME = str(image_id).zfill(6) + FILE_EXTENSION
IMAGE_PATH = PATH_TO_DATASET_IMAGES + IMAGE_NAME
image_source, image = load_image(IMAGE_PATH)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

sv.plot_image(annotated_frame, (16, 16))

###################################################################################
###################################################################################

# VISUALIZE GROUND TRUTH AND ALL PREDICTIONS FOR NUM_PHRASES phrases in NUM_IMAGES images
from tqdm import tqdm
import random

# TDOO: Specify number of images
NUM_IMAGES = 1
NUM_PHRASES = 2
BOX_TRESHOLD = 0.10 # 35
TEXT_TRESHOLD = 0.10  #25


img_h, img_w = [instances_dict["images"][0]["height"],instances_dict["images"][0]["width"]]
images_in_dataset = list(image_to_annotations_map.keys())

# loop through all images in specified dataset
for iamge_count, image_id in enumerate(tqdm(images_in_dataset[0:NUM_IMAGES])):

  IMAGE_NAME = str(image_id).zfill(6) + FILE_EXTENSION
  IMAGE_PATH = PATH_TO_DATASET_IMAGES + IMAGE_NAME

  num_annotations = len(image_to_annotations_map[image_id]) # number of annotations for image

  # loop through all ground truth bounding boxes for the image
  for bbox_index, anno_id in enumerate(image_to_annotations_map[image_id][0:NUM_PHRASES]):

    print(f"\nImage {iamge_count} on phrase {bbox_index}")

    # find the ground truth bounding box
    # convert pixel bounding box (xmin, ymin, w, h) to percentage bounding box (xcenter,ycenter,w,h)
    bbox_pix = instances_dict["annotations"][anno_id]["bbox"]
    bbox_pix_shifted = copy.deepcopy(bbox_pix)
    bbox_pix_shifted[0] += bbox_pix_shifted[2]/2
    bbox_pix_shifted[1] += bbox_pix_shifted[3]/2
    box_gt = torch.tensor([bbox_pix_shifted[0]/img_w, bbox_pix_shifted[1]/img_h, bbox_pix_shifted[2]/img_w, bbox_pix_shifted[3]/img_h])

    # choose randome annotation phrase
    anno_sentences = refs_list[anno_id]['sentences']
    num_sentences = len(anno_sentences)
    TEXT_PROMPT = anno_sentences[random.randint(0,num_sentences-1)]['sent']

    # find the predicted bounding boxes
    image_source, image = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device = DEVICE
    )

    boxes = torch.vstack((boxes,box_gt))
    new = torch.tensor([-1.0])
    logits = torch.cat((logits,new))
    phrases.append("GT")

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    sv.plot_image(annotated_frame, (16, 16))

###################################################################################
###################################################################################


import torch
import copy
from tqdm import tqdm
import os
import supervision as sv
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint

# TODO: Specify the following evaluation parameters
BOX_TRESHOLD = 0.10 # 35
TEXT_TRESHOLD = 0.10  #25


img_h, img_w = [instances_dict["images"][0]["height"],instances_dict["images"][0]["width"]]

def convert_pix_coords_to_absolute_pixels(bbox,img_w,img_h):
  bbox[0] *= img_w
  bbox[2] *= img_w
  bbox[1] *= img_h
  bbox[3] *= img_h

  bbox = bbox.int()
  bbox = bbox.float().reshape(1,-1)

  return bbox


metric_1 = MeanAveragePrecision(box_format = 'cxcywh', iou_type="bbox")
metric_2 = MeanAveragePrecision(box_format = 'cxcywh', iou_type="bbox")
metric_3 = MeanAveragePrecision(box_format = 'cxcywh', iou_type="bbox")

BOX_TRESHOLD = 0.10 # 35
TEXT_TRESHOLD = 0.10  #25

images_in_dataset = list(image_to_annotations_map.keys())

# loop through all images in specified dataset
#for image_id in tqdm(images_in_dataset):
#for image_id in tqdm(images_in_dataset): # TODO: Specify images for debugging
for image_id in images_in_dataset:

  IMAGE_NAME = str(image_id).zfill(6) + FILE_EXTENSION
  IMAGE_PATH = PATH_TO_DATASET_IMAGES + IMAGE_NAME

  num_annotations = len(image_to_annotations_map[image_id]) # number of annotations for image

  img_h = instances_dict["images"][image_id_to_instance_image_idx_map[image_id]]['height']
  img_w = instances_dict["images"][image_id_to_instance_image_idx_map[image_id]]['width']

  # loop through all ground truth bounding boxes for the image
  # for bbox_index, anno_id in enumerate(image_to_annotations_map[image_id]):
  for bbox_index, anno_id in enumerate(image_to_annotations_map[image_id][0:1]):  # TODO: Specify annotaions for debugging

    # find the ground truth bounding box
    # convert pixel bounding box (xmin, ymin, w, h) to percentage bounding box (xcenter,ycenter,w,h)
    bbox_pix = instances_dict["annotations"][anno_id]["bbox"]
    bbox_pix_shifted = copy.deepcopy(bbox_pix)
    bbox_pix_shifted[0] += bbox_pix_shifted[2]/2
    bbox_pix_shifted[1] += bbox_pix_shifted[3]/2
    box_gt = torch.tensor([bbox_pix_shifted[0]/img_w, bbox_pix_shifted[1]/img_h, bbox_pix_shifted[2]/img_w, bbox_pix_shifted[3]/img_h])

    # choose randome annotation phrase
    anno_sentences = refs_list[anno_id]['sentences']
    num_sentences = len(anno_sentences)
    TEXT_PROMPT = anno_sentences[random.randint(0,num_sentences-1)]['sent']


    # find the predicted bounding boxes
    image_source, image = load_image(IMAGE_PATH)
    boxes_pred, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE
    )

    box_gt = convert_pix_coords_to_absolute_pixels(box_gt,img_w,img_h)
    target = [dict(boxes=box_gt,
                  labels=tensor([0]),)]

    if boxes_pred.shape[0] >= 1:
      box_pred = convert_pix_coords_to_absolute_pixels(boxes_pred[0,:],img_w,img_h)
      preds = [dict(
       boxes=box_pred,
       scores=logits[0].reshape(1),
       labels=tensor([0]),)]
      metric_1.update(preds,target)

    if boxes_pred.shape[0] >= 2:
      box_pred = convert_pix_coords_to_absolute_pixels(boxes_pred[1,:],img_w,img_h)
      preds = [dict(
       boxes=box_pred,
       scores=logits[0].reshape(1),
       labels=tensor([0]),)]
      metric_2.update(preds,target)

    if boxes_pred.shape[0] >= 3:
      box_pred = convert_pix_coords_to_absolute_pixels(boxes_pred[2,:],img_w,img_h)
      preds = [dict(
       boxes=box_pred,
       scores=logits[0].reshape(1),
       labels=tensor([0]),)]
      metric_3.update(preds,target)

    # visualize
    # boxes_pred = torch.vstack((boxes_pred,box_gt))
    # new = torch.tensor([-1.0])
    # logits = torch.cat((logits,new))
    # phrases.append("GT")
    # annotated_frame = annotate(image_source=image_source, boxes=boxes_pred, logits=logits, phrases=phrases)
    # %matplotlib inline
    # sv.plot_image(annotated_frame, (16, 16))

print()
print("1")
pprint(metric_1.compute())
print("2")
pprint(metric_2.compute())
print("3")
pprint(metric_3.compute())