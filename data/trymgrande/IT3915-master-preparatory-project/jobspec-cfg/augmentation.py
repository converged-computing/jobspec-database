"""
This script is made with the implementation of the augmentation library Albumentations, as well as a dataset stored using Roboflow. It downloads a copy of the dataset and performs augmentation on the first given number of images, that are then added to the local copy of the dataset.
"""

import sys
import random
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A
from cmath import inf
from PIL import Image
import os, os.path
import shutil

# set parameters
PROJECT_NAME = "Merged-sheep-dataset"
PROJECT_VERSION = 3
PROJECT_DIRECTORY = f"{PROJECT_NAME}-{PROJECT_VERSION}"

AUGMENTATION = True
# AUGMENTATIONS = float('inf')
AUGMENTATIONS = 3250



# download dataset
print("downloading dataset...")
from roboflow import Roboflow
rf = Roboflow(api_key="Sf4Q132h8vYyzxbVfF7t")
project = rf.workspace("it3915masterpreparatoryproject").project("merged-sheep-dataset")
dataset = project.version(PROJECT_VERSION).download("yolov7")
print("dataset downloaded")


# augmentation plot utils

def plot_examples(images, bboxes=None):
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 5

    for i in range(1, len(images)):
        if bboxes is not None:
            img = visualize_bbox(images[i - 1], bboxes[i - 1], class_name="sheep")
        else:
            img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


# From https://albumentations.ai/docs/examples/example_bboxes/
def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=5):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = map(int, bbox)
    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)
    
    return img

# https://albumentations.ai/docs/examples/example_bboxes2/

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)

# copied from: https://www.kaggle.com/code/reighns/augmentations-data-cleaning-and-bounding-boxes/notebook

def draw_rect_with_labels(img, bboxes,class_id, class_dict={1: 'sheep'}, color=None):
    img = img.copy()
    bboxes = bboxes[:, :4]
    bboxes = bboxes.reshape(-1, 4)
    for bbox, label in zip(bboxes, class_id):
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])
        class_name = class_dict[label]
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1) 
        img = cv2.rectangle(img.copy(), pt1, pt2, color, int(max(img.shape[:2]) / 200))
        img = cv2.putText(img.copy(), class_name, (int(bbox[0]), int(bbox[1]) - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color = (255,255,255), lineType=cv2.LINE_AA)
    return img

# augmentation classification

from augmentation_transformations import transformations_dict

transformation_number = sys.argv[1]
print(f"starting augmentation with transformation number {transformation_number}...")
transformation = transformations_dict[int(transformation_number)]
CLASS_LABEL = 'sheep'
CLASS = 0

# load train images
images = []
images_load_path = f"./{PROJECT_DIRECTORY}/train/images/"
valid_images = [".jpg",".gif",".png",".tga"]

LIMIT = AUGMENTATIONS
i = 0
for image_file_name in os.listdir(images_load_path):
    if i == LIMIT:
        break
    ext = os.path.splitext(image_file_name)[1]
    if ext.lower() not in valid_images:
        continue
    images.append(cv2.imread(os.path.join(images_load_path, image_file_name)))
    i += 1


# load train labels
labels = [] # one label per image, multiple bboxes per image
labels_load_path = f"./{PROJECT_DIRECTORY}/train/labels/"
i = 0
for label_file_name in os.listdir(labels_load_path):
    if i == LIMIT:
        break
    label = open(os.path.join(labels_load_path, label_file_name), "r")
    bboxes = []
    for bbox in label:
        bbox = np.array(bbox.split(' ')[1:]).astype(np.float32)
        bboxes.append(bbox)
    labels.append(bboxes)
    i += 1



# augment images using transformation

augmented_labels = []
augmented_images_nd = []
for image, label in zip(images, labels):
    label = np.array(label)

    transform = A.Compose(
        transformation,
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )
    
    augmentation = transform(image=image, bboxes=label, class_labels=[CLASS_LABEL]*len(label))
    augmented_img = augmentation["image"]
    augmented_label = augmentation["bboxes"]
    augmented_labels.append(augmented_label)
    # augmented_category_ids = [int(CLASS_LABEL)]*len(label)
    augmented_images_nd.append(augmented_img)
    
    # draw_image = draw_rect_with_labels(augmented_img, label, [1]*len(label))

    # visualize(
    #     augmented_img,
    #     augmented_label,
    #     augmented_category_ids,
    #     {1: 'sheep', 2: 'sheep', 3: 'sheep', 0: 'sheep'}
    # )

    # plt.figure()
    # plt.imshow(augmented_img)


    # augment_and_show(transform, image)


# plot_examples(augmented_images_nd) # comment out to save runtime


augmented_images = [Image.fromarray(augmented_image_nd) for augmented_image_nd in augmented_images_nd]

# create image save dir
img_save_dir = f"./{PROJECT_DIRECTORY}/train/images"


# save augmented images
for i in range(len(augmented_images)):
    image_file_name = f"{i}.png"
    image_dir = os.path.join(img_save_dir, image_file_name)
    augmented_images[i].save(image_dir)
    print(f"saved augmented image: {image_dir}")


# create labels save dir
labels_save_dir = f"./{PROJECT_DIRECTORY}/train/labels"

# write augmented labels
for i in range(len(augmented_labels)):
    label_file_name = f"{i}.txt"
    label_dir = os.path.join(labels_save_dir, label_file_name)
    with open(label_dir, 'w') as f:
        for bbox in augmented_labels[i]:
            f.write(f"{CLASS} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    print(f"saved augmented label: {label_dir}")

print("finished augmentation")