import os


from EMU import EMAU

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from train import load_dataset, create_dir, get_colormap

""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP


def grayscale_to_rgb(mask, mask_values, colormap):
    h, w = mask.shape
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(colormap[mask_values[pixel]])

    output = np.reshape(output, (h, w, 3))
    # output = np.stack((output,) * 3, axis=-1)
    return output


def save_results(image, mask, pred, save_image_path):
    h, w = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    line = np.ones((h, 10, 3 )) * 255

    pred = grayscale_to_rgb(pred, MASK_VALUES, COLORMAP)

    mask = grayscale_to_rgb(mask/10-1, MASK_VALUES, COLORMAP)
    # mask = np.stack((mask,) * 3, axis=-1)

    cat_images = np.concatenate([image, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Hyperparameters """
    IMG_H = 256
    IMG_W = 256
    NUM_CLASSES = 9

    model_path = os.path.join("files", "model.h5")

    """ Colormap """
    CLASSES, MASK_VALUES, COLORMAP = get_colormap()

    """ Model """
    # model = tf.keras.models.load_model(model_path)
    model = tf.keras.models.load_model(model_path, custom_objects={'EMAU':EMAU})
    """ Load the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset()
    print(
        f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_y)}")
    print("")

    """ Evaluation and Prediction """
    SCORE = []
    i = len(train_x) + len(valid_x) + 1

    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = str(i)
        i += 1

        """ Reading the image """
        image_x = x
        image = image_x / 255.0
        image = np.expand_dims(image, axis=0)

        mask_x = y
        onehot_mask = []
        for color in MASK_VALUES:
            cmap = np.equal(y, color)
            onehot_mask.append(cmap)
        onehot_mask = np.stack(onehot_mask, axis=-1)
        onehot_mask = np.argmax(onehot_mask, axis=-1)
        onehot_mask = onehot_mask.astype(np.int32)

        """ Prediction """
        pred = model.predict(image, verbose=0)[0]
        pred = np.argmax(pred, axis=-1)
        pred = pred.astype(np.float32)

        """ Saving the prediction """
        save_image_path = f"results/{name}.png"
        save_results(image_x, mask_x, pred, save_image_path)

        """ Flatten the array """
        onehot_mask = onehot_mask.flatten()
        pred = pred.flatten()

        labels = [i for i in range(NUM_CLASSES)]

        """ Calculating the metrics values """
        f1_value = f1_score(onehot_mask, pred, labels=labels, average=None, zero_division=0)
        jac_value = jaccard_score(onehot_mask, pred, labels=labels, average=None, zero_division=0)
        # acc_value = accuracy_score(onehot_mask, pred)

        SCORE.append([f1_value, jac_value])
        # SCORE.append([f1_value, jac_value, acc_value])

    """ Metrics values """
    score = np.array(SCORE)
    score = np.mean(score, axis=0)

    f = open("files/score.csv", "w")
    f.write("Class,F1,Jaccard,Accuracy\n")

    l = ["Class", "F1", "Jaccard", "Accuracy"]
    print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s}")
    print("-" * 35)

    for i in range(score.shape[1]):
        class_name = CLASSES[i]
        f1 = score[0, i]
        jac = score[1, i]
        # acc = score[2, i]
        dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
        print(dstr)
        f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")
        # dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f} - {acc:1.5f}"
        # print(dstr)
        # f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f},{acc:1.5f}\n")

    print("-" * 35)
    class_mean = np.mean(score, axis=-1)
    class_name = "Mean"
    f1 = class_mean[0]
    jac = class_mean[1]
    # acc = class_mean[2]
    dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
    print(dstr)
    f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")

    f.close()
