import os
import cv2
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

# Define the round_clip_0_1 function
def round_clip_0_1(images):
    return np.clip(images, 0, 1)

x_train_dir = os.path.join('./data', 'ct_images_dir')
y_train_dir = os.path.join('./data', 'mask_images_dir')

x_test_dir = os.path.join('./data', 'ct_images_test_dir')
y_test_dir = os.path.join('./data', 'mask_images_test_dir')

x_valid_dir = os.path.join('./data', 'ct_images_valid_dir')
y_valid_dir = os.path.join('./data', 'mask_images_valid_dir')

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig('./visualization-kaggle.png')


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (imgaug.Sequential): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (imgaug.Sequential): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    CLASSES = ['unlabeled', 'tumor']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask > 0)]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            seq_det = self.augmentation.to_deterministic()
            image = seq_det.augment_image(image)
            mask = seq_det.augment_image(mask)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integer number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# define augmentations using imgaug
def get_training_augmentation():
    train_transform = [
        iaa.Fliplr(0.5),
        iaa.Affine(scale=(0.5, 1.5), rotate=(-45, 45), translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        iaa.CropAndPad(percent=(-0.25, 0.25)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
        iaa.OneOf([
            iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.Grayscale(alpha=(0.0, 1.0)),
        ]),
    ]
    return iaa.Sequential(train_transform, additional_hooks=iaa.HooksImages(activator=round_clip_0_1))

def get_validation_augmentation():
    test_transform = [
        iaa.PadToFixedSize(384, 480)
    ]
    return iaa.Sequential(test_transform, additional_hooks=iaa.HooksImages(activator=round_clip_0_1))


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
      preprocessing_fn (callable): data normalization function
        (can be specific for each pretrained neural network)
      Return:
        transform: imgaug.Sequential

    """
    return iaa.Sequential([
        iaa.Lambda(image=preprocessing_fn),
    ])


# Define metrics
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['tumor']
LR = 0.0001
EPOCHS = 10

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# define optimizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# compile keras model with defined optimizer, loss and metrics
model.compile(optim, total_loss, metrics)

# Dataset for train images
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=CLASSES,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, 1)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

history = model.fit(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('./ct-losses-figs.png')

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

# load model
model.load_weights('best_model.h5')
