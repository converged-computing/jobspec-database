from statistics import mode
import keras as k
from tensorflow.keras.models import Model
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from data_loader import MAMeDataset
import matplotlib.pyplot as plt
from keras import applications

IMG_SIZE = 256
NUM_CLASSES = 29

def create_model():
  model = applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
    classes=NUM_CLASSES
  )

  # example of freezing layers
  for layer in model.layers[:170]:
    layer.trainable = False

  
  # custom model example
  #Adding custom Layers 
  x = model.output
  x = Flatten()(x)
  x = Dense(512, activation="relu")(x)
  x = Dropout(0.2)(x)
  x = Dense(512, activation="relu")(x)
  predictions = Dense(NUM_CLASSES, activation="softmax")(x)

  # creating the final model 
  model_final = Model(model.input, predictions)

  return model_final