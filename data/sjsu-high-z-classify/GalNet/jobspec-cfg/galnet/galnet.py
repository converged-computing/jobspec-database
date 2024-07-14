#!/usr/bin/env python
# coding: utf-8

# GalNet is DNN built for galaxy classification with GZ2.
# Copyright (C) 2018-2019  J. Andrew Casey-Clyde and Jean Donet.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import argparse
import uuid

import pandas as pd

from keras.callbacks import (CSVLogger, EarlyStopping,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.models import load_model
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model, multi_gpu_utils, plot_model

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from model import model_builder

# top level hyperparameter definitions
INPUT_DIM = (69, 69)
DATA_FORMAT = 'channels_last'
BATCH_SIZE = 256
EPOCHS = 200


def test_model(data, model, class_cols, model_path):
    """Tests models that have already been initialized and trained.

    Parameters
    ----------
        data : :obj:`pandas.core.frame.DataFrame`
            The data frame containing testing data formatted for use
            with :func:`ImageDataGenerator.flow_from_dataframe`
        model : :obj:`keras.models.Model`
            The model to test.
        class_cols : list of str
            The dataframe columns containing the classes. For use with
            multi-output models.
        model_path : str
            Path where model is saved. Used to store test results with
            model.

    Returns
    -------
    score : list of float
        The model's test scores.

    """
    testgen = ImageDataGenerator()
    testgen = testgen.flow_from_dataframe(data,
                                          directory=MODULE_PATH,
                                          x_col='imgpath',
                                          y_col=class_cols,
                                          batchsize=BATCH_SIZE,
                                          target_size=INPUT_DIM,
                                          class_mode='other')

    test_step_size = testgen.n // testgen.batch_size

    score = model.evaluate_generator(testgen,
                                     steps=test_step_size,
                                     verbose=1)
    with open(os.path.join(model_path, 'test.results'), 'a') as f_res:
        print("loss: {0:.4f}".format(score[0]), file=f_res)
        print("acc: {0:.4f}".format(score[1]), file=f_res)

    return score


def train_model(data, class_cols, model_path):
    """Trains a model for galactic image recognition.

    Parameters
    ----------
        data : :obj:`pandas.core.frame.DataFrame`
            The data frame containing training data formatted for use
            with :func:`ImageDataGenerator.flow_from_dataframe`. Data
            will automatically be broken into training and validaiton
            sets.
        class_cols : list of str
            The dataframe columns containing the classes. For use with
            multi-output models.
        model_path : str
            Path where model should be saved.

    Returns
    -------
    model : :obj:`keras.models.Model`
        The trained model.

    """
    # create an ImageDataGenerator, which applies random affine
    # transformations to the data. such augmentation is standard
    datagen = ImageDataGenerator(rotation_range=360,
                                 zoom_range=[.75, 1.3],
                                 width_shift_range=.05,
                                 height_shift_range=.05,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 validation_split=.25)

    # create two sets of generators, one for training data and one for
    # validation data, which can be used to check progress throughout
    # training. the target_size option automatically scales our data to
    # the requested size. We also set up for a multi-output model, even
    # though we are currently only checking one question, which will
    # allow some flexibility should this goal change
    traingen = datagen.flow_from_dataframe(data,
                                           directory=MODULE_PATH,
                                           x_col='imgpath',
                                           y_col=class_cols,
                                           batchsize=BATCH_SIZE,
                                           target_size=INPUT_DIM,
                                           class_mode='other',
                                           subset='training')

    valgen = datagen.flow_from_dataframe(data,
                                         directory=MODULE_PATH,
                                         x_col='imgpath',
                                         y_col=class_cols,
                                         batchsize=BATCH_SIZE,
                                         target_size=INPUT_DIM,
                                         class_mode='other',
                                         subset='validation')

    # now we actually build the model, which is defined in model.py
    model = model_builder(input_dim=traingen.image_shape)

    # save an image of the model as defined in model.py. can be useful
    # for quickly checking that you have the architecture you want.
    # note that this has to happen before we distribute over gpu.
    plot_model(model, to_file=os.path.join(model_path, 'model.png'),
               show_shapes=True, show_layer_names=True)

    # calculate the number of steps per epoch (or validation) such that
    # all (or nearly all) images are used
    train_step_size = traingen.n // traingen.batch_size
    val_step_size = valgen.n // valgen.batch_size

    # set up callbacks for saving and logging
    # XXX: will need to append history if we continue training a model
    monitor = 'val_loss'  # should monitor the same quanitity for all
    base_patience = 10  # ensure we try LR reduction a few times before stop

    checkpoint = ModelCheckpoint(os.path.join(model_path, 'model.h5'),
                                 monitor=monitor, save_best_only=True)
    csv_logger = CSVLogger(os.path.join(model_path, 'training.log'))
    lr_plateau = ReduceLROnPlateau(monitor=monitor, factor=0.1,
                                   patience=base_patience, min_lr=0.)
    stop = EarlyStopping(monitor=monitor, patience=5*base_patience)

    # set up for a multi-gpu model
    # HACK: fixes an issue in keras where these don't play nice with
    #       xla_gpus (which cause double counting of available gpus).
    available_devices = [multi_gpu_utils._normalize_device_name(name)
                         for name
                         in multi_gpu_utils._get_available_devices()]

    # this line is our actual keras fix; it's the '/' that's key
    n_gpus = len([x for x in available_devices if '/gpu' in x])

    if n_gpus > 1:  # only use multi_gpu if we have multiple gpus
        parallel_model = multi_gpu_model(model, gpus=n_gpus)

        # compile the model. note that the names of outputs in dicts (e.g.,
        # 't01') should match the names of the relevant output layers found
        # in the model definition
        parallel_model.compile(optimizer=Nadam(lr=0.0001),
                               loss={'t01': 'categorical_crossentropy'},
                               loss_weights={'t01': 1.},
                               metrics=['accuracy'])

        # train the model
        history = parallel_model.fit_generator(generator=traingen,
                                               steps_per_epoch=train_step_size,
                                               validation_data=valgen,
                                               validation_steps=val_step_size,
                                               epochs=EPOCHS,
                                               callbacks=[checkpoint,
                                                          csv_logger,
                                                          lr_plateau,
                                                          stop],
                                               verbose=1)

    else:
        model.compile(optimizer=Nadam(lr=0.0001),
                      loss={'t01': 'categorical_crossentropy'},
                      loss_weights={'t01': 1.},
                      metrics=['accuracy'])

        # train the model
        history = model.fit_generator(generator=traingen,
                                      steps_per_epoch=train_step_size,
                                      validation_data=valgen,
                                      validation_steps=val_step_size,
                                      epochs=EPOCHS,
                                      callbacks=[checkpoint,
                                                 csv_logger,
                                                 lr_plateau,
                                                 stop],
                                      verbose=1)

    # necessary for recoverring the original model later, instead of
    # the parallelized model. this matters for transfer learning
    model.save(os.path.join(model_path, 'model.h5'))

    # XXX: the following graphs are only computed for the current
    #      training session. This is ok until we decide to continue
    #      training on a model, instead of starting fresh.

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(model_path, 'acc.png'))
    plt.close()

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(model_path, 'loss.png'))
    plt.close()

    return model


def main(argv):
    """The main function.

    The main function, which handles user input and common tasks when
    running galnet.

    Parameters
    ----------
    argv : :obj:`argparse.Namespace`
        Runtime options set from the terminal.

    Returns
    -------
    score : list of float
        The test set score as [loss, accuracy]

    """
    # safely create output directory for our model/statistics
    # we also create a unique id in the process, to ensure we can tell
    # specific configurations of models apart in other contexts by
    # referring to their uuids. model configuration is stored as part
    # of each model.h5 file (just load and investigate its properties)
    if argv.TRAIN:
        model_path = os.path.join(argv.DATA, argv.MODEL, str(uuid.uuid4()))
    elif argv.TEST:
        model_path = os.path.join(argv.DATA, argv.MODEL)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # import our dataset, which is a pandas dataframe containing path
    # information to the actual image data
    gz2 = pd.read_hdf(os.path.join(argv.DATA, 'gz2.h5'))
    gz2 = gz2[gz2['a01'] >= .3]  # adopt agreement threshold of DS18

    # choose the questions we want to classify. note that the number
    # of columns here should match the number of output layers in
    # model.py
    class_cols = ['t01a01', 't01a02', 't01a03']

    # split the data into a training set and a test set. the test set
    # will be set aside entirely until the very end
    train, test = train_test_split(gz2, test_size=.25)

    # if we only want to test, we need to do a lot less
    if argv.TEST:
        model = load_model(os.path.join(model_path, 'model.h5'))
        score = test(test, model, class_cols, model_path)

    elif argv.TRAIN:
        model = train_model(train, class_cols, model_path)
        score = test_model(test, model, class_cols, model_path)

    return score


if __name__ == '__main__':
    # set up an absolute path to the module to make life easier
    MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..'))

    # set up plotting defaults
    plt.rcParams.update({'font.size': 14,
                         'figure.figsize': (12, 12)})

    # set up command line options for use with hpc
    PARSER = argparse.ArgumentParser(description="Run a CNN.")

    # independent commands
    PARSER.add_argument('-d', '--data', dest='DATA', action='store',
                        default='data', help="Data folder.")
    PARSER.add_argument('-m', '--model', dest='MODEL', action='store',
                        default='model', help="Location (folder) inside data "
                        "to save model")

    # run mode. users must select either training or testing
    MODE = PARSER.add_mutually_exclusive_group()
    MODE.add_argument('--train', dest='TRAIN', action='store_true',
                      default=False)
    MODE.add_argument('--test', dest='TEST', action='store_true',
                      default=False)

    ARGS = PARSER.parse_args()
    ARGS.DATA = os.path.join(MODULE_PATH, ARGS.DATA)
    main(ARGS)
