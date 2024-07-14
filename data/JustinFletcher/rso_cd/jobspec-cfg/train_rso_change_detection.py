
import os

import sys
import time
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from dataset_generator import DatasetGenerator

from utils.classification_analysis import MulticlassClassificationAnalysis

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Input


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.losses import CategoricalCrossentropy


from callbacks.TensorboardValidationCallback import TensorboardValidationCallback
from callbacks.CustomMetrics import CustomMetrics



class SampleModel(keras.Model):

    def __init__(self, num_classes=10):

        super(SampleModel, self).__init__(name='my_model')
        self.num_classes = num_classes

        # Define your layers here.
        self.dense_1 = keras.layers.Dense(128, activation='relu')
        self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        """
        Define your forward pass here, using layers you previously defined in
        `__init__`).
        """

        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want
        # to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


def generate_toy_dataset(num_samples=1):

    # Make toy data.
    data = np.random.random((num_samples, 32))
    labels = np.random.random((num_samples, 10))

    # Instantiates a toy dataset instance.
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    return(dataset)


def generate_toy_image(num_samples=1):

    # Make toy data.
    data = np.random.random((num_samples, 32))
    labels = np.random.random((num_samples, 10))

    # Instantiates a toy dataset instance.
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    return(dataset)


def cast_image_to_float(image, classes):

    return tf.cast(image, tf.float32), classes

# TODO: Add metrics to measure confusion matrix



def main(_):

    # Set the GPUs we want the script to use/see
    print("GPU List = " + str(FLAGS.gpu_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_list


    train_tfrecord_list = [FLAGS.dataset_path + "rso_change_detection_0.tfrecords",
                           FLAGS.dataset_path + "rso_change_detection_1.tfrecords",
                           FLAGS.dataset_path + "rso_change_detection_2.tfrecords",
                           FLAGS.dataset_path + "rso_change_detection_3.tfrecords"]

    valid_tfrecord_list = [FLAGS.dataset_path + "rso_change_detection_4.tfrecords"]

    train_generator = DatasetGenerator(train_tfrecord_list,
                                       num_images=FLAGS.num_train_images,
                                       num_channels=1,
                                       augment=FLAGS.augment_training,
                                       shuffle=FLAGS.shuffle_training,
                                       batch_size=FLAGS.batch_size,
                                       num_threads=FLAGS.num_dataset_threads,
                                       buffer_size=FLAGS.dataset_buffer_size,
                                       encoding_function=cast_image_to_float,
                                       cache_dataset_memory=FLAGS.cache_in_memory,
                                       cache_dataset_file=FLAGS.cache_in_file,
                                       cache_name="train_" + FLAGS.cache_name)

    valid_generator = DatasetGenerator(valid_tfrecord_list,
                                       num_images=FLAGS.num_valid_images,
                                       num_channels=1,
                                       augment=False,
                                       shuffle=False,
                                       batch_size=FLAGS.batch_size,
                                       num_threads=FLAGS.num_dataset_threads,
                                       buffer_size=FLAGS.dataset_buffer_size,
                                       encoding_function=cast_image_to_float,
                                       cache_dataset_memory=FLAGS.cache_in_memory,
                                       cache_dataset_file=FLAGS.cache_in_file,
                                       cache_name="valid_" + FLAGS.cache_name)

    # test_generator = DatasetGenerator(test_tfrecord_name,
    #                                   num_images=FLAGS.num_test_images,
    #                                   num_channels=1,
    #                                   augment=False,
    #                                   shuffle=False,
    #                                   batch_size=FLAGS.infer_batch_size,
    #                                   num_threads=FLAGS.num_dataset_threads,
    #                                   buffer=FLAGS.dataset_buffer_size,
    #                                   encoding_function=cast_image_to_float,
    #                                   return_filename=True,
    #                                   cache_dataset_memory=False,
    #                                   cache_dataset_file=False,
    #                                   cache_name="")

    if FLAGS.train_with_keras_fit:

        # Instantiates the subclassed model.
        # model = SampleModel(num_classes=10)
        input_shape = (32, 19, 1)
        # This returns a tensor
        inputs = Input(shape=input_shape)

        x = Conv2D(8, kernel_size=2, activation=None)(inputs)

        def keras_resnet_block(input_fmap, num_filters=8):

            # a layer instance is callable on a tensor, and returns a tensor
            x = Conv2D(num_filters,
                       kernel_size=2,
                       padding='same')(input_fmap)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(num_filters,
                       kernel_size=2,
                       padding='same')(x)
            x = BatchNormalization()(x)

            x = Add()([input_fmap, x])

            # output_fmap = ReLU(x)
            output_fmap = x

            return(output_fmap)

        num_resnet_blocks = 16

        for n in range(num_resnet_blocks):

            x = keras_resnet_block(x)

        x = Flatten()(x)

        x = Dense(32, activation='relu')(x)
        predictions = Dense(4, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=predictions)

        # model = ResNet50(include_top=True,
        #                  weights=None,
        #                  input_shape=image_input_shape,
        #                  classes=4)

        # Make our callbacks/metrics
        callbacks = list()

        # checkpoint_path = "./checkpoints/weights_" + FLAGS.model_name + "_" + FLAGS.run_name + ".h5"
        # checkpoints_best = ModelCheckpoint(filepath=checkpoint_path,
        #                                    monitor='val_max_f1',
        #                                    verbose=1,
        #                                    save_best_only=True,
        #                                    save_weights_only=True,
        #                                    mode='max',
        #                                    period=1)
        # callbacks.append(checkpoints_best)

        # Use TensorBoard values as metrics.
        metrics = list()
        custom_metrics = CustomMetrics()
        metrics.append(custom_metrics.max_f1)

        # Create and append a default tensorbaord callback.
        tensorboard = TensorBoard(log_dir=FLAGS.log_path + FLAGS.run_name,
                                  histogram_freq=0,
                                  write_graph=False)
        callbacks.append(tensorboard)

        analyzer = MulticlassClassificationAnalysis

        # Add a custom tensorboard validation callback.
        tensorboard_valid = TensorboardValidationCallback(model,
                                                          train_generator,
                                                          valid_generator,
                                                          analyzer,
                                                          tensorboard,
                                                          custom_metrics,
                                                          epoch_frequency=FLAGS.validation_callback_frequency,
                                                          class_count=4,
                                                          num_plot_images=5)
        callbacks.append(tensorboard_valid)

        # The compile step specifies the training configuration.
        # sample_model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
        #                      loss='sparse_categorical_crossentropy',
        #                      metrics=['accuracy'])

        # sample_weight = [0.01, 0.33, 0.33, 0.33]
        # loss = CategoricalCrossentropy(sample_weight=sample_weight)
        optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_generator.dataset,
                  steps_per_epoch=len(train_generator),
                  epochs=FLAGS.num_training_epochs,
                  verbose=1,
                  validation_data=valid_generator.dataset,
                  validation_steps=len(valid_generator),
                  callbacks=callbacks)

    if FLAGS.train_with_estimator:

        # Instantiates the subclassed model.
        sample_model = SampleModel(num_classes=10)

        # The compile step specifies the training configuration.
        sample_model.compile(optimizer=tf.train.RMSPropOptimizer(FLAGS.learning_rate),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        # Create an Estimator from the compiled Keras model. Note the initial
        # model state of the keras model is preserved in the created Estimator.
        sample_est = tf.keras.estimator.model_to_estimator(
            keras_model=sample_model)

        sample_est.train(input_fn=generate_toy_dataset, steps=2000)


if __name__ == '__main__':

    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    # parser.add_argument('--train_tfrecord', type=str,
    #                     default=os.path.join(dataset_path,
    #                                          "tiny_train_rn_16mc.tfrecords"),
    #                     help='Path to the training TFRecord file.')

    # parser.add_argument('--valid_tfrecord', type=str,
    #                     default=os.path.join(dataset_path,
    #                                          "tiny_valid_rn_16mc.tfrecords"),
    #                     help='Path to the validation TFRecord file.')

    # parser.add_argument('--test_tfrecord', type=str,
    #                     default=os.path.join(dataset_path,
    #                                          "tiny_test_rn_16mc.tfrecords"),
    #                     help='Path to the testing TFRecord file.')


    parser.add_argument('--dataset_path', type=str,
                        default="C:\\research\\rso_change_detection\\data\\tfrecords\\",
                        help='Path to the training TFRecord file.')

    parser.add_argument('--log_path', type=str,
                        default="C:\\research\\log\\",
                        help='Path to the training TFRecord file.')

    parser.add_argument('--num_train_images', type=int,
                        default=(4096 * 4),
                        help='Number of images in the training set.')

    parser.add_argument('--num_valid_images', type=int,
                        default=4096,
                        help='Number of images in the validation set.')

    parser.add_argument('--num_test_images', type=int,
                        default=36,
                        help='Number of images in the testing set.')

    tb_file = '{}_{}/'.format("rso_cd", time.strftime('%Y%m%d%H%M'))

    parser.add_argument('--run_name', type=str,
                        default=tb_file,
                        help='The name of this run.')

    parser.add_argument('--base_model_name', type=str,
                        default="DarkNet",
                        help='The name of the base network/feature extractor to be used by SSD.')

    parser.add_argument('--learning_rate', type=float,
                        default=1e-4,
                        help='Initial learning rate.')

    parser.add_argument('--num_training_epochs', type=int,
                        default=64,
                        help='Number of epochs to train model.')

    parser.add_argument('--validation_callback_frequency', type=int,
                        default=16,
                        help='Number of epochs between validation callbacks.')

    parser.add_argument('--dataset_buffer_size', type=int,
                        default=128,
                        help='Number of images to prefetch in input pipeline.')

    parser.add_argument('--num_dataset_threads', type=int,
                        default=8,
                        help='Number of threads used by the input pipeline.')

    parser.add_argument('--batch_size', type=int,
                        default=2048,
                        help='Batch size to use in training and validation.')

    parser.add_argument('--infer_batch_size', type=int,
                        default=1,
                        help='Batch size to use in testing/inference.')

    parser.add_argument('--use_tensorboard', action='store_true',
                        default=False,
                        help='Should tensorboard events be created?')

    parser.add_argument('--cache_in_memory', action='store_true',
                        default=False,
                        help='Should we cache the dataset in memory?')

    parser.add_argument('--cache_in_file', action='store_true',
                        default=False,
                        help='Should we cache the dataset to file?')

    parser.add_argument('--cache_name', type=str,
                        default="n/a",
                        help='Name to use as part of the cache file.')

    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--augment_training', action='store_true',
                        default=False,
                        help='Should we augment the training set?')

    parser.add_argument('--shuffle_training', action='store_true',
                        default=False,
                        help='Should we shuffle the training set?')

    parser.add_argument('--num_classes', type=int,
                        default=2,
                        help='Number of classes (needed for ResNet only)')

    parser.add_argument("--train_with_estimator",
                        type=bool,
                        default=False,
                        help="")

    parser.add_argument("--train_with_keras_fit",
                        type=bool,
                        default=True,
                        help="")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
