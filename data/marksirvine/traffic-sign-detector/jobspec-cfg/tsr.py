from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
import batch_generator as bg

import cPickle as pickle

import scipy as scp

import numpy as np

data = pickle.load(open('dataset.pkl','rb'))

here = os.path.dirname(__file__)
sys.path.append(here)
sys.path.append(os.path.join(here, '..', 'CIFAR10'))


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('log-frequency', 1,
                            'Number of steps between logging results to the console and saving summaries. (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 5,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 1,
                            'Number of steps between model saves. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('max-steps', 50,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 0.01, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_float('momentum', 0.9, "The momentum value used in the update rule")
tf.app.flags.DEFINE_float('weight-decay', 0.0001, "The value of the weight decay used in the update rule")

#Image info
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
CLASS_COUNT = 43

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size,
                                                                       lr=FLAGS.learning_rate))
checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)


def deepnn(x_image, img_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), class_count=CLASS_COUNT):
    """deepnn builds the graph for a deep net for classifying CIFAR10 images.

    Args:
        x_image: an input tensor whose ``shape[1:] = img_space``
            (i.e. a batch of images conforming to the shape specified in ``img_shape``)
        img_shape: Input image shape: (width, height, depth)
        class_count: number of classes in dataset

    Returns: A tensor of shape (N_examples, 10), with values equal to the logits of
      classifying the object images into one of 10 classes
      (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
    """

    # First convolutional layer - maps one RGB image to 32 feature maps.
    #2
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name='conv1'
    )

    # norm1 = tf.nn.local_response_normalization(conv1,
    #                                           alpha=1e-4,
    #                                           beta=0.75,
    #                                           depth_radius=2,
    #                                           bias=2.0)

    # norm1 = tf.nn.l2_normalize(conv1, 2 ,epsilon=1e-12)

    #3
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=[2,2],
        padding="same",
        name='pool1'
    )
    #4
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name='conv2'
    )

    # norm2 = tf.nn.local_response_normalization(conv2,
    #                                           alpha=1e-4,
    #                                           beta=0.75,
    #                                           depth_radius=2,
    #                                           bias=2.0)
    # norm2 = tf.nn.l2_normalize(conv2, 2 ,epsilon=1e-12)

    #5
    pool2 = tf.layers.average_pooling2d(
        inputs=conv2,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool2'
    )
    #6
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name='conv3'
    )
    #7
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool3'
    )
    #8
    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 64], name='pool3_flattened')
    fc1 = tf.layers.dense(inputs=pool3_flat, units=64, name='fc1')
    #9
    logits = tf.layers.dense(inputs=fc1, units=class_count, name='fc2')

    return logits


def main(_):
    tf.reset_default_graph()

    # Build the graph for the deep net
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS])
        x_image = tf.reshape(x, [-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
        normalizedImages = tf.map_fn(lambda image: normalization(image), x_image)
        y_ = tf.placeholder(tf.float32, [None, CLASS_COUNT])


    with tf.name_scope('model'):
        y_conv = deepnn(x_image)


    trainVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainVariables
                       if 'bias' not in v.name]) * 0.0001

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)) #+ weights_norm
    cross_entropy = cross_entropy + lossL2;

    #accuracy and error
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    incorrect_prediction = tf.not_equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    error = tf.reduce_mean(tf.cast(incorrect_prediction, tf.float32), name='error')


    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow

    # We need to update the dependencies of the minimization op so that it all ops in the `UPDATE_OPS`
    # are added as a dependency, this ensures that we update the mean and variance of the batch normalisation
    # layers
    # See https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization for more

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        #train using momentum
        train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum).minimize(cross_entropy,global_step=global_step)


    #summaries
    training_loss_value = tf.placeholder(tf.float32)
    validation_loss_value = tf.placeholder(tf.float32)
    accuracy_value = tf.placeholder(tf.float32)
    error_value = tf.placeholder(tf.float32)

    overall_training_loss_summary = tf.summary.scalar("Overall Training Loss", training_loss_value)
    overall_validation_loss_summary = tf.summary.scalar("Ovearll Testing Loss", validation_loss_value)
    overall_accuracy_summary = tf.summary.scalar("Overall Accuracy", accuracy_value)
    overall_error_summary = tf.summary.scalar("Overall Error", error_value)

    overall_summary = tf.summary.merge([overall_training_loss_summary, overall_validation_loss_summary, overall_accuracy_summary, overall_error_summary])



    #create saver to save checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    #creating the session
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        overall_writer = tf.summary.FileWriter(run_log_dir + "_overall", sess.graph)

        sess.run(tf.global_variables_initializer())

        whitening()

        #variable to store the previous validation accuracy
        previous_validation_accuracy = 0.0

        #TRAINING AND VALIDATION
        for step in range(FLAGS.max_steps):

            training_loss_sum = 0
            training_images_count = 0

            #perform one training epoch
            for (trainImages, trainLabels) in bg.batch_generator(data,'train'):

                #train the CNN and get the training summary
                _ = sess.run(train_step, feed_dict={x_image: trainImages, y_: trainLabels})

                #add the loss to the overall sum
                training_loss_sum += sess.run(cross_entropy, feed_dict={x_image: trainImages, y_: trainLabels})
                training_images_count += len(trainImages)

            #calculate the average training loss
            average_training_loss = training_loss_sum / training_images_count



            # Validation: Monitoring accuracy using validation set
            if (step + 1) % FLAGS.log_frequency == 0:

                validation_accuracy_sum = 0
                validation_batch_count = 0

                validation_loss_sum = 0;
                validation_images_count = 0


                #calculate the average accuracy over all of the batches in the test data
                for (testImages, testLabels) in bg.batch_generator(data,'test'):

                    #sum up each batches validation accuracy
                    validation_accuracy_sum += sess.run(accuracy, feed_dict={x_image: testImages, y_: testLabels})
                    validation_batch_count += 1

                    #sum up each batch loss
                    validation_loss_sum += sess.run(cross_entropy, feed_dict={x_image: testImages, y_: testLabels})
                    validation_images_count += len(testImages)


                #calculate the average loss for each image
                average_validation_loss = validation_loss_sum / validation_images_count

                #calculate the average validation accuracy over all of the batches
                validation_accuracy = validation_accuracy_sum / validation_batch_count

                overall_summary_str = sess.run(overall_summary, feed_dict={training_loss_value: average_training_loss, validation_loss_value: average_validation_loss, accuracy_value: validation_accuracy, error_value: (1-validation_accuracy)})

                #print the accuracy
                print('step {}, accuracy on validation set : {}'.format(step, validation_accuracy))

                #add summaries for tensorboard
                overall_writer.add_summary(overall_summary_str, step)

                if validation_accuracy < previous_validation_accuracy:
                    print('Accuracy dropped. Reducing learning rate.')
                    FLAGS.learning_rate = FLAGS.learning_rate / 10

                print('Learning rate: {}'.format(FLAGS.learning_rate))
                previous_validation_accuracy = validation_accuracy


            # Save the model checkpoint periodically.
            if (step + 1) % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, checkpoint_path, global_step=step)

            if (step + 1) % FLAGS.flush_frequency == 0:
                overall_writer.flush()

        #TESTING

        # Resetting the internal batch indexes
        test_accuracy_sum = 0
        batch_count = 0

        #Test the CNN on all of the test images
        for (testImages, testLabels) in bg.batch_generator(data, 'test'):

            #get the accuracy for each test batch and add it to the sum of accuracies for all batches
            test_accuracy_sum += sess.run(accuracy, feed_dict={x_image: testImages, y_: testLabels})
            batch_count += 1


        #Display the overall test accuracy
        test_accuracy = test_accuracy_sum / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        print('model saved to ' + checkpoint_path)

        #close the writers
        overall_writer.close()


def createFilterImages(sess):
    print("Creating filter images")

    #printing the filters
    trainVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]

    filters = sess.run(trainVariables)

    for a in range(32):
        image = []
        for i in range(5):
            dim1 = []
            for j in range(5):
                dim2 = []
                for k in range(3):
                    dim2.append(filters[i][j][k][a])
                dim1.append(dim2)
            image.append(dim1)

        scp.misc.imsave("filters/first/" + str(a+1) + ".jpg", image)

    print("Finished creating filter images")


def normalization(image):

    #get each channel
    rChannel = image[:,:,0]
    gChannel = image[:,:,1]
    bChannel = image[:,:,2]

    #calculate the mean for each channel
    rMean = tf.reduce_mean(rChannel)
    gMean = tf.reduce_mean(gChannel)
    bMean = tf.reduce_mean(bChannel)

    rNormalized = rChannel - rMean
    gNormalized = gChannel - gMean
    bNormalized = bChannel - bMean

    rNormalized = tf.stack([rNormalized],2)
    gNormalized = tf.stack([gNormalized],2)
    bNormalized = tf.stack([bNormalized],2)

    new_image = tf.concat([rNormalized, gNormalized, bNormalized],2)

    return new_image

def whitening():

    rChannel = []
    gChannel = []
    bChannel = []

    for i in range(len(data[0])):
        for j in range(IMG_WIDTH):
            for k in range(IMG_HEIGHT):
                rChannel.append(data[0][i][0][j][k][0])
                gChannel.append(data[0][i][0][j][k][1])
                bChannel.append(data[0][i][0][j][k][2])

    print("Separated the colour channels")

    #calculate means
    rMean = np.mean(rChannel)
    print("Calculated r channel mean: {}".format(rMean))
    gMean = np.mean(gChannel)
    print("Calculated g channel mean: {}".format(gMean))
    bMean = np.mean(bChannel)
    print("Calculated b channel mean: {}".format(bMean))

    #calculate standard deviations
    rStd = np.std(rChannel)
    print("Calculated r channel standard deviation: {}".format(rStd))
    gStd = np.std(gChannel)
    print("Calculated g channel standard deviation: {}".format(gStd))
    bStd = np.std(bChannel)
    print("Calculated b channel standard deviation: {}".format(bStd))

    #perform the whitening
    for i in range(len(data[0])):
        for j in range(IMG_WIDTH):
            for k in range(IMG_HEIGHT):
                data[0][i][0][j][k][0] -= rMean
                data[0][i][0][j][k][1] -= gMean
                data[0][i][0][j][k][2] -= bMean
                if(rStd > 0):
                    data[0][i][0][j][k][0] = data[0][i][0][j][k][0] / rStd
                if(gStd > 0):
                    data[0][i][0][j][k][1] = data[0][i][0][j][k][1] / gStd
                if(bStd > 0):
                    data[0][i][0][j][k][2] = data[0][i][0][j][k][2] / bStd

    for i in range(len(data[1])):
        for j in range(IMG_WIDTH):
            for k in range(IMG_HEIGHT):
                data[1][i][0][j][k][0] -= rMean
                data[1][i][0][j][k][1] -= gMean
                data[1][i][0][j][k][2] -= bMean
                if(rStd > 0):
                    data[1][i][0][j][k][0] = data[1][i][0][j][k][0] / rStd
                if(gStd > 0):
                    data[1][i][0][j][k][1] = data[1][i][0][j][k][1] / gStd
                if(bStd > 0):
                    data[1][i][0][j][k][2] = data[1][i][0][j][k][2] / bStd

    print("Whitening complete")


def imageWhitening(image):
    #get each channel
    rChannel = image[:,:,0]
    gChannel = image[:,:,1]
    bChannel = image[:,:,2]


    #calculate the mean for each channel
    rMean = tf.reduce_mean(rChannel)
    gMean = tf.reduce_mean(gChannel)
    bMean = tf.reduce_mean(bChannel)

    one = tf.constant(1, tf.float32)

    #calculate the standard deviation for each channel
    rStd = tf.maximum(one, tf.contrib.keras.backend.std(rChannel))
    gStd = tf.maximum(one, tf.contrib.keras.backend.std(gChannel))
    bStd = tf.maximum(one, tf.contrib.keras.backend.std(bChannel))

    #whitening each channel
    rWhitened = (rChannel - rMean) / rStd
    gWhitened = (gChannel - gMean) / gStd
    bWhitened = (bChannel - bMean) / bStd

    rWhitened = tf.stack([rWhitened],2)
    gWhitened = tf.stack([gWhitened],2)
    bWhitened = tf.stack([bWhitened],2)

    new_image = tf.concat([rWhitened, gWhitened, bWhitened],2)
    return new_image


if __name__ == '__main__':
    tf.app.run(main=main)
