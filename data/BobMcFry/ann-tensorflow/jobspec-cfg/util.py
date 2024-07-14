import sys
import argparse
import importlib
import tensorflow as tf
import numpy as np
from ex04.svhn_helper import SVHN

SEED = 5
np.random.seed(SEED)
tf.set_random_seed(SEED)

weights_n = 0

def get_weights_and_bias(shape, shape_b=None, dtype=tf.float32,
        initializer_w=tf.random_normal_initializer(),
        initializer_b=tf.zeros_initializer()):
    if not shape_b:
        shape_b = shape[-1:]

    global weights_n

    weights_n += 1
    with tf.variable_scope('weights%d' % weights_n):
        return (
                tf.get_variable('W', initializer=initializer_w,
                                shape=shape, dtype=dtype),
                tf.get_variable('b', shape=shape_b, initializer=initializer_b)
                )


norm_n = 0


def batch_norm_layer(input):
    '''Create a layer that normalizes the batch with its mean and variance.'''
    global norm_n
    norm_n += 1
    with tf.variable_scope('norm%d' % norm_n):
        mean, var = tf.nn.moments(input, axes=[0, 1, 1])
        return tf.nn.batch_normalization(input, mean, var, 0, 1, 1e-10)

pool_n = 0


def max_pool_layer(input, ksize, strides):
    global pool_n
    pool_n += 1
    with tf.variable_scope('pool%d' % pool_n):
        return tf.nn.max_pool(input,
                ksize=ksize, strides=strides, padding='SAME')

conv_n = 0


def conv_layer(input, kshape, strides=(1, 1, 1, 1), activation=tf.nn.tanh,
        use_bias=True, padding='SAME'):
    '''Create a convolutional layer with activation function and variable
    initialisation.

    Parameters
    ----------
    input   :   tf.Variable
                Input to the layer
    kshape  :   tuple or list
                Shape of the kernel tensor
    strides :   tuple or list
                Strides
    activation  :   function
                    Activation functioin
    use_bias    :   bool
    padding :   str

    Returns
    -------
    tf.Variable
            The variable representing the layer activation (tanh(conv + bias))

    '''
    global conv_n
    conv_n += 1
    # this adds a prefix to all variable names
    with tf.variable_scope('conv%d' % conv_n):
        (fan_in, fan_out) = kshape[1:3]
        if activation == tf.nn.tanh:
            initializer = tf.random_normal_initializer(stddev=fan_in ** (-0.5))
        elif activation == tf.nn.relu:
            initializer = tf.random_normal_initializer(stddev=2 / fan_in)
        else:
            initializer = tf.random_normal_initializer()
        kernels = tf.Variable(initializer=initializer, name='kernels')
        if use_bias:
            bias_shape = (kshape[-1],)
            biases = tf.Variable(tf.constant(0.1), name='bias')
        conv = tf.nn.conv2d(
            input,
            kernels,
            strides,
            padding=padding,
            name='conv')
        if not activation:
            activation = tf.identity
        if use_bias:
            return activation(conv + biases, name='activation')
        else:
            return activation(conv, name='activation')


fc_n = 0


def fully_connected(input, n_out, with_activation=False, activation=tf.nn.tanh,
        use_bias=True):
    '''Create a fully connected layer with fixed activation function and variable
    initialisation. The activation function is ``tf.nn.tanh`` and variables are
    initialised from a truncated normal distribution with an stddev of 0.1

    Parameters
    ----------
    input   :   tf.Variable
                Input to the layer
    n_out   :   int
                Number of neurons in the layer
    with_activation :   bool
                        Return activation or drive (useful when planning to use
                        ``softmax_cross_entropy_with_logits`` which requires
                        unscaled logits)


    Returns
    -------
    tf.Variable
            The variable representing the layer activation
    '''
    global fc_n
    fc_n += 1
    with tf.variable_scope('fully%d' % fc_n):
        (fan_in, fan_out) = (input.shape[-1].value, n_out)
        if activation == tf.nn.tanh:
            init_W = tf.random_normal_initializer(stddev=fan_in ** (-0.5))
        elif activation == tf.nn.relu:
            init_W = tf.random_normal_initializer(stddev=2 / fan_in)
        else:
            init_W = tf.random_normal_initializer()
        init_b = tf.constant_initializer(0.1)
        W = tf.get_variable(
                'weights',
                initializer=init_W,
                shape=(input.shape[-1], n_out), # the last dim of the input
               dtype=tf.float32                 # is the 1st dim of the weights
            )
        if use_bias:
            bias = tf.get_variable('bias', initializer=init_b, shape=(n_out,))
        if use_bias:
            drive = tf.matmul(input, W) + bias
        else:
            drive = tf.matmul(input, W)
        if with_activation:
            return activation(drive)
        else:
            return drive



weighted_pool_n = 0


def weighted_pool_layer(input_layer, ksize, strides=(1, 1, 1, 1)):
    '''Helper function to do mixed max/avg pooling

    Parameters
    ----------
    input_layer :   tf.Tensor
                    4D tensor
    Returns
    -------
    tf.Tensor
           Tthe 4D tensor after being pooled
    '''
    global weighted_pool_n
    weighted_pool_n += 1
    with tf.variable_scope('weight_pool%d' % weighted_pool_n):
        a = tf.get_variable('a',
                initializer=tf.truncated_normal_initializer(),
                shape=(1,),
                dtype=tf.float32, trainable=True)
        max_pool = tf.nn.max_pool(input_layer, ksize, strides, padding='SAME')
        avg_pool = tf.nn.avg_pool(input_layer, ksize, strides, padding='SAME')
        pool = (a * max_pool + (1 - a) * avg_pool)
        return pool

inc_n = 0

def inception2d(x, in_channels, filter_count):
    '''Helper function to create inception module

    Parameters
    ----------
    in_channels :   int
                    number of input channels
    filter_count    :   int
                        number of filters to use for soemthing ?

    Returns
    -------
    tf.Tensor
           Tensor with filter_count*3 +1 output channels
    '''
    global inc_n
    inc_n += 1
    with tf.variable_scope('inception%d' % inc_n):
        # bias dimension = 3*filter_count and then the extra in_channels for the avg
        # pooling
        bias = tf.Variable(tf.truncated_normal([3*filter_count + in_channels]))

        # 1x1
        one_filter = tf.Variable(tf.truncated_normal([1, 1, in_channels,
            filter_count]))
        one_by_one = tf.nn.conv2d(x, one_filter,
                    strides=[1, 1, 1, 1], padding='SAME')

        # 3x3
        three_filter = tf.Variable(tf.truncated_normal([3, 3, in_channels,
            filter_count]))
        three_by_three = tf.nn.conv2d(x,
                    three_filter, strides=[1, 1, 1, 1], padding='SAME')

        # 5x5
        five_filter = tf.Variable(tf.truncated_normal([5, 5, in_channels,
            filter_count]))
        five_by_five = tf.nn.conv2d(x, five_filter,
                    strides=[1, 1, 1, 1], padding='SAME')

        # avg pooling
        pooling = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                padding='SAME')

        x = tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)
        # Concat in the 4th dim to stack
        x = tf.nn.bias_add(x, bias)
        return tf.nn.relu(x)


class ParameterTest(object):
    '''Test one set of parameters to the train() function.'''
    def __init__(self, model, batch_size, epochs,
            train_function, learning_rate, ignore_saved):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.accuracy = None
        self.train_function=train_function
        # sadly, we cannot always retrieve this from any optimizer
        self.learning_rate = learning_rate
        self.ignore_saved = ignore_saved

    def run(self):
        '''Run the training process with the specified settings.'''

        # self.save_fname = 'checkpoints/{name}_{batch}_{lr}_{epochs}_{opti}_{act}.ckpt'.format(
        #         name=self.model.__class__.__name__,
        #         batch=self.batch_size,
        #         lr=self.learning_rate,
        #         epochs=self.epochs,
        #         opti=self.model.opt.get_name(),
        #         act=self.model.act_fn.__name__
        # )
        self.save_fname = 'weights'     # use this for the competition
        self.accuracy = self.train_function(self.model, self.batch_size,
                self.epochs, self.save_fname, return_records=False,
                record_step=30, ignore_saved=self.ignore_saved)

    def __str__(self):
        return ('{opti:30}, learning rate={lr:5.4f}, batch size={bs:<5d}, '
                'epochs={epochs:<5d}, accuracy={acc:4.3f}'.format(
                    lr=self.learning_rate,
                    opti=self.model.opt.get_name(),
                    bs=self.batch_size,
                    epochs=self.epochs,
                    acc=self.accuracy
                )
        )

def get_optimizer(name):
    if isinstance(name, tf.train.Optimizer):
        return name
    else:
        return getattr(tf.train, name + 'Optimizer')

def main():
    tf_optimizers = {class_name[:-len('Optimizer')] for class_name in dir(tf.train) if 'Optimizer'
            in class_name and class_name != 'Optimizer'}
    parser = argparse.ArgumentParser(description='Test the net on one parameter set')
    parser.add_argument('-o', '--optimizer', required=True, type=str,
            choices=tf_optimizers, help='Optimization algorithm')
    parser.add_argument('-l', '--learning-rate', required=True, type=float,
            help='Learning rate for the optimizer')
    parser.add_argument('-b', '--batch-size', required=True, type=int,
            help='Batch size')
    parser.add_argument('-e', '--epochs', required=True, type=int,
            help='Number of epochs')
    parser.add_argument('-f', '--file', required=True, type=str,
            help='File to write result to')
    parser.add_argument('-m', '--model', required=True, type=str,
            help='Package path where Model class is located')
    parser.add_argument('-t', '--train', required=True, type=str,
            help='Module to search for train_model() function.')
    parser.add_argument('-i', '--ignore-saved', action='store_true',
            help='Ignore any saved weights.')

    args = parser.parse_args()
    model_cls = __import__(args.model, globals(), locals(), ['Model']).Model
    train_fn = __import__(args.train, globals(), locals(),
            ['train_model']).train_model

    optimizer_cls = get_optimizer(args.optimizer)
    optimizer = optimizer_cls(args.learning_rate)
    model = model_cls(optimizer, tf.nn.relu)

    pt = ParameterTest(model, args.batch_size, args.epochs,
            train_fn, args.learning_rate, args.ignore_saved)
    pt.run()
    print(pt)
    # the OS ensures sequential writes with concurrent processes
    with open(args.file, 'a') as f:
        f.write(str(pt) + '\n')
        f.flush()

    # from ex04.investigate_data import plot_mispredictions
    # svhn = SVHN()
    # plot_mispredictions(model, pt.save_fname, svhn._validation_data,
    #         svhn._validation_labels)

if __name__ == '__main__':
    main()
