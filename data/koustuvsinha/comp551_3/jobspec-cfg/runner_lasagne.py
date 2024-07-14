#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import pandas as pd
from scipy.ndimage.interpolation import zoom
from secrets import slack
from slacker import Slacker
import logging
import traceback
import scipy.io
import csv
import argparse
from lasagne.regularization import regularize_layer_params_weighted, l2

# Ignore if not using slack
slackClient = Slacker(slack)

def log_slack(msg):
    print(msg)
    try:
        slackClient.chat.post_message('#ml',msg,as_user='mimi')
    except Exception as e:
        #print('Slack connectivity issue')
        pass

def load_dataset(type='mnist'):
    import gzip

    def padwithzeros(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        npad = ((0,0), (0,0), (16,16),(16,16))
        data = np.pad(data,npad,mode='constant', constant_values=0) # 60 x 60
        #data = data.reshape(-1, 1, 60, 60)
        return data / np.float32(256)

    def load_data_images(filename):
        data = np.fromfile(filename,dtype='uint8')
        data = data.reshape(-1, 1, 60, 60)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def load_data_labels(filename):
        df = pd.read_csv(filename)
        data = np.array(df['Prediction'].values.astype(np.int32))
        return data

    def change(r):
        return list(r).index(1)

    if type == 'mnist':
        # We can now download and read the training and test set images and labels.
        X_train = load_mnist_images('train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test

    if type == 'data':
        X_train = load_data_images('train_x.bin')
        y_train = load_data_labels('train_y.csv')
        X_test = load_data_images('test_x.bin')
        #y_test = load_mnist_labels('')
        X_train, X_val = X_train[:-30000], X_train[-30000:]
        y_train, y_val = y_train[:-30000], y_train[-30000:]
        return X_train, y_train, X_val, y_val, X_test

    if type == 'dirty-mnist':
        mat = scipy.io.loadmat('mnist-with-awgn.mat')
        npad = ((0,0), (0,0), (16,16),(16,16))
        X_train = mat['train_x'].reshape(-1,1,28,28)
        X_test = mat['test_x'].reshape(-1,1,28,28)
        X_train = np.pad(X_train,npad,mode='constant', constant_values=0)
        X_test = np.pad(X_test,npad,mode='constant', constant_values=0)
        y_train = np.array(np.apply_along_axis(change,1,mat['train_y'])).astype(np.int32)
        y_test = np.array(np.apply_along_axis(change,1,mat['test_y'])).astype(np.int32)

        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn(input_var=None, param_values=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 60, 60),
                                        input_var=input_var)

    lconv1 = lasagne.layers.Conv2DLayer(
            l_in, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    lconv2 = lasagne.layers.Conv2DLayer(
            lconv1, num_filters=32, filter_size=(5, 5),pad=2,
            nonlinearity=lasagne.nonlinearities.rectify)
    #lpool2 = lasagne.layers.MaxPool2DLayer(lconv2, pool_size=(2, 2))

    lconv3 = lasagne.layers.Conv2DLayer(
            lconv2, num_filters=32, filter_size=(5, 5),pad=2,stride=2,
            nonlinearity=lasagne.nonlinearities.rectify)

    if param_values:
        lasagne.layers.set_all_param_values(lconv3,param_values)

    lconv4 = lasagne.layers.Conv2DLayer(
            lconv3, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    lconv5 = lasagne.layers.Conv2DLayer(
            lconv4, num_filters=32, filter_size=(5, 5),pad=2,
            nonlinearity=lasagne.nonlinearities.rectify)
    #lpool2 = lasagne.layers.MaxPool2DLayer(lconv2, pool_size=(2, 2))

    lconv6 = lasagne.layers.Conv2DLayer(
            lconv5, num_filters=32, filter_size=(5, 5),pad=2,stride=2,
            nonlinearity=lasagne.nonlinearities.rectify)

    #lconv4 = lasagne.layers.Conv2DLayer(
    #        lconv2, num_filters=32, filter_size=(5, 5),stride=2,
    #        nonlinearity=lasagne.nonlinearities.rectify)

    l_hidden1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(lconv6, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_hidden1, p=.5),
            num_units=19,
            nonlinearity=lasagne.nonlinearities.softmax)

    #return (network,l_in,lconv1,lpool1,lconv2,lpool2,l_hidden1)
    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def iterate_minibatches_2(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

def train(network,label='train',input_var=None,target_var=None,X_train=None,X_val=None,X_test=None,y_train=None,y_val=None,y_test=None,num_epochs=10,mini_batch=50,test_acc=True):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.
    #l2_penalty = regularize_layer_params_weighted(network, l2)
    #loss = loss + l2_penalty
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.002, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()# + l2_penalty
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    ## Idea : take this "updates" trained after MNIST and then re-use it on
    # our data

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        #log_slack("Within epoch {}".format(epoch + 1))
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, mini_batch, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        log_slack("Training data pass done")

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, mini_batch, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        log_slack("Validation data pass done")
        # Then we log_slack the results for this epoch:
        ttaken = time.time() - start_time
        cur_epoch = epoch + 1
        train_loss = train_err / train_batches
        val_loss = val_err / val_batches
        val_acc = val_acc / val_batches * 100
        log_slack("Epoch {} of {} took {:.3f}s".format(
            cur_epoch, num_epochs, ttaken))
        log_slack("  training loss:\t\t{:.6f}".format(train_loss))
        log_slack("  validation loss:\t\t{:.6f}".format(val_loss))
        log_slack("  validation accuracy:\t\t{:.2f} %".format(
            val_acc))

        fwriter.writerow([label,cur_epoch,train_loss,val_loss,val_acc,ttaken])
        if epoch % 50 == 0:
            # save weights
            np.savez('int_model_{}.npz'.format(epoch), *lasagne.layers.get_all_param_values(network))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0

    if test_acc:
        for batch in iterate_minibatches(X_test, y_test, mini_batch, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        log_slack("Final results:")
        log_slack("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        log_slack("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    return network,val_fn

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
# running 1 epoch for the poor cpu
def main(args=None, fwriter=None):
    num_epochs = args.epochs if args.epochs else 100
    num_mini = args.mini if args.mini else 50
    if not fwriter:
        fwriter = csv.reader(open('log.csv','wb'), delimiter=',')
    #fwriter = open('log.csv','a')
    #fwriter.write(['type','epoch','train_loss','val_loss','val_acc','time'])

    # Load the dataset
    log_slack("Loading data...")
    X_train, y_train, X_val, y_val, X_test = load_dataset(type='data')
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    log_slack("Building and compiling functions ...")
    network_d = build_cnn(input_var)

    if args.start:
        with np.load(args.start) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network_d, param_values)

    log_slack("Training start")
    network_d,val_fn = train(network_d,input_var=input_var,target_var=target_var,X_train=X_train,X_val=X_val,X_test=X_test,y_train=y_train,y_val=y_val,y_test=None,num_epochs=num_epochs,mini_batch=num_mini,test_acc=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.savez('final_model_' + timestr + '.npz', *lasagne.layers.get_all_param_values(network_d))
    test_prediction = lasagne.layers.get_output(network_d, deterministic=True)
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
    preds = np.concatenate([predict_fn(inputs) for inputs in iterate_minibatches_2(X_test, num_mini)])

    subm = np.empty((len(preds), 2))
    subm[:, 0] = np.arange(1, len(preds) + 1)
    subm[:, 1] = preds

    np.savetxt('submission' + timestr + '.csv', subm, fmt='%d', delimiter=',',header='ImageId,Label', comments='')
    log_slack('submission done')
    # Done

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs',type=int,help='Number of epochs',required=True)
    parser.add_argument('-m','--mini',type=int,help='Number of minibatches')
    parser.add_argument('-s','--start',type=str,help='Starting weights')

    args = parser.parse_args()
    try:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        with open('log_'+timestr+'.csv', 'wb') as csvfile:
            fwriter = csv.writer(csvfile, delimiter=',')
            main(args=args,fwriter=fwriter)
    except Exception as e:
        logging.error(traceback.format_exc())
        log_slack('Error')
        log_slack(e)
