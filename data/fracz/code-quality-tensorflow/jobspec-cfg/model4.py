import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import csv
import math
from random import shuffle
import sys
import os
import argparse
import random
from datetime import datetime

random.seed(datetime.now())

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasetName", help="Dataset name.")
    parser.add_argument("-nh", "--numHidden", help="Num hidden", type=int, default=128)
    parser.add_argument("-s", "--steps", help="Training steps.", type=int, default=50000)
    parser.add_argument("-b", "--batchSize", help="Batch size.", type=int, default=64)
    parser.add_argument("-d", "--displayStep", help="Display step.", type=int, default=200)
    parser.add_argument("-t", "--tokensCount", help="Vocabulary size.", type=int, default=129)
    args = parser.parse_args()
    return args

args = parseArguments()

####################### PARAMS
vocabulary_size = args.tokensCount # liczba tokenow w kodzie
embedding_size = 100 # rozmiar wektora wejsciowego (arbitralny chyba?)
num_hidden = args.numHidden
num_classes = 2
training_steps = args.steps
batch_size = args.batchSize
display_step = args.displayStep
learning_rate = 1e-4
restore = False

################# DATA INPUT

datasetName = args.datasetName

print("""
*****************************************************
Dataset: {}
Num hidden: {}, Steps: {}, Display step: {}
*****************************************************

""".format(datasetName, num_hidden, training_steps, display_step))

savePath = './trained/model4/{}/{}-{}'.format(datasetName, training_steps, num_hidden)
if not os.path.exists(savePath):
    os.makedirs(savePath)

class RefactorDataset():
    def __init__(self):
        self.data_before = np.genfromtxt('./input/{}/input-before.csv'.format(datasetName), delimiter=',')
        self.data_after = np.genfromtxt('./input/{}/input-after.csv'.format(datasetName), delimiter=',')
        self.lengths_before = np.genfromtxt('./input/{}/lengths-before.csv'.format(datasetName), delimiter=',')
        self.lengths_after = np.genfromtxt('./input/{}/lengths-after.csv'.format(datasetName), delimiter=',')
        self.max_seqlen = len(self.data_before[0])
        self.test_len = math.floor(len(self.data_before) * .15)
        self.batch_id = 0
        self.test_data = self.next(self.test_len)
        print("Input size: " + str(self.max_seqlen) + 'x' + str(len(self.data_before)))

    def validation(self, batch_size):
        return self.test_data[0:batch_size]

    def test(self):
        return self.test_data

    def next(self, batch_size):
        # Return a batch of data. When dataset end is reached, start over.
        if self.batch_id == len(self.data_before):
            self.batch_id = self.test_len
        batch_before = np.array([]).reshape(0, self.max_seqlen)
        batch_after = np.array([]).reshape(0, self.max_seqlen)
        seqlen_before = np.array([])
        seqlen_after = np.array([])
        batch_labels = np.array([]).reshape(0, 2)
        for i in range(self.batch_id, min(self.batch_id + batch_size, len(self.data_before))):
            inverse = not random.getrandbits(1)
            if inverse:
                batch_before = np.vstack([batch_before, self.data_after[i]])
                batch_after = np.vstack([batch_after, self.data_before[i]])
                seqlen_before = np.append(seqlen_before, self.lengths_after[i])
                seqlen_after = np.append(seqlen_after, self.lengths_before[i])
                batch_labels = np.vstack([batch_labels, [0, 1]])
            else:
                batch_before = np.vstack([batch_before, self.data_before[i]])
                batch_after = np.vstack([batch_after, self.data_after[i]])
                seqlen_before = np.append(seqlen_before, self.lengths_before[i])
                seqlen_after = np.append(seqlen_after, self.lengths_after[i])
                batch_labels = np.vstack([batch_labels, [1, 0]])

        self.batch_id = min(self.batch_id + batch_size, len(self.data_before))
        return batch_before, batch_after, seqlen_before, seqlen_after, batch_labels

dataset = RefactorDataset()

#batch_before, batch_after, seqlen_before, seqlen_after, batch_labels = dataset.next(5)
#print("BEFORE:\n", batch_before, seqlen_before, "\nAFTER:\n", batch_after, seqlen_after, "\nLABELS:\n", batch_labels)


############################################ RNN

train_inputs_before = tf.placeholder(tf.int32, shape=[None, dataset.max_seqlen], name="train_inputs_before")
train_inputs_after = tf.placeholder(tf.int32, shape=[None, dataset.max_seqlen], name="train_inputs_after")
train_outputs = tf.placeholder(tf.float32, shape=[None, num_classes], name="train_outputs")
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
seqlen_before = tf.placeholder(tf.int32, [None], name="seqlen_before")
seqlen_after = tf.placeholder(tf.int32, [None], name="seqlen_after")

# https://www.tensorflow.org/tutorials/word2vec#building_the_graph
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed_before = tf.nn.embedding_lookup(embeddings, train_inputs_before)
embed_after = tf.nn.embedding_lookup(embeddings, train_inputs_after)

weights = tf.Variable(tf.random_normal([4*num_hidden, num_classes]))
biases = tf.Variable(tf.random_normal([num_classes]))

def BiRNN(x, seqlen):
    inputs = tf.unstack(x, num=dataset.max_seqlen, axis=1)
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32, sequence_length=seqlen)
    outputs = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])
    outputs = tf.reduce_max(outputs, axis=1)
    return outputs

with tf.variable_scope('before'):
    outputs_before = BiRNN(embed_before, seqlen_before)

with tf.variable_scope('after'):
    outputs_after = BiRNN(embed_after, seqlen_after)

outputs = tf.concat([outputs_before, outputs_after], 1)
logits = tf.matmul(outputs, weights) + biases

prediction = tf.nn.softmax(logits, name="prediction")
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_outputs))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(train_outputs, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

saver = tf.train.Saver()

config = tf.ConfigProto(
#    device_count = {'GPU': 2}
#    , log_device_placement=True
)



with tf.Session(config=config) as sess:
    if restore:
        new_saver = tf.train.import_meta_graph(savePath + '/model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(savePath))
        all_vars = tf.get_collection('vars')
        for v in all_vars:
            v_ = sess.run(v)
            print(v_)
    else:
        sess.run(init)

        test_before, test_after, test_seqlen_before, test_seqlen_after, test_labels = dataset.test()

        for step in range(1, training_steps+1):
            batch_before, batch_after, batch_seqlen_before, batch_seqlen_after, batch_labels = dataset.next(batch_size)

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={train_inputs_before: batch_before,
                                          train_inputs_after: batch_after,
                                          train_outputs: batch_labels,
                                          seqlen_before: batch_seqlen_before,
                                          seqlen_after: batch_seqlen_after})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={train_inputs_before: test_before,
                                                                      train_inputs_after: test_after,
                                                                      train_outputs: test_labels,
                                                                      seqlen_before: test_seqlen_before,
                                                                      seqlen_after: test_seqlen_after})
                print(str(step) + "\t" + "{:.4f}".format(loss).replace('.', ',') + "\t" + "{:.3f}".format(acc).replace('.', ','))
                sys.stdout.flush()

        print("Optimization Finished!")

        saver.save(sess, savePath + '/model')
