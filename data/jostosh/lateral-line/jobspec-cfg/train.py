import os

import numpy as np
import tensorflow as tf

from latline.common.data_util import DataBatcher, read_data
from latline.common.tf_utils import init_summary_writer
from latline.experiment_config import ExperimentConfig, init_log_dir
from latline.models import define_train_step, define_loss, create_model
from latline.util.config import _get_log_dir


def train(config):
    """
    Trains a convolutional neural network to locate multiple spheres in a simulated artificial lateral line experiment.
    Currently, this implementation is a port from the original Theano implementation, which is why it still misses some
    functionality that is mentioned in the current version of the paper.
    :param config:  Experiment config generated from command line input.
    """

    # Create TensorFlow session
    sess = tf.Session()

    # Read the data from storage
    test_x, test_y, train_x, train_y = read_data(config)
    test_x = test_x[:, :, :, -config.tau:]
    train_x = train_x[:, :, :, -config.tau:]

    # Set up two data batchers that provide a straightforward interface for moving through data batches
    data_batcher_train = DataBatcher(train_x, train_y, config.batch_size)
    data_batcher_test = DataBatcher(test_x, test_y, len(test_x))

    excitation0, excitation1, out = create_model(config, train_x, train_y)

    # Next, we define the loss that depends on the error + some L2 regularization of the weights
    # For reporting performance, the MSE is used
    loss, mse, target = define_loss(out, train_y, config.loss)

    # We define the train step which is the Op that should be called when training.
    train_step, global_step = define_train_step(loss, config.lr, config.optimizer)

    # Initialize writer for TensorBoard logging
    logdir = _get_log_dir(ExperimentConfig)
    print("Saving results to {}".format(logdir))
    train_summary, train_writer = init_summary_writer(logdir, sess, 'train')
    test_summary, test_writer = init_summary_writer(logdir, sess, 'test')

    # Set the number of batches inside a single train iteration
    n_batches = int(np.ceil(train_x.shape[0] / config.batch_size))

    # Initialize the graph variables
    sess.run(tf.global_variables_initializer())

    # Create a saver
    print("Creating saver")
    saver = tf.train.Saver()

    with open(os.path.join(logdir, 'results.csv'), 'w') as f:
        f.write('epoch,mse,loss\n')
        # Now we start training!
        for epoch in range(config.n_epochs):
            for batch_index in range(n_batches):
                batch_x, batch_y = data_batcher_train.next_batch()
                fdict = {
                    excitation0: batch_x[:, 0, :, :],
                    excitation1: batch_x[:, 1, :, :],
                    target: batch_y
                }
                if batch_index % 50 == 0:
                    mean_square_error, l, _, summ, step = sess.run(
                        fetches=[mse, loss, train_step, train_summary, global_step],
                        feed_dict=fdict
                    )
                    train_writer.add_summary(summ, global_step=step)
                else:
                    mean_square_error, l, _ = sess.run([mse, loss, train_step], feed_dict=fdict)

            test_x, test_y = data_batcher_test.next_batch()

            mean_square_error, summ, step, loss_num = sess.run(
                fetches=[mse, test_summary, global_step, loss],
                feed_dict={
                    excitation0: test_x[:, 0, :, :],
                    excitation1: test_x[:, 1, :, :],
                    target: test_y
                }
            )
            test_writer.add_summary(summ, global_step=step)
            print("Epoch ({}/{}), Test MSE: {}".format(epoch, config.n_epochs, mean_square_error))
            f.write(str(epoch) + ',' + str(mean_square_error) + ',' + str(loss_num) + '\n')
            f.flush()

            if (epoch + 1) % 50 == 0:
                save_path = saver.save(sess, os.path.join(logdir, 'model.ckpt'))
                print("Stored weights at:", save_path)


if __name__ == "__main__":
    ExperimentConfig.load()
    config = ExperimentConfig
    train(config)
