
import sys
import csv
import time
import argparse
import itertools
import numpy as np
import tensorflow as tf
from concurrent.futures import *

sys.path.append("/.")
from baseline_model import *

'''
This file serves as a canonical example of how to perform a sequential
experiment using object-based TensorFlow models on Hokulea. This
example uses a single node, iterates over experimental parameters producing
outputs, and writes those outputs to a file.
'''


def generalization_experiment(exp_parameters):

    # Unpack the experimental parameters.
    (thread_count, batch_size, batch_interval, rep) = exp_parameters

    # Reset the default graph.
    tf.reset_default_graph()

    # Declare experimental measurement vars.
    steps = []
    val_losses = []
    train_losses = []
    mean_running_times = []

    # Instantiate a model.
    model = Model(FLAGS.input_size, FLAGS.label_size, FLAGS.learning_rate,
                  thread_count, FLAGS.val_enqueue_threads,
                  FLAGS.data_dir, FLAGS.train_file, FLAGS.validation_file)

    # Get input data.
    image_batch, label_batch = model.get_train_batch_ops(batch_size=batch_size)

    (val_image_batch, val_label_batch) = model.get_val_batch_ops(
        batch_size=FLAGS.val_batch_size)

    tf.summary.merge_all()

    # Instantiate a session and initialize it.
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=10.0)

    with sv.managed_session() as sess:

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir +
        #                                      '/train', sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        # Declare timekeeping vars.
        running_times = []
        optimize_step_running_time = 0

        # Print a line for debug.
        print('step | train_loss | train_error | val_loss | \
               val_error | t | total_time')

        # Load the validation set batch into memory.
        val_images, val_labels = sess.run([val_image_batch, val_label_batch])

        # Iterate until max steps.
        for i in range(FLAGS.max_steps):

            # Check for break.
            if sv.should_stop():
                break

            # If we have reached a testing interval, test.
            if i % FLAGS.test_interval == 1:

                # Update the batch, so as to not underestimate the train error.
                train_images, train_labels = sess.run([image_batch,
                                                       label_batch])

                # Make a dict to load the batch onto the placeholders.
                train_dict = {model.stimulus_placeholder: train_images,
                              model.target_placeholder: train_labels,
                              model.keep_prob: 1.0}

                # Compute error over the training set.
                train_error = sess.run(model.error, train_dict)

                # Compute loss over the training set.
                train_loss = sess.run(model.loss, train_dict)

                # Make a dict to load the val batch onto the placeholders.
                val_dict = {model.stimulus_placeholder: val_images,
                            model.target_placeholder: val_labels,
                            model.keep_prob: 1.0}

                # Compute error over the validation set.
                val_error = sess.run(model.error, val_dict)

                # Compute loss over the validation set.
                val_loss = sess.run(model.loss, val_dict)

                # Store the data we wish to manually report.
                steps.append(i)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                mean_running_times.append(np.mean(running_times))

                # Print relevant values.
                print('%d | %.6f | %.2f | %.6f | %.2f | %.6f | %.2f'
                      % (i,
                         train_loss,
                         train_error,
                         val_loss,
                         val_error,
                         np.mean(running_times),
                         np.sum(running_times)))

            # Hack the start time.
            start_time = time.time()

            # If it is a batch refresh interval, refresh the batch.
            if((i % batch_interval == 0) or (i == 0)):

                # Update the batch.
                train_images, train_labels = sess.run([image_batch,
                                                       label_batch])

            # Make a dict to load the batch onto the placeholders.
            train_dict = {model.stimulus_placeholder: train_images,
                          model.target_placeholder: train_labels,
                          model.keep_prob: FLAGS.keep_prob}

            # Run a single step of the model.
            sess.run(model.optimize, feed_dict=train_dict)

            # train_writer.add_summary(summary, i)

            # Update timekeeping variables.
            stop_time = time.time()
            optimize_step_running_time = stop_time - start_time
            running_times.append(optimize_step_running_time)

        # Close the summary writers.
        # test_writer.close()
        # train_writer.close()
        sv.stop()
        sess.close()

    return(steps, train_losses, val_losses, mean_running_times)


def main(_):

    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Create a list to store result vectors.
    experimental_outputs = []

    # Establish the dependent variables of the experiment.
    reps = range(5)
    thread_counts = [8, 16, 32]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    batch_intervals = [1, 2, 4, 8, 16, 32, 64, 128]

    # Produce the Cartesian set of configurations.
    experimental_configurations = itertools.product(thread_counts,
                                                    batch_sizes,
                                                    batch_intervals,
                                                    reps)

    # TODO: Create a distributed approach by parallizing over configs.

    # Iterate over each experimental config.
    for experimental_configuration in experimental_configurations:

        results = generalization_experiment(experimental_configuration)

        experimental_outputs.append([experimental_configuration, results])

    # Accomodate Python 3+
    # with open(FLAGS.log_dir + '/sa_generalization_out.csv', 'w') as csvfile:

    # Accomodate Python 2.7 on Hokulea.
    with open(FLAGS.log_dir +
              '/evaluate_model_stability.csv', 'wb') as csvfile:

        # Open a writer and write the header.
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['thread_count',
                            'batch_size',
                            'batch_interval',
                            'rep_num',
                            'step_num',
                            'train_loss',
                            'val_loss',
                            'mean_running_time'])

        # Iterate over each output.
        for (experimental_configuration, results) in experimental_outputs:

            # TODO: Generalize this pattern to not rely on var names.

            # Unpack the experimental configuration.
            (thread_count,
             batch_size,
             batch_interval,
             rep) = experimental_configuration

            # Unpack the cooresponding results.
            (steps, train_losses, val_losses, mean_running_times) = results

            # Iterate over the results vectors for each config.
            for (step, tl, vl, mrt) in zip(steps,
                                           train_losses,
                                           val_losses,
                                           mean_running_times):

                # Write the data to a csv.
                csvwriter.writerow([thread_count,
                                    batch_size,
                                    batch_interval,
                                    rep,
                                    step,
                                    tl,
                                    vl,
                                    mrt])


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.
    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Number of steps to run trainer.')

    parser.add_argument('--test_interval', type=int, default=100,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    parser.add_argument('--data_dir', type=str,
                        default='../data/mnist',
                        help='Directory from which to pull data.')

    parser.add_argument('--log_dir', type=str,
                        default='../log/evaluate_model_stability_large/',
                        help='Summaries log directory.')

    parser.add_argument('--val_batch_size', type=int,
                        default=10000,
                        help='Validation set batch size.')

    parser.add_argument('--keep_prob', type=float,
                        default=1.0,
                        help='Keep probability for output layer dropout.')

    parser.add_argument('--input_size', type=int,
                        default=28 * 28,
                        help='Dimensionality of the input space.')

    parser.add_argument('--label_size', type=int,
                        default=10,
                        help='Dimensinoality of the output space.')

    parser.add_argument('--train_file', type=str,
                        default='train.tfrecords',
                        help='Training dataset filename.')

    parser.add_argument('--validation_file', type=str,
                        default='validation.tfrecords',
                        help='Validation dataset filename.')

    parser.add_argument('--val_enqueue_threads', type=int,
                        default=32,
                        help='Number of threads to enqueue val examples.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # # Run the main function as TF app.
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
