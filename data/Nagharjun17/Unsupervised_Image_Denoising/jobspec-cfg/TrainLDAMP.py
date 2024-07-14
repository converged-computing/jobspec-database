import numpy as np
import argparse
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import LearnedDAMP as LDAMP
from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="/scratch/nm4074/imageprocessing/D-AMP_Toolbox/LDAMP_TensorFlow/TrainingData",
    help="Dataset path")
parser.add_argument(
    "--model_path",
    type=str,
    default="/scratch/nm4074/imageprocessing/D-AMP_Toolbox/LDAMP_TensorFlow/saved_models/LDAMP",
    help="Saved model path")
FLAGS, unparsed = parser.parse_known_args()

print(FLAGS)

## Network Parameters
height_img = 40
width_img = 40
channel_img = 1 # RGB -> 3, Grayscale -> 1
filter_height = 3
filter_width = 3
num_filters = 64
n_DnCNN_layers=16
max_n_DAMP_layers=10

## Training Parameters
max_Epoch_Fails=3#How many training epochs to run without improvement in the validation error
LayerbyLayer=True #Train only the last layer of the network
learning_rate = 0.00001
EPOCHS = 50
n_Train_Images=128*1600
n_Val_Images=10000
BATCH_SIZE = 128

## Problem Parameters
sampling_rate=.2
sigma_w=30./255.#Noise std
n=channel_img*height_img*width_img
m=int(np.round(sampling_rate*n))
measurement_mode='gaussian'

# Parameters to to initalize weights
init_mu = 0
init_sigma = 0.1

train_start_time=time.time()
for n_DAMP_layers in range(1,max_n_DAMP_layers+1,1):
    tf.reset_default_graph()

    LDAMP.SetNetworkParams(new_height_img=height_img, new_width_img=width_img, new_channel_img=channel_img, \
                           new_filter_height=filter_height, new_filter_width=filter_width, new_num_filters=num_filters, \
                           new_n_DnCNN_layers=n_DnCNN_layers, new_n_DAMP_layers=n_DAMP_layers,
                           new_sampling_rate=sampling_rate, \
                           new_BATCH_SIZE=BATCH_SIZE, new_sigma_w=sigma_w, new_n=n, new_m=m, new_training=True)
    LDAMP.ListNetworkParameters()

    training_tf = tf.placeholder(tf.bool, name='training')
    x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

    ## Initialize the variable theta which stores the weights and biases
    n_layers_trained = n_DAMP_layers
    theta = [None] * n_layers_trained
    for iter in range(n_layers_trained):
        with tf.compat.v1.variable_scope("Iter" + str(iter)):
            theta_thisIter = LDAMP.init_vars_DnCNN(init_mu, init_sigma)
        theta[iter] = theta_thisIter

    ## Construct the measurement model and handles/placeholders
    [A_handle, At_handle, A_val, A_val_tf] = LDAMP.GenerateMeasurementOperators(measurement_mode)
    y_measured = LDAMP.GenerateNoisyCSData_handles(x_true, A_handle, sigma_w, A_val_tf)

    ## Construct the reconstruction model
    (x_hat, MSE_history, NMSE_history, PSNR_history, r_final, rvar_final, div_overN, _) = LDAMP.LDAMP(y_measured,A_handle,At_handle,A_val_tf,theta,x_true,tie=False,training=training_tf,LayerbyLayer=LayerbyLayer, test=False)

    ## Define loss and determine which variables to train
    nfp = np.float32(height_img * width_img)
    cost = LDAMP.MCSURE_loss(x_hat, div_overN, r_final, tf.sqrt(rvar_final))

    iter = n_DAMP_layers - 1
    vars_to_train=[]#List of only the variables in the last layer.
    for l in range(0, n_DnCNN_layers):
        vars_to_train.extend([theta[iter][0][l]])
    for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, beta, and gamma
        gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
        beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
        var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
        mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
        gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
        beta = [v for v in tf.global_variables() if v.name == beta_name][0]
        moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
        moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
        vars_to_train.extend([gamma,beta,moving_variance,moving_mean])

    LDAMP.CountParameters()

    ## Load and Preprocess Training Data
    train_images = np.load(FLAGS.data_path + '/TrainingData_patch'+str(height_img)+'.npy')
    train_images=train_images[range(n_Train_Images),0,:,:]
    assert (len(train_images)>=n_Train_Images), "Requested too much training data"

    val_images = np.load(FLAGS.data_path + '/ValidationData_patch'+str(height_img)+'.npy')
    val_images=val_images[:,0,:,:]
    assert (len(val_images)>=n_Val_Images), "Requested too much validation data"

    x_train = np.transpose(np.reshape(train_images, (-1, channel_img * height_img * width_img)))
    x_val = np.transpose(np.reshape(val_images, (-1, channel_img * height_img * width_img)))

    ## Train the Model

    optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = optimizer0.minimize(cost, var_list=vars_to_train)

    saver_best = tf.train.Saver() 
    saver_dict={}
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        start_time = time.time()
        print("Load Initial Weights ...")
        ##Load previous values for the weights
        saver_initvars_name_chckpt = LDAMP.GenLDAMPFilename(path=FLAGS.model_path) + ".ckpt"
        for iter in range(n_layers_trained):#Create a dictionary with all the variables except those associated with the optimizer.
            for l in range(0, n_DnCNN_layers):
                saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/w": theta[iter][0][l]})
            for l in range(1, n_DnCNN_layers - 1):  # Associate variance, means, and beta
                gamma_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/gamma:0"
                beta_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/beta:0"
                var_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance:0"
                mean_name = "Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean:0"
                gamma = [v for v in tf.global_variables() if v.name == gamma_name][0]
                beta = [v for v in tf.global_variables() if v.name == beta_name][0]
                moving_variance = [v for v in tf.global_variables() if v.name == var_name][0]
                moving_mean = [v for v in tf.global_variables() if v.name == mean_name][0]
                saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/gamma": gamma})
                saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/beta": beta})
                saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_variance": moving_variance})
                saver_dict.update({"Iter" + str(iter) + "/l" + str(l) + "/BN/moving_mean": moving_mean})
            saver_initvars = tf.train.Saver(saver_dict)
            saver_initvars.restore(sess, saver_initvars_name_chckpt)
            print("Loaded weights from %s" % saver_initvars_name_chckpt)
        time_taken = time.time() - start_time

        print("Training ...")
        print()
        if __name__ == '__main__':
            print("**********************")
            save_name = LDAMP.GenLDAMPFilename(path=FLAGS.model_path)
            save_name_chckpt = save_name + ".ckpt"
            val_values = []
            print("Initial Weights Validation Value:")
            rand_inds = np.random.choice(len(val_images), n_Val_Images, replace=False)
            start_time = time.time()
            for offset in range(0, n_Val_Images - BATCH_SIZE + 1, BATCH_SIZE):  # Subtract batch size-1 to avoid eerrors when len(train_images) is not a multiple of the batch size
                end = offset + BATCH_SIZE

                # Generate a new measurement matrix
                A_val=LDAMP.GenerateMeasurementMatrix(measurement_mode)
                batch_x_val = x_val[:, rand_inds[offset:end]]

                # Run optimization. This will both generate compressive measurements and then recontruct from them.
                loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, A_val_tf:A_val, training_tf:False})
                val_values.append(loss_val)
            time_taken = time.time() - start_time
            print(np.mean(val_values))
            best_val_error = np.inf
            failed_epochs=0
            for i in range(EPOCHS):
                if failed_epochs>=max_Epoch_Fails:
                    break
                train_values = []
                print ("This Training iteration ...")
                rand_inds=np.random.choice(len(train_images), n_Train_Images,replace=False)
                start_time = time.time()
                for offset in range(0, n_Train_Images-BATCH_SIZE+1, BATCH_SIZE):
                    end = offset + BATCH_SIZE

                    # Generate a new measurement matrix
                    A_val = LDAMP.GenerateMeasurementMatrix(measurement_mode)
                    batch_x_train = x_train[:, rand_inds[offset:end]]

                    # Run optimization. This will both generate compressive measurements and then recontruct from them.
                    _, loss_val = sess.run([optimizer,cost], feed_dict={x_true: batch_x_train, A_val_tf:A_val, training_tf:True})
                    train_values.append(loss_val)
                time_taken = time.time() - start_time
                print(np.mean(train_values))
                val_values = []
                print("EPOCH ",i+1," Validation Value:" )
                rand_inds = np.random.choice(len(val_images), n_Val_Images, replace=False)
                start_time = time.time()
                for offset in range(0, n_Val_Images-BATCH_SIZE+1, BATCH_SIZE):
                    end = offset + BATCH_SIZE

                    # Generate a new measurement matrix
                    A_val = LDAMP.GenerateMeasurementMatrix(measurement_mode)
                    batch_x_val = x_val[:, rand_inds[offset:end]]

                    # Run optimization. This will both generate compressive measurements and then recontruct from them.
                    loss_val = sess.run(cost, feed_dict={x_true: batch_x_val, A_val_tf: A_val, training_tf:False})
                    val_values.append(loss_val)
                time_taken = time.time() - start_time
                print(np.mean(val_values))
                if(np.mean(val_values) < best_val_error):
                    failed_epochs=0
                    best_val_error = np.mean(val_values)
                    best_sess = sess
                    print("********************")
                    save_path = saver_best.save(best_sess, save_name_chckpt)
                    print("Best session model saved in file: %s" % save_path)
                else:
                    failed_epochs=failed_epochs+1
                print("********************")
                total_train_time = time.time() - train_start_time
                print("Training time so far: %.2f seconds" % total_train_time)
