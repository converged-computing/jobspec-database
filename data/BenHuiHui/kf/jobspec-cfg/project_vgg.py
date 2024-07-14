import os
import time
import csv
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from PIL import Image

slim = tf.contrib.slim

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

tfrecords_filename='invasive-train.tfrecords'
im_width=224
im_height=224

#Hyper Parameter to play with
batch_size=32
num_epochs=10

lr = 0.001
decay_rate=0.1
decay_per=40 #epoch

train_imgdir = 'transferred_train/'
test_imgdir = 'transferred_test/'
train_images = os.listdir(train_imgdir)
test_images = os.listdir(test_imgdir)
N_CLASSES = 132

# Read training labels
train_labels = []
with open('train.csv', 'r') as f:
	reader = csv.reader(f)
	headers = reader.next()
	for d in reader:
		train_labels.append(int(d[1]))

num_iter = len(train_labels)/batch_size

start_time = time.time()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

if os.path.exists(tfrecords_filename):
	print tfrecords_filename, "already exists"
else:
	writer = tf.python_io.TFRecordWriter(tfrecords_filename)
	print "Saving prepocessed file to '%s'" % tfrecords_filename
	for img_path in train_images:
		idx = int(img_path.split('.')[0])
		label = train_labels[idx]
		img = Image.open(os.path.join(train_imgdir, img_path))
		img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))

		example = tf.train.Example(features=tf.train.Features(feature={
					'image_raw': _bytes_feature(img.tostring()),
					'label': _int64_feature(label)
				}))
		writer.write(example.SerializeToString())
	writer.close()
	print("Preprocessing done in %s seconds" % (time.time() - start_time))

#Function to read the data from tfrecords
def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [im_height, im_width, 3])
    label = tf.cast(features['label'], tf.int32)
    images, labels = tf.train.shuffle_batch([image, label],
        batch_size=batch_size, capacity=256, num_threads=2, min_after_dequeue=32)
    return images, labels

def infer(inputs, is_training=True):
    inputs = tf.cast(inputs, tf.float32)
    inputs = ((inputs / 255.0)-0.5)*2
    #Use Pretrained Base Model
    with tf.variable_scope("vgg_16"):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
    #Append fully connected layer
    net = slim.flatten(net)
    net = slim.fully_connected(net, 512,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/fc1')
    net = slim.fully_connected(net, N_CLASSES,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            scope='finetune/fc2')
    return net

def losses(logits, labels):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss
        
def optimize(losses):
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(lr, global_step,
                                             num_iter*decay_per, decay_rate, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(losses, global_step=global_step)#,
                #var_list=slim.get_model_variables("finetune"))
    return train_op

tf.reset_default_graph()

#Create the training graph
filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=num_epochs)
image, label = read_and_decode(filename_queue)
prediction = infer(image)
loss = losses(prediction, label)
train_op = optimize(loss)

print "Training started"
with tf.Session() as sess:
    
    init_op = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())
    restore = slim.assign_from_checkpoint_fn(
               'vgg_16.ckpt',
               slim.get_model_variables("vgg_16"))
    sess.run(init_op)
    restore(sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for e in range(num_epochs):
        avg_loss, acc = 0, 0
        for i in range(num_iter):
            _, l = sess.run([train_op, loss])
            avg_loss += l/num_iter
        print "Epoch%03d avg_loss: %f" % (e+1, avg_loss)
    
    coord.request_stop()
    coord.join(threads)
    print 'Training Done'
    saver = tf.train.Saver(slim.get_model_variables())
    saver.save(sess, str(num_epochs) + 'model.ckpt')
    sess.close()

tf.reset_default_graph()

im_placeholder = tf.placeholder(tf.uint8, [None, im_height, im_width, 3])
logits = infer(im_placeholder, is_training=False)
prediction = tf.nn.softmax(logits)
predicted_labels = tf.argmax(prediction, 1)
test_labels = []
images = []

def saveImageList(images, labels, save_to_file):
     dic = OrderedDict()
     dic['image_name'] = images
     dic['category'] = np.array(labels, dtype=np.int32)
     data = pd.DataFrame(dic)
     data.to_csv(save_to_file, index=None, header=['image_name','category'])

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, str(num_epochs) + 'model.ckpt')
    
    for i, img_path in enumerate(test_images):
        print "\rProcessing %d/%d"%(i, len(test_images)),
        images.append(img_path)

        img = Image.open(os.path.join(test_imgdir, img_path))
        img = np.array(img.resize((im_width,im_height), Image.ANTIALIAS))
        
        prob = sess.run([predicted_labels], feed_dict={im_placeholder:np.expand_dims(img, axis=0)})
        print ('Prediction ', prob[0])
        test_labels.extend(prob[0])
            
    filename_output = str(num_epochs) + "_predictionVGG.csv"
    print "Writing result to", filename_output
    saveImageList(images, test_labels, filename_output)

    sess.close()
