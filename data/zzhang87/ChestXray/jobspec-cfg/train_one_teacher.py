import keras 
import numpy as np
import pandas as pd
import cv2
import os
import json
import pdb
import argparse
import math
from shutil import rmtree

import keras.backend as K
from keras.callbacks import ProgbarLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.applications.inception_v3 import preprocess_input as inception_pre
from keras.applications.mobilenet import preprocess_input as mobilenet_pre
from keras.applications.resnet50 import preprocess_input as resnet_pre
from keras.applications.densenet import preprocess_input as densenet_pre
from datagenerator import ImageDataGenerator

from utils import load_filelist, create_model, calc_weights, bp_mll_loss

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--data_dir', help = 'Directory to the file lists and labels.')
	ap.add_argument('--image_dir', help = 'Directory to the raw images.')
	ap.add_argument('--partition_id', type = int, default = 1,
					help = 'Partition index (1-based).')
	ap.add_argument('--partition_num', type = int, default = 1,
					help = 'Number of partitions.')
	ap.add_argument('--train_dir', help = 'Directory the trained model and events will be saved to.')
	ap.add_argument('--model_name', default = 'inception',
					help = 'Network architecture to use. One of inception, resnet, densenet, or mobilenet.')
	ap.add_argument('--batch_size', type = int, default = 32)
	ap.add_argument('--num_epoch', type = int, default = 1)
	ap.add_argument('--optimizer', default = 'SGD',
					help = 'Optimizer to train the model. One of SGD, adam, or rmsprop.')
	ap.add_argument('--initial_lr', type = float, default = 1e-2, help = 'Initial learning rate.')
	ap.add_argument('--weight_loss', action = 'store_true')
	ap.add_argument('--bp_mll', action = 'store_true')

	args = ap.parse_args()

	assert(args.model_name in ['inception', 'resnet', 'mobilenet', 'densenet'])
	assert(args.optimizer in ['SGD', 'adam', 'rmsprop'])

	if os.path.isdir(args.train_dir):
		rmtree(args.train_dir)

	os.mkdir(args.train_dir)

	if args.model_name in ['inception']:
		image_size = 299

	else:
		image_size = 224

	with open(os.path.join(args.data_dir, 'label_map.json'), 'r') as f:
		label_map = json.load(f)

	with open(os.path.join(args.train_dir, 'label_map.json'), 'w') as f:
		json.dump(label_map, f)

	num_class = len(list(label_map.keys()))

	with open(os.path.join(args.data_dir, 'num_samples.json'), 'r') as f:
		num_samples = json.load(f)

	X_train, Y_train = load_filelist(args.data_dir, 'train', args.partition_id, args.partition_num)

	X_val, Y_val = load_filelist(args.data_dir, 'val', args.partition_id, args.partition_num)

	preprocess_input = {
		'inception': inception_pre,
		'resnet': resnet_pre,
		'mobilenet': mobilenet_pre,
		'densenet': densenet_pre
	}

	gen_train = ImageDataGenerator(rotation_range = 10,
								width_shift_range = 0.1,
								height_shift_range = 0.1,
								zoom_range = 0.1,
								fill_mode = 'constant',
								cval = 0,
								horizontal_flip = True,
								preprocessing_function = preprocess_input[args.model_name])

	gen_val = ImageDataGenerator(preprocessing_function = preprocess_input[args.model_name])

	# for x, y in gen_train.flow_from_list(x=X_train, y=Y_train, directory=args.image_dir,
	# 						batch_size = args.batch_size, target_size=(image_size,image_size)):
	# 	for i in range(x.shape[0]):
	# 		img = x[i].astype(np.uint8)
	# 		label = y[i]

	# 		cv2.imshow("show", img)
	# 		cv2.waitKey(600)
	# 		cv2.destroyAllWindows()
	# 		print(label)

	model_config = {'model_name': args.model_name,
					'optimizer': args.optimizer,
					'initial_lr': args.initial_lr}

	with open(os.path.join(args.train_dir, 'model_config.json'), 'w') as f:
		json.dump(model_config, f)

	loss = None
	if args.bp_mll:
		loss = bp_mll_loss

	model = create_model(model_config, image_size, label_map, loss)

	class_weight = None
	if args.weight_loss:
		class_weight = calc_weights(Y_train)

	tensorbard = TensorBoard(args.train_dir)
	reducelr = ReduceLROnPlateau(monitor = 'loss', factor = 0.9, patience = 5, mode = 'min')
	earlystop = EarlyStopping(monitor = 'val_mauc', min_delta = 1e-4,
								patience = max(5, args.num_epoch / 10), mode = 'max')
	ckpt = ModelCheckpoint(os.path.join(args.train_dir, 'weights.{epoch:03d}-{val_mauc:.2f}.hdf5'),
								monitor = 'val_mauc', save_best_only = True, mode = 'max')

	history = model.fit_generator(gen_train.flow_from_list(x=X_train, y=Y_train, directory=args.image_dir,
							batch_size = args.batch_size, target_size=(image_size,image_size)), epochs = args.num_epoch,
							steps_per_epoch = math.ceil(len(X_train) / float(args.batch_size)),
							validation_data = gen_val.flow_from_list(x=X_val, y=Y_val, directory=args.image_dir,
							batch_size = args.batch_size, target_size=(image_size,image_size)),
							validation_steps = math.ceil(len(X_val) / float(args.batch_size)),
							class_weight = class_weight, verbose = 2,
							callbacks = [tensorbard, reducelr, earlystop, ckpt])


if __name__ == "__main__":
	main()
