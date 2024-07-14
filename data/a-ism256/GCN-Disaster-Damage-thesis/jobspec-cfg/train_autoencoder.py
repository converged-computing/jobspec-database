import gc
import json
import math
import os.path as osp
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import tensorflow
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from utils import score_cm

tensorflow.random.set_seed(42)


def step_decay(epoch: int, lr: float):
	if epoch < 100:
		return 0.0005
	else:
		return 0.0001


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        clear_session()


def deepSSAEMulti(n_dim, n_hidden1, n_hidden2, n_classes):
	input_layer = Input(shape=(n_dim,))
	encoded = Dense(n_hidden1, activation='relu')(input_layer)
	encoded = Dense(n_hidden2, activation='relu', name="low_dim_features")(encoded)
	decoded = Dense(n_hidden1, activation='relu')(encoded)
	decoded = Dense(n_dim, activation='sigmoid')(decoded)
	
	classifier = Dense(n_classes, activation='softmax')(encoded)
	
	adamDino = optimizers.RMSprop(lr=0.0005)
	adamDino1 = optimizers.RMSprop(lr=0.0005)
	autoencoder = Model(inputs=[input_layer], outputs=[decoded])
	autoencoder.compile(optimizer=adamDino, loss=['mse'])
	
	ssautoencoder = Model(inputs=[input_layer], outputs=[decoded, classifier])
	ssautoencoder.compile(optimizer=adamDino1, loss=['mse','categorical_crossentropy'], loss_weights=[1., 1.])
	return [autoencoder, ssautoencoder]
	

def feature_extraction(model, data, layer_name):
	feat_extr = Model(inputs= model.input, outputs= model.get_layer(layer_name).output)
	return feat_extr.predict(data)


def learn_SingleReprSS(X_tot, train_idx, y_train, verbose=True):
	n_classes = len(np.unique(y_train))
	train_idx = train_idx.astype("int")
	X_train = X_tot[train_idx]
	encoded_Y_train = to_categorical(y_train, n_classes)
	n_col = X_tot.shape[1]
	n_units = round(n_col*1e-2)
		
	n_feat = math.ceil( n_units -1)
	n_feat_2 = math.ceil( n_units * 0.5)
	n_feat_4 = math.ceil( n_units * 0.25)
	
	n_hidden1 = randint(n_feat_2, n_feat)
	n_hidden2 = randint(n_feat_4, n_feat_2-1)
		
	ae, ssae = deepSSAEMulti(n_col, n_hidden1, n_hidden2, n_classes)
	lr_schedule = LearningRateScheduler(step_decay)
	clear_memory = ClearMemory()
	for epoch in range(200):
		if verbose:
			print(f'Epoch {epoch+1}')	
		ae.fit(X_tot, X_tot, epochs=1, batch_size=16, shuffle=True, verbose=0, callbacks=[lr_schedule, clear_memory])
		ssae.fit(X_train, [X_train, encoded_Y_train], epochs=1, batch_size=8, shuffle=True, verbose=0, callbacks=[lr_schedule, clear_memory])			
	new_train_feat = feature_extraction(ae, X_tot, "low_dim_features")
	return new_train_feat


def learn_representationSS(X_tot, train_idx, y_train, ens_size, verbose=True):
	intermediate_reprs = np.array([])
	for l in range(ens_size):
		if verbose:
			print(f'\nLearn representation {l+1}')
		embeddings = learn_SingleReprSS(X_tot, train_idx, y_train, verbose)
		if intermediate_reprs.size == 0:
			intermediate_reprs = embeddings
		else:
			intermediate_reprs = np.column_stack((intermediate_reprs, embeddings))
	return intermediate_reprs


def _make_cost_m(cm):
	s = np.max(cm)
	return (- cm + s)


def cluster_embeddings(embeddings, y_pred):
	clusters = KMeans(n_clusters=len(np.unique(y_pred)), random_state=42).fit_predict(embeddings)
	cm = confusion_matrix(y_pred, clusters)
	indexes = linear_sum_assignment(_make_cost_m(cm))
	cm2 = cm[:, indexes[1]]
	accuracy, precision, recall, specificity, f1 = score_cm(cm2)
	return accuracy, precision, recall, specificity, f1


if __name__ == "__main__":

	with open('exp_settings.json', 'r') as JSON:
		settings_dict = json.load(JSON)

	disaster = settings_dict['data_ss']['disaster']
	path = '/home/ami31/scratch/datasets/xbd/tier3_bldgs/'

	labels = pd.read_csv(list(Path(path + disaster).glob('*.csv*'))[0], index_col=0)
	labels.drop(columns=['xcoord','ycoord', 'long', 'lat'], inplace=True)
	labels.drop(index=labels.loc[labels['class']=='un-classified'].index, inplace=True)

	zone_func = lambda row: '_'.join(row.name.split('_', 2)[:2])
	labels['zone'] = labels.apply(zone_func, axis=1)
	for zone in labels['zone'].unique():
		if (labels[labels['zone'] == zone]['class'] == 'no-damage').all():
			labels.drop(index=labels.loc[labels['zone']==zone].index, inplace=True)

	if settings_dict['merge_classes']:
		label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':2}
	else:
		label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}

	labels['class'] = labels['class'].apply(lambda x: label_dict[x])

	#sample dataset so it fits in memory
	if labels.shape[0] > settings_dict['data_ss']['reduced_size']:
		idx, _ = train_test_split(
			np.arange(labels.shape[0]), train_size=settings_dict['data_ss']['reduced_size'],
			stratify=labels['class'].values, random_state=42)
		labels = labels.iloc[idx,:]

	x = []
	y = []

	for post_image_file in labels.index.values.tolist():  
		y.append(labels.loc[post_image_file,'class'])
		pre_image = Image.open(osp.join(path, disaster, post_image_file.replace('post', 'pre')))
		post_image = Image.open(osp.join(path, disaster, post_image_file))
		pre_image = pre_image.resize((128, 128))
		post_image = post_image.resize((128, 128))
		pre_image = img_to_array(pre_image)
		post_image = img_to_array(post_image)
		images = np.concatenate((pre_image, post_image))
		x.append(images.flatten())

	x = np.stack(x)
	x = MinMaxScaler().fit_transform(x)

	y = np.array(y)
	#extract hold set
	idx, hold_idx = train_test_split(
		np.arange(y.shape[0]), test_size=0.5,
		stratify=y, random_state=42
	)
	n_labeled_samples = round(settings_dict['data_ss']['labeled_size'] * y.shape[0])
	#select labeled samples
	train_idx, test_idx = train_test_split(
		np.arange(idx.shape[0]), train_size=n_labeled_samples,
		stratify=y[idx], random_state=42
	)
	print(f'Number of labeled samples: {train_idx.shape[0]}')
	print(f'Number of test samples: {test_idx.shape[0]}')
	print(f'Number of hold samples: {hold_idx.shape[0]}')
	y_train = y[train_idx]

	embeddings = learn_representationSS(x, train_idx, y_train, 30)

	train_acc, train_prec, train_rec, train_spec, train_f1 = cluster_embeddings(embeddings[train_idx], y_train)
	test_acc, test_prec, test_rec, test_spec, test_f1 = cluster_embeddings(embeddings[test_idx], y[test_idx])
	hold_acc, hold_prec, hold_rec, hold_spec, hold_f1 = cluster_embeddings(embeddings[hold_idx], y[hold_idx])
	full_acc, full_prec, full_rec, full_spec, full_f1 = cluster_embeddings(embeddings, y)

	print(f'\nTrain accuracy: {train_acc:.4f}')
	print(f'Train precision: {train_prec:.4f}')
	print(f'Train recall: {train_rec:.4f}')
	print(f'Train specificity: {train_spec:.4f}')
	print(f'Train f1: {train_f1:.4f}')
	print(f'\nTest accuracy: {test_acc:.4f}')
	print(f'Test precision: {test_prec:.4f}')
	print(f'Test recall: {test_rec:.4f}')
	print(f'Test specificity: {test_spec:.4f}')
	print(f'Test f1: {test_f1:.4f}')
	print(f'\nHold accuracy: {hold_acc:.4f}')
	print(f'Hold precision: {hold_prec:.4f}')
	print(f'Hold recall: {hold_rec:.4f}')
	print(f'Hold specificity: {hold_spec:.4f}')
	print(f'Hold f1: {hold_f1:.4f}')
	print(f'\nFull accuracy: {full_acc:.4f}')
	print(f'Full precision: {full_prec:.4f}')
	print(f'Full recall: {full_rec:.4f}')
	print(f'Full specificity: {full_spec:.4f}')
	print(f'Full f1: {full_f1:.4f}')

	print(f"\nLabeled size: {settings_dict['data_ss']['labeled_size']}")
	print(f"Reduced dataset size: {settings_dict['data_ss']['reduced_size']}")