import re
from pathlib import Path
from collections import defaultdict
#import os.path
import json
import random
import tensorflow as tf
import numpy as np
import sys
import getopt

tf.debugging.set_log_device_placement(True)
opts = {k:v  for (k,v) in getopt.getopt(sys.argv[1:],"c:d:k:l:")[0]}
lang_wild_re = re.compile(r'^(?:.*/)?(?:[^/]{19})_+(\w{6})')
lang_code_re = re.compile(r'^(?:.*/)?(\w{3})-')
def get_lang_wild(fn):
	m=lang_code_re.match(fn)
	if m:
		return lang2wild[m.group(1)]
	return lang_wild_re.match(fn).group(1)

lang2wild = {
"kab": "KABCEB",
"ind": "INZTSI",
"sun": "SUNIBS",
"jav": "JAVNRF",
"eus": "EUSEAB",
"tam": "TCVWTC",
"kan": "ERVWTC",
"tel": "TCWWTC",
"hin": "HNDSKV",
"por": "PORARA",
"rus": "RUSS76",
"eng": "EN1NIV",
"mar": "MARWTC",
"tha": "THATSV",
"iba": "IBATIV",
"cnh": "CNHBSM"
}
train_dir="train"
val_dir = "valid/"
train_filenames = list(map(str,filter(lambda fn : str(fn)[-4:] == '.npy' and fn.is_file(),Path(train_dir).iterdir() )))
val_filenames = list(map(str,filter(lambda fn : str(fn)[-4:] == '.npy' and str(fn)[6:9] not in [
] and fn.is_file(),Path(val_dir).iterdir() )))
random.shuffle(train_filenames)
langs = {l:i for i,l in enumerate(set([
	wild for wild in map(get_lang_wild,train_filenames) if wild
]))}

#Model
len2backprop=256
batch_size = 64
dense_dim = 39
if '-d' in opts:
	dense_dim = int(opts['-d'])
cnn_dim,cnn_ker = 3,4
if '-k' in opts:
	cnn_ker = int(opts['-k'])
dense1 = tf.keras.layers.Dense(dense_dim, activation="relu", batch_size=batch_size)
cnn=[]
cnn_n=1
for li in range(cnn_n):
	cnn.append(tf.keras.layers.Conv1D(cnn_dim, activation="relu", kernel_size=cnn_ker, batch_size=batch_size))
lstm_size = 90
densee_size=len(langs)
lstme = tf.keras.layers.LSTM(lstm_size,batch_size=batch_size,return_state=True,activation="sigmoid")
lstm=[]
lstm_n = 1
if '-l' in opts:
	lstm_n = int(opts['-l'])
for li in range(lstm_n):
	lstm.append(tf.keras.layers.LSTM(lstm_size,batch_size=batch_size,return_state=True, return_sequences=True))
densee = tf.keras.layers.Dense(densee_size,activation=None)
layers = [dense1,lstme,densee]+cnn+lstm

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)

# Prepare the training dataset.

mean_outs = defaultdict(float)
def loss_fn(x, batches, mean_outs):
	cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	res = tf.constant(0.)
	for (b,true_w) in batches:
		class_loss = 0.
		cluster_loss = 0.
		v=x[b]
		mean_class_loss = 0.
		#mean_ground_loss = tf.math.reduce_mean(tf.math.square(v-mean_outs[true_w]))
		#try:
		mean_ground_loss = 2*cce([tf.one_hot(true_w,tf.shape(v)[0])], [v])
			#tf.math.reduce_mean(tf.math.square(v-tf.one_hot(true_w,tf.shape(v)[0])))
		#except:
		#	pass
		res += mean_ground_loss #### tf.math.maximum(mean_class_loss/2, tf.math.minimum(mean_class_loss, mean_ground_loss))
#	print("Loss FN size:", len(batches),batches)
	print('Loss',res)
	return res


def saveLayers(variant,epoch):
	name = variant+'.model.'+str(epoch)+'.json'
	of = open(name,'w')
	json.dump([list(map(lambda z:z.tolist(),layer.get_weights())) for layer in layers],
			of)

def doEpoch(filenames,training=True):
	samples = [None]*batch_size
	unbackpropagated_len = 0
	delta_queue = [(0000000,0)]*batch_size
	curr_fi = 0
	accurate_preds = 0
	loss = 0
	x = None

	lstm_h = []
	lstm_c = []
	lstme_c = tf.zeros([batch_size,lstm_size])
	lstme_h = tf.zeros([batch_size,lstm_size])
	for li in range(lstm_n):
		lstm_c.append(tf.zeros([batch_size,lstm_size]))
		lstm_h.append(tf.zeros([batch_size,lstm_size]))
	trainable_weights = None
	grads=None
	total_proc = 0
	start_i =0
	released_list = list(range(batch_size))

	#This loop generate variable-sized batches
	while True:
		while not delta_queue[0][0] and curr_fi < len(filenames):
			print("Oho"+"fi",curr_fi)
			released = released_list.pop()
			samples[released] = [0,np.swapaxes(np.load(filenames[curr_fi]),0,1),langs[get_lang_wild(filenames[curr_fi])]]
			new_sample_len = len(samples[released][1])
			shift = 0
			i = 0
			while i < batch_size-1 and shift+delta_queue[i+1][0] < new_sample_len:
				delta_queue[i] = delta_queue[i+1]
				shift += delta_queue[i][0]
				i += 1
			delta_queue[i]=(new_sample_len-shift,released)
			if i < batch_size - 1:
				delta_queue[i+1]=(delta_queue[i+1][0]-delta_queue[i][0],delta_queue[i+1][1])
			curr_fi += 1
		while start_i < batch_size and not delta_queue[start_i][0]:
			start_i += 1
		if start_i == batch_size:
			break

		blen = delta_queue[start_i][0]
		wid =  len(samples[delta_queue[start_i][1]][1][0])
		print("BLEN: ",blen)
		if blen:
			batch = tf.constant([samples[b][1][samples[b][0]:samples[b][0]+blen] if samples[b]!=None else [[0]*wid]*blen for b in range(batch_size)])
			b2proc=[]
			i=start_i
			while i<batch_size and (i==start_i or not delta_queue[i][0]):
				b2proc.append((delta_queue[i][1], samples[delta_queue[i][1]][2]))
				i += 1
				total_proc += 1
			for b in range(batch_size):
				if samples[b] != None:
					samples[b][0]=samples[b][0]+blen	
	
			#run batch
			with tf.GradientTape() as tape:
				x=dense1(batch)
				x=tf.reshape(x,[batch_size*blen,dense_dim,1])
				for li in range(cnn_n):
					x = cnn[li](x)
				x=tf.reshape(x,[batch_size,blen,cnn_dim*(dense_dim-cnn_ker*cnn_n+cnn_n)])
				xline=[x]
				for li in range(lstm_n):
					x, lstm_h[li], lstm_c[li] = lstm[li](x,[lstm_h[li],lstm_c[li]])
					xline.append(x)
				x, lstme_h, lstme_c = lstme(tf.concat(xline,2),[lstme_h,lstme_c])
				x = densee(x)
				loss = loss_fn(x,b2proc,mean_outs)


			if not trainable_weights:
				trainable_weights = []
				for l in layers:
					trainable_weights.extend(l.trainable_weights)
			new_grads = tape.gradient(loss, trainable_weights)
			grads = new_grads if grads == None else [x+y for x,y in zip(grads,new_grads)]
			for bd in b2proc:
				if tf.argmax(x[bd[0]]) == bd[1]:
					accurate_preds +=  1
				samples[bd[0]]=None	
			unbackpropagated_len += blen
			if training: # and (unbackpropagated_len >= len2backprop or curr_fi == len(filenames) - 1):
					unbackpropagated_len = 0
					print("Grads!")
					#print('Grads:',len(grads),len(trainable_weights),(grads))
					optimizer.apply_gradients(zip(grads, trainable_weights))
					grads = None

			
	
		for bd in b2proc:
			batch_state_mask = tf.expand_dims(tf.one_hot(bd[0],batch_size,0.,1.),1,len(langs))
			for li in range(lstm_n):
				lstm_h[li] *= batch_state_mask
				lstm_c[li] *= batch_state_mask
			lstme_h *= batch_state_mask
			lstme_c *= batch_state_mask



			# Use the gradient tape to automatically retrieve
        		# the gradients of the trainable variables with respect to the loss.
		released_list.extend([bd[0] for bd in b2proc])
		delta_queue[start_i]=[0,None]

	print ("Training" if training else "Valid.",total_proc,"Prec:" + str(1.*accurate_preds/total_proc))

	
epochs = 250
def variant_name():
	return  ''.join([k+str(v) for k,v in opts.items()])

for epoch in range(epochs):
	doEpoch(train_filenames,True)
	saveLayers(variant_name(),epoch)
	if epoch > 1:
		doEpoch(val_filenames,False)

