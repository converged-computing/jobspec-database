import json
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array

from train_autoencoder import learn_representationSS, cluster_embeddings

tensorflow.random.set_seed(42)


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
        stratify=labels['class'].values, random_state=42
    )
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

accuracy = np.empty(30)
precision = np.empty(30)
recall = np.empty(30)
specificity = np.empty(30)
f1 = np.empty(30)

for seed in range(30):
    print(f'Running seed {seed}')
    #select labeled samples
    train_idx, _ = train_test_split(
        np.arange(idx.shape[0]), train_size=n_labeled_samples,
        stratify=y[idx], random_state=seed
    )
    y_train = y[train_idx]

    embeddings = learn_representationSS(x, train_idx, y_train, 30, verbose=False)
    results = cluster_embeddings(embeddings[hold_idx], y[hold_idx])
    accuracy[seed] = results[0]
    precision[seed] = results[1]
    recall[seed] = results[2]
    specificity[seed] = results[3]
    f1[seed] = results[4]

    print(f'Done seed {seed}')

np.save('results/T-test GCN AE/ae_acc_ttest.npy', accuracy)
np.save('results/T-test GCN AE/ae_prec_ttest.npy', precision)
np.save('results/T-test GCN AE/ae_rec_ttest.npy', recall)
np.save('results/T-test GCN AE/ae_spec_ttest.npy', specificity)
np.save('results/T-test GCN AE/ae_f1_ttest.npy', f1)