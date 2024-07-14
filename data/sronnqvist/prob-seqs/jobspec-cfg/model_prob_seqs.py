import gzip
import json
import pickle
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
#from keras.models import Model
#from keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

MAX_LEN = 10
THRESHOLD = 0.4

def load_prob_seqs(pred_filename, data_filename):
    data, labels = [[]], [[]]
    pred_file = open(pred_filename)
    data_file = gzip.open(data_filename)
    _ = data_file.readline()
    last_doc = None
    for i, (pred_line, data_line) in enumerate(zip(pred_file, data_file)):
        probs = json.loads('['+pred_line.replace('\"','')+']')[0]
        datum = json.loads(data_line)
        labs = datum[1]
        if len(datum) > 2:
            doc = datum[2]
        else:
            doc = i
        if doc != last_doc:
            if last_doc is not None:
                # New doc
                data.append([])
                labels.append([])
            last_doc = doc
        data[-1].append(probs)
        labels[-1] = labs
    return data, labels


data, labels = [], []
for fold in range(1,11):
    print("Loading fold", fold)
    d, l = load_prob_seqs("../output/pred_fulldoc-fold-%d.txt" % fold, "../data/train-10-fold-prep/fulldoc-fold-%d-test-processed.jsonl.gz" % fold)
    #d, l = load_prob_seqs("../output/pred_fold-%d.txt" % fold, "../data/train-10-fold-prep/fold-%d-test-processed.jsonl.gz" % fold)
    data += d
    labels += l

print("Loading dev/test")
#dev_data, dev_labels = load_prob_seqs("../output/pred_fulldoc_dev.txt", "../data/fulldoc_dev-processed.jsonl.gz")
dev_data, dev_labels = load_prob_seqs("../output/pred_fulldoc-train_beg-dev.txt", "../data/dev-processed.jsonl.gz")
#test_data, test_labels = load_prob_seqs("../output/pred_fulldoc_test.txt", "../data/fulldoc_test-processed.jsonl.gz")


# Save data for possible future use
print("Saving...")
#json.dump({'probs': data, 'labels': labels}, gzip.open("probs.json.gz", 'wt', encoding="ascii"))
#pickle.dump({'probs': data, 'labels': labels}, open("fulldoc_probs.pickle", 'wb'))
#pickle.dump({'probs': dev_data, 'labels': dev_labels}, open("fulldoc-train_beg-dev_probs.pickle", 'wb'))
#pickle.dump({'probs': test_data, 'labels': test_labels}, open("fulldoc_test_probs.pickle", 'wb'))

"""
with open('probs.pickle', 'rb') as f:
    pickled = pickle.load(f)
data = pickled['probs']
labels = pickled['labels']
"""

print("Preparing...")
if MAX_LEN is not None:
    data = [d[:MAX_LEN] for d in data]
    dev_data = [d[:MAX_LEN] for d in dev_data]

seq_len = max([len(d) for d in data])
num_classes = len(data[0][0])

print('seq_len', seq_len, 'num_classes', num_classes)

padded_data = [[[0.]*num_classes]*(seq_len-len(d))+d for d in data]
padded_dev_data = [[[0.]*num_classes]*(seq_len-len(d))+d for d in dev_data]
array = np.array(padded_data)
dev_array = np.array(padded_dev_data)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
dev_labels = mlb.transform(dev_labels)

input = Input(shape=(array.shape[1],array.shape[2]))
lstm = Bidirectional(LSTM(array.shape[2]))(input)
output = Dense(array.shape[2], activation='sigmoid')(lstm)

model = Model(inputs=input, outputs=output)
optimizer = Adam(learning_rate=0.005)


metrics = [
    #Precision(thresholds=THRESHOLD),
    #Recall(thresholds=THRESHOLD),
    #F1Score(num_classes=num_classes, threshold=THRESHOLD, average='micro')
    F1Score(num_classes=num_classes, threshold=0.3, name="F1@0.3", average='micro'),
    F1Score(num_classes=num_classes, threshold=0.35, name="F1@0.35", average='micro'),
    F1Score(num_classes=num_classes, threshold=0.4, name="F1@0.4", average='micro'),
]

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=metrics
)

hist = model.fit(
    array,
    labels,
    batch_size=128,
    verbose=1,
    epochs=15,
    validation_data=(dev_array, dev_labels)
)
