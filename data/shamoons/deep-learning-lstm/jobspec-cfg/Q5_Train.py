'''
PS#2
Q5 A Large Character Level LSTM

'''

import re
import numpy as np
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

# Load and clean a text file


def fClean_Load(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    # Clean text
    words = re.findall(r'[a-z\.]+', text.lower())
    return ' '.join(words)


# load text / Complete novel "A Tale of Two Cities"
raw_text = fClean_Load('AToTC.txt')

# organize into sequences of characters

length = 100
lines = list()
for i in range(length, len(raw_text)):
    seq = raw_text[i-length:i+1]
    lines.append(seq)
print('Total lines: %d' % len(lines))

chars = sorted(list(set(''.join(lines))))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]

X = X / float(vocab_size)
X = np.reshape(X, (X.shape[0], length, 1))
y = to_categorical(y, num_classes=vocab_size)

##############################################################################
####################### Select and fit an appropriate model ##################
# 1) LSTM size, 2) Dropout, 3) epochs, and 4) batch_size #####################
##############################################################################

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X, y, epochs=10, verbose=1, batch_size=256)

# Save and test using code from the Q4_Test
model.save('LargeLSTM_model.h5')
dump(mapping, open('LargeLSTM_mapping.pkl', 'wb'))
