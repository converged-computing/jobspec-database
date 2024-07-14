import tensorflow as tf
from tensorflow import keras 


# expected setup here is 1 node with multiple n x GPUs
# each device will run a copy of the model (__replica__)
# at each training step:
# current batch of data (global batch) is split into n sub-batches (local batches)
# each n replicat independently processes the local, run a forward and backward pass
# output gradient of weights with respect to loss of model in local batch
# weight updates originating from local gradients are efficiently merged across n replicas
# this is done at the end of every step so replicas stay in sync

# single host multi deivce API in Keras
strategy = tf.distribute.MirroredStrategy()

print(f'Number of devices: {strategy.num_replicas_in_sync}')

def get_compiled_model():
    # simple 2-layer densely connected neural network
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    return model 

def get_dataset():
    batch_size = 32
    num_val_samples = 10000

    # go get MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # preprocess data
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # reserve num_val_samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]

    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )


# open strategy scope
with strategy.scope():
    # everything that creates variables should be in this scope
    # i.e. model construction and compile

    model = get_compiled_model()

train_dataset, val_dataset, test_dataset = get_dataset()

model.fit(train_dataset, epochs=2, validation_data=val_dataset)

model.evaluate(test_dataset)