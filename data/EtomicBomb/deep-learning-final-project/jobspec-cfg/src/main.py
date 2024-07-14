#!/usr/bin/env python

import time
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from argparse import ArgumentParser
import time
from tensorflow.data import Dataset
from typing import Literal

from dataset import get_dataset, get_index
from augment import VideoRandomPerspective, VideoRandomFlip, VideoRandomContrast, VideoRandomMultiply, VideoRandomAdd, VideoRandomNoise, VideoCropAndResize, ClipZeroOne, Scale, Gray2RGB
from dimensions import Dimensions

def get_data(
    mode: Literal['train', 'test'],
    extract_root: str, 
    data_root: str, 
    batch_size: int,
    frame_count: int,
    validation_steps: int = None,
    exclude: int = set(),
):
    src_shape = Dimensions(
        batch_size=batch_size,
        frame_count=frame_count,
        height=112,
        width=224,
        channels=1,
    )
    data = Dataset.load(str(Path(data_root, f'{mode}{frame_count}.dataset')))
    data = data.shuffle(data.cardinality(), reshuffle_each_iteration=True, seed=42)
    if mode == 'test':
        data = data.take(validation_steps)
    data = data.repeat()
    data = data.map(lambda pts, path, label: (pts, tf.strings.join([extract_root, path]), label))
    data = get_dataset(data, s=src_shape)

    model = keras.Sequential([
        keras.Input(shape=src_shape.example_shape, batch_size=src_shape.batch_size),
        VideoCropAndResize(),
        Scale(),
        Gray2RGB(),
    ])

    if mode == 'train':
        def maybe(tag: str, value: any):
            return [value] if tag not in exclude else []
        rng = tf.random.Generator.from_non_deterministic_state()
        model = keras.Sequential([
            keras.Input(shape=src_shape.example_shape, batch_size=src_shape.batch_size),
            *maybe('noise', VideoRandomNoise(rng=rng)),
            *maybe('perspective', VideoRandomPerspective(rng=rng)),
            VideoRandomFlip(rng=rng),
            *maybe('contrast', VideoRandomContrast(rng=rng)),
            *maybe('madd', VideoRandomMultiply(rng=rng)),
            *maybe('madd', VideoRandomAdd(rng=rng)),
            ClipZeroOne(),
            model,
        ])
    data = data.map(
        lambda x, y: (model(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    data = data.prefetch(tf.data.AUTOTUNE)
    return data

def train_test(extract_root: str, data_root: str, batch_size=2, frame_count=32, validation_steps=20):
    train = get_data(
        mode='train',
        extract_root=extract_root,
        data_root=data_root,
        batch_size=batch_size,
        frame_count=frame_count,
        validation_steps=validation_steps,
    )
    test = get_data(
        mode='test',
        extract_root=extract_root,
        data_root=data_root,
        batch_size=batch_size,
        frame_count=frame_count,
        validation_steps=validation_steps,
    )

    return train, test

def index(exclude: set[str], run_id: str, learning_rate: float, filename: str):
    frame_count = 32
    data_split = json.loads(Path('data/split.json').read_text())
    data = get_index('data/extract', paths=data_split['train'], frame_count=frame_count)
    Dataset.save(data, f'data/train{frame_count}.dataset')
    data = get_index('data/extract', paths=data_split['test'], frame_count=frame_count)
    Dataset.save(data, f'data/test{frame_count}.dataset')

def demo(exclude: set[str], run_id: str, learning_rate: float, filename: str):
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    data = get_data(
        mode='train',
        extract_root='data/extract/',
        data_root='data',
        batch_size=3,
        frame_count=32,
        validation_steps=20,
        exclude=exclude,
    )

    fig, ax = plt.subplots()
    print(f"data: {data}")
    data = data.as_numpy_iterator()
    
    data = iter((frame, label) for batch, labels in data for video, label in zip(batch, labels) for frame in video)
    image = ax.imshow(next(data)[0], cmap='gray')
    def animate(data):
        x, y = data
        image.set_data(x)
        print(y)
        return [image]
    ani = FuncAnimation(fig, animate, data, cache_frame_data=False, blit=True, interval=1)
    plt.show()

@keras.utils.register_keras_serializable()
class VideoMobileNet(keras.layers.Layer):
    def __init__(self, *args, start=None, end=None, trainable=True, **kwargs):
        super().__init__(*args, **kwargs)
        model = keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
        start = model.get_layer(start) if start is not None else model.layers[0]
        end = model.get_layer(end)
        model = keras.Model(inputs=start.output,outputs=end.output)
        for layer in model.layers:
           layer.trainable = trainable
        self.model = model
    def call(self, x):
        batch_size, frame_count, height, width, channels = x.shape
        x = tf.reshape(x, (batch_size*frame_count, height, width, channels))
        x = self.model(x)
        _, height, width, channels = x.shape
        x = tf.reshape(x, (batch_size, frame_count, height, width, channels))
        return x

@keras.utils.register_keras_serializable()
class Video1DConvolution(keras.layers.Layer):
    def __init__(self, filters, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = keras.layers.Conv1D(filters, kernel_size, padding='same', data_format='channels_first')
    def call(self, x):
        batch_size, frame_count, height, width, channels = x.shape
        x = tf.reshape(x, (batch_size, frame_count, height * width * channels))
        x = self.m(x)
        x = tf.reshape(x, (batch_size, x.shape[1], height, width, channels))
        return x

def model_two_plus_one():
    s = Dimensions(
        batch_size=4,
        frame_count=32,
        height=224,
        width=224,
        channels=3,
    )

    return keras.Sequential([
        keras.Input(shape=s.example_shape, batch_size=s.batch_size),
        tf.keras.layers.Rescaling(2.0, -1.0), # [0,1] -> [-1, 1]
        VideoMobileNet(start=None,end='block_3_depthwise', trainable=False),
        # Video1DConvolution(32, 20),
        VideoMobileNet(start='block_3_depthwise',end='block_6_depthwise', trainable=False),
        # Video1DConvolution(32, 20),
        VideoMobileNet(start='block_6_depthwise',end='block_13_depthwise', trainable=False),
        Video1DConvolution(32, 20),
        VideoMobileNet(start='block_13_depthwise',end='block_16_expand', trainable=True),
        Video1DConvolution(32, 20),
        VideoMobileNet(start='block_16_expand',end='out_relu', trainable=True),
        Video1DConvolution(32, 20),
        keras.layers.Conv3D(1, (10, 3, 3), strides=(3, 3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(3, activation='sigmoid'),
    ])

def experiment(exclude: set[str], run_id: str, learning_rate: float, filename: str):
    model = model_two_plus_one()

    train_data = get_data(
        mode='train',
        extract_root='data/extract/',
        data_root='data',
        batch_size=3,
        frame_count=32,
        exclude=exclude,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'data/checkpoints/{run_id}-{timestamp}-{{epoch:3d}}.keras',
    )
    history = model.fit(
        train_data,
        steps_per_epoch=2000,
        epochs=10, 
        callbacks=[model_checkpoint_callback],
    )
    print(history)

    model.summary()

def evaluate(exclude: set[str], run_id: str, learning_rate: float, filename: str):
    validation_steps = 1000
    test_data = get_data(
        mode='test',
        extract_root='data/extract/',
        data_root='data',
        batch_size=3,
        frame_count=32,
        validation_steps=validation_steps,
        exclude=exclude,
    )
    model = model_two_plus_one()
    model.load_weights(filename)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    result = model.evaluate(test_data, steps=validation_steps, return_dict=True)
    Path(filename).with_suffix('.json').write_text(json.dumps(result))

if __name__ == '__main__':
    modes = dict(demo=demo, index=index, experiment=experiment, evaluate=evaluate)
    parser = ArgumentParser(
        prog='drowsiness classifier',
        description='sees if someone is drowsy',
    )
    parser.add_argument('-m', '--mode', choices=list(modes), default=next(iter(modes)))
    parser.add_argument('--exclude', nargs='*', help='exclude augmentations')
    parser.add_argument('--run-id', default='run', help='the name of checkpoints')
    parser.add_argument('--learning-rate', default=1e-7, type=float, help='learning rate')
    parser.add_argument('--filename', default='', help='filename for evaluation')
    args = parser.parse_args()
    print('received arguments', args)
    modes[args.mode](
        set(args.exclude) if args.exclude is not None else set(),
        run_id=args.run_id,
        learning_rate=args.learning_rate,
        filename=args.filename,
    )
