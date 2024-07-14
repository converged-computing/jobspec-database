import os
import tensorflow as tf
from nn.networks import NetworkGenerator
from nn.losses import *
from nn.image_loader import load_images
from nn.arg_parser import ArgParser
from nn.util import setup_directories

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tf.config.run_functions_eagerly(True)



def train_model(x_train, y_train, imaging_method, model_name, loss_name, stride):
    print(f"\nTraining model '{model_name}' with loss '{loss_name}' and a stride of {stride}\n")
    artifact_remover = NetworkGenerator.get_model(model_name, stride)

    # loss, early stopping and optimizer
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = sobel_loss if loss_name == "sobel" else ssim_loss
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=50,
                                                      restore_best_weights=True)

    artifact_remover.compile(loss=loss, optimizer=optim)
    artifact_remover.fit(x_train,
            y_train,
            epochs=200,
            shuffle=False,
            batch_size=10,
            verbose=2,
            callbacks=[early_stopping],
            validation_data=(x_test, y_test))

    artifact_remover.save(f"./saved_model/{imaging_method.lower()}_{model_name}_{loss_name}_{stride}_trained_model.h5")
    return artifact_remover


if __name__ == "__main__":
    setup_directories()
    parser = ArgParser()

    resize = False
    if (parser.args.stride == 2):
        resize = True

    x_train, y_train, x_test, y_test = load_images(parser.args.imaging_method, parser.args.nimages, 0.2, resize)
    
    train_model(x_train, y_train, parser.args.imaging_method, parser.args.model, parser.args.loss, parser.args.stride)

