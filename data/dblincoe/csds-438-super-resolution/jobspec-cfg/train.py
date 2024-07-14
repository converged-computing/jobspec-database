import os

import click
import tensorflow as tf
from tensorflow.keras.losses import Loss, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.keras.utils.generic_utils import default

from data.data_loaders import load_images_from_folder
from data.data_utils import (
    augment_images,
    create_shuffled_batches,
    downsample_images,
    split_train_valid_images,
)
from models.common import evaluate
from models.edsr import EDSR
from models.srgan import SRResNet


class Trainer:
    def __init__(
        self,
        model: EDSR(),
        loss: Loss,
        learning_rate: None,
        fresh_run: bool = False,
        checkpoint_dir_base: str = "./ckpts/",
        saved_model_dir_base: str = "./output/",
    ) -> None:

        self.model = model
        self.loss = loss

        self.optimizer = Adam(learning_rate)

        self.checkpoint = tf.train.Checkpoint(
            model=model, optimizer=Adam(learning_rate)
        )

        self.checkpoint_mngr = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(checkpoint_dir_base, model.save_name),
            max_to_keep=2,
        )

        self.saved_model_dir_base = saved_model_dir_base

        if not fresh_run:
            self.rebuild()

    def train(self, train_data, valid_data, epochs):
        """Trains the model using a training and validation set"""

        # if the total number of runs is greater than epochs, save the model and exit
        if (
            self.checkpoint_mngr.latest_checkpoint
            and int(self.checkpoint_mngr.latest_checkpoint.split("-")[-1]) > epochs
        ):
            self.model.save(os.path.join(self.saved_model_dir_base, self.model.save_name))
            return

        for epoch in range(epochs):

            loss_mean = Mean()

            for lr_img_batch, hr_img_batch in train_data:
                loss = self.train_step(lr_img_batch, hr_img_batch)
                loss_mean(loss)

            # Compute PSNR on validation dataset
            psnr_value, ssim_value = evaluate(self.model, valid_data)
            print(
                f"Epoch {epoch+1}: Loss = {loss_mean.result().numpy()}, PSNR = {psnr_value.numpy()}, SSIM = {ssim_value.numpy()}"
            )

            self.checkpoint_mngr.save()

        print("Done Training!")

        self.model.save(os.path.join(self.saved_model_dir_base, self.model.save_name))

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr, hr = tf.cast(lr, tf.float32), tf.cast(hr, tf.float32)

            sr = self.model(lr, training=True)
            step_loss = self.loss(hr, sr)

            gradients = tape.gradient(step_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

        return step_loss

    def rebuild(self):
        """Rebuilds the checkpoint"""
        if self.checkpoint_mngr.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_mngr.latest_checkpoint)


@click.command()
@click.option(
    "--model", type=click.Choice(["edsr", "srresnet", "srgan"]), default="edsr"
)
@click.option("--scale", type=int, default=4)
@click.option("--epochs", type=int, default=10)
@click.option("--patch-size", type=int, default=100)
@click.option("--max-patch-num", type=int, default=100)
@click.option("--batch-size", type=int, default=16)
@click.option("--data-folder", type=str, default='data-default')
@click.option("--name-suffix", type=str, default="")
@click.option("--train-test-split", type=float, default=0.8)
@click.option("--fresh-run", is_flag=True)
def main(
        model, 
        scale, 
        epochs, 
        patch_size, 
        max_patch_num, 
        batch_size, 
        data_folder, 
        name_suffix,
        train_test_split, 
        fresh_run
    ):

    # Load, augment, and downsample images
    load_hr_imgs = load_images_from_folder(data_folder)
    hr_imgs = augment_images(
        load_hr_imgs, patch_size=(patch_size, patch_size), max_patches=max_patch_num
    )
    lr_imgs = downsample_images(hr_imgs, scale)
    
    # Split and batch data
    train_data, valid_data = split_train_valid_images(lr_imgs, hr_imgs, train_test_split)
    batched_train_data = create_shuffled_batches(train_data, batch_size)

    # Select Model
    if model == "edsr":
        model_obj = EDSR(scale=scale, name_suffix=name_suffix)
        loss = MeanAbsoluteError()
    elif model == 'srresnet':
        model_obj = SRResNet(scale=scale, name_suffix=name_suffix)
        loss = MeanAbsoluteError()
    elif model == 'srgan':
        # TODO: Impliment this
        raise ValueError(f"Not implimented yet")

    # Build trainer object
    trainer = Trainer(
        model=model_obj,
        loss=loss,
        learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]),
        fresh_run=fresh_run,
    )

    # Train model
    trainer.train(batched_train_data, valid_data, epochs=epochs)


if __name__ == "__main__":
    main()
