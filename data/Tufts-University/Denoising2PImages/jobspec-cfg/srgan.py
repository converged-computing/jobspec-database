# Based on: https://medium.com/analytics-vidhya/implementing-srresnet-srgan-super-resolution-with-tensorflow-89900d2ec9b2
# Paper: https://arxiv.org/pdf/1609.04802.pdf (SRGAN)
# Pytorch Impl: https://github.com/Lornatang/SRGAN-PyTorch/blob/main/model.py

import os
import tensorflow as tf
import keras
import model_builder
import metrics
import RESNET
import train
import shutil
import pathlib
import basics 

# === SRGAN ===
def _get_spatial_ndim(x):
    return keras.backend.ndim(x) - 2

def _get_num_channels(x):
    return keras.backend.int_shape(x)[-1]

def _conv(x, num_filters, kernel_size, padding='same', **kwargs):
    n = _get_spatial_ndim(x)
    if n not in (2, 3):
        raise NotImplementedError(f'{n}D convolution is not supported')

    return (keras.layers.Conv2D if n == 2 else
            keras.layers.Conv3D)(
                num_filters, kernel_size, padding=padding, **kwargs)(x)

def _residual_blocks(x, repeat):
  num_channels = _get_num_channels(x)

  for _ in range(repeat):
    short_skip = x
    x = _conv(x,num_channels,3)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.PReLU()(x)
    x = _conv(x,num_channels,3)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, short_skip])
  return x

def _residual_disc_blocks(x):
  num_channels = _get_num_channels(x)
  channels = [num_channels * n for n in range(1,5)]
  print(channels)

  x = _conv(x,num_channels,3,strides = 2)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.LeakyReLU()(x)
  
  for i in range(len(channels)):
    x = _conv(x,channels[i],3,strides = 1)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = _conv(x,channels[i],3,strides = 2)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
  return x

# Build a discriminator model
def build_discriminator_model(input_shape = (50,256,256,1),
                          *,
                          num_channels,
                          num_residual_blocks,
                          num_channel_out =1):
  print('=== Building Discriminator Model --------------------------------------------')
  inputs = keras.layers.Input(input_shape)
  x = _conv(inputs, num_channels, 3)
  x = keras.layers.LeakyReLU()(x)
  
  
  x = _residual_disc_blocks(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(1024)(x)
  x = keras.layers.LeakyReLU()(x)
  outputs = keras.layers.Dense(1,activation='sigmoid')(x)

  model = keras.Model(inputs,outputs,name='Discriminator')
  print('--------------------------------------------------------------------')

  return model

def build_and_compile_srgan(config):
    learning_rate = config['initial_learning_rate']
    generator = RESNET.build_generator_model((*config['input_shape'], 1),
                num_channels=config['num_channels'],
                num_residual_blocks=config['num_residual_blocks'],
                num_channel_out = 1)
    
    generator = model_builder.compile_model(generator, learning_rate, config['loss'], config['metrics'],0,
                config['ssim_FSize'],config['ssim_FSig'])
    
    discriminator = build_discriminator_model((*config['input_shape'], 1),
                num_channels=config['num_channels'],
                num_residual_blocks=config['num_residual_blocks'],
                num_channel_out =1)
    discriminator.summary()

    return generator, discriminator

def SRGAN_fit_model(model_name, strategy, config, initial_path, output_dir,training_data, validation_data):
    generator, discriminator, care = model_builder.build_and_compile_model(model_name, strategy, config)
    Gen_flag, CARE_flag = basics.SRGAN_Weight_search(pathlib.Path(output_dir))
    if Gen_flag == 1:
        Gen_final_weights_path = str(pathlib.Path(output_dir) / 'Pretrained.hdf5')
    else: 
        generator, Gen_final_weights_path = generator_train(generator, model_name, config, output_dir, training_data, validation_data)
    generator.load_weights(Gen_final_weights_path)
    if CARE_flag == 1:
        CARE_final_weights_path = str(pathlib.Path(output_dir) / 'CARE_Pretrained.hdf5')
    else: 
        if os.path.exists((initial_path + '/Denoising2PImages/' + 'CARE_Pretrained.hdf5')):
            CARE_final_weights_path = (initial_path + '/Denoising2PImages/' + 'CARE_Pretrained.hdf5')
            print(f'CARE Pretrained weights found in GitLab Repository path :{CARE_final_weights_path}')
        else:
            raise Exception('CARE Model needs to be pretrained, please confirm you have weights for standard CARE model')
    care.load_weights(CARE_final_weights_path)

    srgan_checkpoint_dir = str(pathlib.Path(output_dir) / 'ckpt' / 'srgan')
    print(f'Checkpoints saved in {srgan_checkpoint_dir}')
    os.makedirs(srgan_checkpoint_dir, exist_ok=True)
    with strategy.scope():            
        learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

        srgan_checkpoint = tf.train.Checkpoint(psnr=tf.Variable(0.0),
                                            ssim=tf.Variable(0.0), 
                                            generator_optimizer=generator_optimizer,
                                            discriminator_optimizer=discriminator_optimizer,
                                            generator=generator,
                                            discriminator=discriminator)

        srgan_checkpoint_manager = tf.train.CheckpointManager(checkpoint=srgan_checkpoint,
                                directory=srgan_checkpoint_dir,
                                max_to_keep=3)
    
        if srgan_checkpoint_manager.latest_checkpoint:
            srgan_checkpoint.restore(srgan_checkpoint_manager.latest_checkpoint)
        perceptual_loss_metric = tf.keras.metrics.Mean()
        discriminator_loss_metric = tf.keras.metrics.Mean()
        psnr_metric = tf.keras.metrics.Mean()
        ssim_metric = tf.keras.metrics.Mean()
        best_val_ssim = None
        for i in range(config['epochs']):
            for _, batch in enumerate(training_data):
                perceptual_loss, discriminator_loss = strategy.run(train_step, args=(batch,srgan_checkpoint,care))
                perceptual_loss_metric(perceptual_loss)
                discriminator_loss_metric(discriminator_loss)
                lr = batch[0]
                hr = batch[1]
                sr = srgan_checkpoint.generator.predict(lr)
                psnr_value = metrics.psnr(hr, sr)
                hr = tf.cast(hr,tf.double)
                sr = tf.cast(sr,tf.double)
                ssim_value = metrics.ssim(hr, sr)
                psnr_metric(psnr_value)
                ssim_metric(ssim_value)
            CARE_loss = perceptual_loss_metric.result()
            dis_loss = discriminator_loss_metric.result()
            psnr_train = psnr_metric.result()
            ssim_train = ssim_metric.result()
            print(f'Training --> Epoch # {i}: CARE_loss = {CARE_loss:.4f}, Discrim_loss = {dis_loss:.4f}, PSNR = {psnr_train:.4f}, SSIM = {ssim_train:.4f}')
            perceptual_loss_metric.reset_states()
            discriminator_loss_metric.reset_states()
            psnr_metric.reset_states()
            ssim_metric.reset_states()

            srgan_checkpoint.psnr.assign(psnr_train)
            srgan_checkpoint.ssim.assign(ssim_train)


            for _, val_batch in enumerate(validation_data):
                lr = val_batch[0]
                hr = val_batch[1]
                sr = srgan_checkpoint.generator.predict(lr)
                hr_output = srgan_checkpoint.discriminator.predict(hr)
                sr_output = srgan_checkpoint.discriminator.predict(sr)

                con_loss = metrics.calculate_content_loss(hr, sr, care)
                gen_loss = metrics.calculate_generator_loss(sr_output)/len(sr_output)
                perc_loss = con_loss + 0.001 * gen_loss
                disc_loss = metrics.calculate_discriminator_loss(hr_output, sr_output)/len(sr_output)

                perceptual_loss_metric(perc_loss)
                discriminator_loss_metric(disc_loss)

                psnr_value = metrics.psnr(hr, sr)
                hr = tf.cast(hr,tf.double)
                sr = tf.cast(sr,tf.double)
                ssim_value = metrics.ssim(hr, sr)
                psnr_metric(psnr_value)
                ssim_metric(ssim_value)
            CARE_loss = perceptual_loss_metric.result()
            dis_loss = discriminator_loss_metric.result()
            total_ssim = ssim_metric.result()
            psnr_train = psnr_metric.result()
            ssim_train = ssim_metric.result()
            if best_val_ssim == None or total_ssim > best_val_ssim:
                print('New Checkpoint Saved')
                srgan_checkpoint_manager.save()
                best_val_ssim = total_ssim
            print(f'Validation --> Epoch # {i}: CARE_loss = {CARE_loss:.4f}, Discrim_loss = {dis_loss:.4f}, PSNR = {psnr_train:.4f}, SSIM = {ssim_train:.4f}')
            perceptual_loss_metric.reset_states()
            discriminator_loss_metric.reset_states()
            psnr_metric.reset_states()
            ssim_metric.reset_states()
    return srgan_checkpoint, srgan_checkpoint_manager
learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

@tf.function
def train_step(images,srgan_checkpoint,CARE):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        lr = images[0]
        hr = images[1]
        sr = srgan_checkpoint.generator(lr, training=True)
        hr_output = srgan_checkpoint.discriminator(hr, training=True)
        sr_output = srgan_checkpoint.discriminator(sr, training=True)

        con_loss = metrics.calculate_content_loss(hr, sr, CARE)
        gen_loss = metrics.calculate_generator_loss(sr_output)/len(sr_output)
        perc_loss = con_loss + 0.001 * gen_loss
        disc_loss = metrics.calculate_discriminator_loss(hr_output, sr_output)/len(sr_output)

    gradients_of_generator = gen_tape.gradient(perc_loss, srgan_checkpoint.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, srgan_checkpoint.discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, srgan_checkpoint.generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, srgan_checkpoint.discriminator.trainable_variables))

    return perc_loss, disc_loss

def generator_train(generator, model_name, config, output_dir, training_data, validation_data):
    generator = train.fit_model(generator, model_name, config, output_dir,training_data, validation_data)
    os.chdir(pathlib.Path(output_dir))
    model_paths = [model_path for model_path in os.listdir() if model_path.endswith(".hdf5") ]
    assert len(model_paths) != 0, f'No models found under {output_dir}'
    latest = max(model_paths, key=os.path.getmtime)
    final_weights_path = str(pathlib.Path(output_dir) / 'Pretrained.hdf5')
    source = str(pathlib.Path(output_dir) / latest)
    print(f'Location of source file: "{source}"')
    print(f'Location of Final Weights file: "{final_weights_path}"')
    shutil.copy(source, final_weights_path)
    print(f'Pretrained Weights are saved to: "{final_weights_path}"')

    return generator, final_weights_path