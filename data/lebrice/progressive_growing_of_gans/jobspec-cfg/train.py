# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import numpy as np
import tensorflow as tf

import config
import tfutil
import dataset
import misc
from enum import Enum

import gaussian_blur


class BlurScheduleType(Enum):
    NOBLUR = "NOBLUR"
    LINEAR = "LINEAR"
    EXPONENTIAL_DECAY = "EXPDECAY"
    RANDOM = "RANDOM"

def scale_schedule(cur_nimg: int, total_kimg: int, blur_type: BlurScheduleType, image_resolution = 128) -> float:
    """
    a schedule for the blurring std.
    """
    progress_percentage = cur_nimg / (total_kimg * 1000)
    initial_value = gaussian_blur.maximum_reasonable_std(image_resolution)
    final_value = 0.01 # desired value at the end of training.

    # Weird enum behaviour.
    blur_type = BlurScheduleType[blur_type.name]

    if blur_type is BlurScheduleType.EXPONENTIAL_DECAY:
        decay_rate = np.log(final_value / initial_value)
        return initial_value * np.exp(decay_rate * progress_percentage)
    elif blur_type is BlurScheduleType.LINEAR:
        # linear decay from highest STD to lowest std.
        return initial_value + progress_percentage * (final_value - initial_value)
    elif blur_type is BlurScheduleType.RANDOM:
        return np.random.uniform(final_value, initial_value)
    else:
        # No blurring.
        return 0

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.

def setup_snapshot_image_grid(G, training_set,
    size    = '1080p',      # '1080p' = to be viewed on 1080p display, '4k' = to be viewed on 4k display.
    layout  = 'random'):    # 'random' = grid contents are selected randomly, 'row_per_class' = each row corresponds to one class label.

    # Select size.
    gw = 1; gh = 1
    if size == '1080p':
        gw = np.clip(1920 // G.output_shape[3], 3, 32)
        gh = np.clip(1080 // G.output_shape[2], 2, 32)
    if size == '4k':
        gw = np.clip(3840 // G.output_shape[3], 7, 32)
        gh = np.clip(2160 // G.output_shape[2], 4, 32)

    # Fill in reals and labels.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=training_set.dtype)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        while True:
            real, label = training_set.get_minibatch_np(1)
            if layout == 'row_per_class' and training_set.label_size > 0:
                if label[0, y % training_set.label_size] == 0.0:
                    continue
            reals[idx] = real[0]
            labels[idx] = label[0]
            break

    # Generate latents.
    latents = misc.random_latents(gw * gh, G)
    return (gw, gh), reals, labels, latents

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tfutil.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 4,        # Image resolution used at the beginning.
        lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
        lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
        minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        G_lrate_base            = 0.001,    # Learning rate for the generator.
        G_lrate_dict            = {},       # Resolution-specific overrides.
        D_lrate_base            = 0.001,    # Learning rate for the discriminator.
        D_lrate_dict            = {},       # Resolution-specific overrides.
        tick_kimg_base          = 160,      # Default interval of progress snapshots.
        tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:20, 1024:10}, # Resolution-specific overrides.
        ): 

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = self.kimg - phase_idx * phase_dur

        # Level-of-detail and resolution.
        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))

        # Minibatch size.
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % config.num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * config.num_gpus)

        # Other parameters.
        self.G_lrate = G_lrate_dict.get(self.resolution, G_lrate_base)
        self.D_lrate = D_lrate_dict.get(self.resolution, D_lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)


#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_progressive_gan(
    G_smoothing             = 0.999,        # Exponential running average of generator weights.
    D_repeats               = 1,            # How many times the discriminator is trained per G iteration.
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,         # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 15000,        # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,        # Enable mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 1,            # How often to export image snapshots?
    network_snapshot_ticks  = 10,           # How often to export network snapshots?
    save_tf_graph           = False,        # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,        # Include weight histograms in the tfevents file?
    resume_run_id           = None,         # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,         # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0.0,          # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0,           # Assumed wallclock time at the beginning. Affects reporting.
    resume_tick: int = 0,
    blur_schedule_type: BlurScheduleType = BlurScheduleType.NOBLUR,
    ):

    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            G, D, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tfutil.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config.G)
            D = tfutil.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **config.D)
            Gs = G.clone('Gs')
        Gs_update_op = Gs.setup_as_moving_average_of(G, beta=G_smoothing)
    G.print_layers(); D.print_layers()

    print('Building TensorFlow graph...')

    with tf.name_scope('Inputs'):
        default_scale = tf.constant(0.0)
        print_op = tf.print("default scaled was used (no blurring)", default_scale)
        with tf.control_dependencies([print_op]):
            default_scale = tf.identity(default_scale)

        scale = tf.placeholder_with_default(default_scale, shape=[], name="scale")
        # added default values to not have to pass it in the feed_dict all the time.
        lod_in_default = tf.constant(0.0, name="lod_in")
        lr_default = tf.constant(1e-4)
        minibatch_default = tf.constant(16)
        lod_in = tf.placeholder_with_default(lod_in_default, name='lod_in', shape=[])
        lrate_in        = tf.placeholder_with_default(lr_default, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder_with_default(minibatch_default, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        reals, labels   = training_set.get_minibatch_tf()
        reals_split     = tf.split(reals, config.num_gpus)
        labels_split    = tf.split(labels, config.num_gpus)
    G_opt = tfutil.Optimizer(name='TrainG', learning_rate=lrate_in, **config.G_opt)
    D_opt = tfutil.Optimizer(name='TrainD', learning_rate=lrate_in, **config.D_opt)
    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)]
            reals_gpu = process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            labels_gpu = labels_split[gpu]
            with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                G_loss = tfutil.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set, minibatch_size=minibatch_split, scale=scale, **config.G_loss)
            with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                D_loss = tfutil.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_split, reals=reals_gpu, labels=labels_gpu, scale=scale, **config.D_loss)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

    print('Setting up snapshot image grid...')
    grid_size, grid_reals, grid_labels, grid_latents = setup_snapshot_image_grid(G, training_set, **config.grid)
    sched = TrainingSchedule(total_kimg * 1000, training_set, **config.sched)
    
    print("Blur schedule type:", blur_schedule_type)

    grid_fakes = Gs.run(grid_latents, grid_labels, minibatch_size=sched.minibatch//config.num_gpus)

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc, resume_run_id)
    misc.save_image_grid(grid_reals, os.path.join(result_subdir, 'reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)
    misc.save_image_grid(grid_fakes, os.path.join(result_subdir, 'fakes%06d.png' % int(resume_kimg)), drange=drange_net, grid_size=grid_size)
    summary_log = tf.summary.FileWriter(result_subdir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = resume_tick
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    prev_lod = -1.0

    
    scale_value: float 

    while cur_nimg < total_kimg * 1000:
        # Choose training parameters and configure training ops.
        sched = TrainingSchedule(cur_nimg, training_set, **config.sched)
        scale_value = scale_schedule(cur_nimg, total_kimg, blur_schedule_type)

        training_set.configure(sched.minibatch, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops.
        for repeat in range(minibatch_repeats):
            for _ in range(D_repeats):
                # get the scale value for this batch.
                scale_value = scale_schedule(cur_nimg, total_kimg, blur_schedule_type)

                tfutil.run([D_train_op, Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch, scale: scale_value})
                cur_nimg += sched.minibatch
            tfutil.run([G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch, scale: scale_value})

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d scale %-2.3f time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %.1f' % (
                tfutil.autosummary('Progress/tick', cur_tick),
                tfutil.autosummary('Progress/kimg', cur_nimg / 1000.0),
                tfutil.autosummary('Progress/lod', sched.lod),
                tfutil.autosummary('Progress/minibatch', sched.minibatch),
                tfutil.autosummary('Progress/scale', scale_value),
                misc.format_time(tfutil.autosummary('Timing/total_sec', total_time)),
                tfutil.autosummary('Timing/sec_per_tick', tick_time),
                tfutil.autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                tfutil.autosummary('Timing/maintenance_sec', maintenance_time)))
            tfutil.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            tfutil.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            tfutil.save_summaries(summary_log, cur_nimg, scale=scale_value)

            # Save snapshots.
            if cur_tick % image_snapshot_ticks == 0 or done:
                grid_fakes = Gs.run(grid_latents, grid_labels, minibatch_size=sched.minibatch//config.num_gpus)
                misc.save_image_grid(grid_fakes, os.path.join(result_subdir, 'raw_fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)
                
                blurred_fakes = tfutil.run([gaussian_blur.image_at_scale(grid_fakes, scale)], {scale: scale_value})
                misc.save_image_grid(blurred_fakes[0], os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)

            if cur_tick % network_snapshot_ticks == 0 or done:
                misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000)))

            # Record start time of the next tick.
            tick_start_time = time.time()

    # Write final results.
    misc.save_pkl((G, D, Gs), os.path.join(result_subdir, 'network-final.pkl'))
    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.



if __name__ == "__main__":
    
    import sys
    import argparse
    argv = sys.argv
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Training Script',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    parser.add_argument("--resume-run-id", type=int, default=None)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("-b", "--blur-schedule", type=str, choices=["NOBLUR", "LINEAR", "EXPDECAY", "RANDOM", "NOBLUR"], default=None)
    parser.add_argument("-nimg", "--train-k-images", type=int, default=1_000)
    
    args = parser.parse_args(argv[1:])
    print("Arguments used:", args)
    print("run_name:", args.run_name)
    
    # if we resume a run, we might want to choose a different kind of blurring for the rest of the run?
    if args.resume_run_id:
        print("we are resuming a previous training session with run id:", args.resume_run_id)
        misc.restore_config(args.resume_run_id, config)

    if args.blur_schedule is not None:
        config.train.blur_schedule_type = BlurScheduleType(args.blur_schedule)
    else:
        config.train.blur_schedule_type = config.train.get("blur_schedule_type", BlurScheduleType.NOBLUR)
    print("Chosen blur schedule type: ", config.train.blur_schedule_type)

    if args.run_name:
        config.desc = args.run_name
    else:
        config.desc += f"-{config.train.blur_schedule_type.value}"
    
    print("config desc:", config.desc)    
    config.train.total_kimg = args.train_k_images

    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    print('Running %s()...' % config.train['func'])
    tfutil.call_func_by_name(**config.train)
    print('Exiting...')

#----------------------------------------------------------------------------
