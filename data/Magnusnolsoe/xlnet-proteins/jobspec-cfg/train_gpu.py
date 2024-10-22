"""Pretraining on GPUs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import math
import json
import time
import numpy as np

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf

import data_utils
import model_utils
from gpu_utils import assign_to_gpu, average_grads_and_vars
import function_builder
import tensorboard_utils as tb

# GPU config
flags.DEFINE_integer("num_hosts", default=1,
      help="Number of hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")
flags.DEFINE_bool("use_tpu", default=False,
      help="Whether to use TPUs for training.")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_integer("num_passes", default=1,
      help="Number of passed used for training.")
flags.DEFINE_string("record_info_dir", default=None,
      help="Path to local directory containing `record_info-lm.json`.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_string("init_checkpoint", default=None,
      help="checkpoint path for initializing the model.")
flags.DEFINE_string("run_id", default=None,
      help="Id of current training run.")

# Optimization config
flags.DEFINE_float("learning_rate", default=1e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=1.0,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.001,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")
flags.DEFINE_float("adam_epsilon", default=1e-8,
      help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly",
      help="poly or cos")
flags.DEFINE_float("weight_decay", default=0.0,
      help="weight decay")

# Training config
flags.DEFINE_integer("train_batch_size", default=16,
      help="Size of train batch.")
flags.DEFINE_integer("epochs", default=100,
      help="Total number of epochs.")
flags.DEFINE_integer("log_steps", default=1000,
      help="Number of steps for logging training performance.")
flags.DEFINE_integer("save_steps", default=None,
      help="number of steps for model checkpointing.")

# Data config
flags.DEFINE_integer('seq_len', default=0,
      help='Sequence length for pretraining.')
flags.DEFINE_integer('reuse_len', default=0,
      help="How many tokens to be reused in the next batch. "
      "Could be half of seq_len")
flags.DEFINE_bool("bi_data", default=True,
      help="Use bidirectional data streams, i.e., forward & backward.")
flags.DEFINE_integer("mask_alpha", default=6,
      help="How many tokens to form a group.")
flags.DEFINE_integer("mask_beta", default=1,
      help="How many tokens to mask within each group.")
flags.DEFINE_integer("num_predict", default=None,
      help="Number of tokens to predict in partial prediction.")
flags.DEFINE_integer('perm_size', default=None,
  help='perm size.')
flags.DEFINE_bool("uncased", False,
      help="Use uncased inputs or not.")
flags.DEFINE_integer("n_token", 32000, help="Vocab size")

# Model config
flags.DEFINE_integer("mem_len", default=0,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=32,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=8,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=32,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.0,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_string("ff_activation", default="relu",
      help="Activation type used in position-wise feed-forward.")
flags.DEFINE_bool("use_bfloat16", False,
      help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

# Logging config
flags.DEFINE_string("tb_logging_dir", default="logging",
                    help="The directory to save the logs for Tensorboard.")


FLAGS = flags.FLAGS


def get_model_fn():
  def model_fn(features, labels, mems, is_training):
    #### Get loss from inputs
    total_loss, new_mems, monitor_dict = function_builder.get_loss(
        FLAGS, features, labels, mems, is_training)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    # GPU
    #assert is_training
    if is_training:
        all_vars = tf.trainable_variables()
        grads = tf.gradients(total_loss, all_vars)
        grads_and_vars = list(zip(grads, all_vars))
    
        return total_loss, new_mems, grads_and_vars
    else:
        return total_loss, new_mems

  return model_fn


def single_core_graph(is_training, features, mems):
  model_fn = get_model_fn()

  model_ret = model_fn(
      features=features,
      labels=None,
      mems=mems,
      is_training=is_training)

  return model_ret


def create_mems_tf(bsz_per_core):
  mems = [tf.placeholder(dtype=tf.float32,
                         shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model])
          for layer in range(FLAGS.n_layer)]

  return mems


def initialize_mems_np(bsz_per_core):
  mems_np = [np.zeros(shape=[FLAGS.mem_len, bsz_per_core, FLAGS.d_model],
                      dtype=np.float32)
             for layer in range(FLAGS.n_layer)]

  return mems_np

def train(ps_device):
  ##### Get input function and model function

  train_input_fn, record_info_dict = data_utils.get_input_fn(
      info_dir=os.path.join(FLAGS.record_info_dir, "train"),
      split="train",
      bsz_per_host=FLAGS.train_batch_size,
      seq_len=FLAGS.seq_len,
      reuse_len=FLAGS.reuse_len,
      bi_data=FLAGS.bi_data,
      num_hosts=1,
      num_core_per_host=1, # set to one no matter how many GPUs
      perm_size=FLAGS.perm_size,
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      use_bfloat16=FLAGS.use_bfloat16,
      num_predict=FLAGS.num_predict)
  
  valid_input_fn, record_info_dict_valid = data_utils.get_input_fn(
          info_dir=os.path.join(FLAGS.record_info_dir, "valid"),
          split="valid",
          bsz_per_host=FLAGS.train_batch_size,
          seq_len=FLAGS.seq_len,
          reuse_len=FLAGS.reuse_len,
          bi_data=FLAGS.bi_data,
          num_hosts=1,
          num_core_per_host=1,
          perm_size=FLAGS.perm_size,
          mask_alpha=FLAGS.mask_alpha,
          mask_beta=FLAGS.mask_beta,
          use_bfloat16=FLAGS.use_bfloat16,
          num_predict=FLAGS.num_predict)

  # for key, info in record_info_dict.items():
  num_train_batches = record_info_dict["num_batch"]
  tf.logging.info("num of train batches {}".format(record_info_dict["num_batch"]))
  tf.logging.info("num of validation batches {}".format(record_info_dict_valid["num_batch"]))
  
  ##### Create input tensors / placeholders
  bsz_per_core = FLAGS.train_batch_size // FLAGS.num_core_per_host

  params = {
      "batch_size": FLAGS.train_batch_size # the whole batch
  }
  train_set = train_input_fn(params)
  valid_set = valid_input_fn(params)

  t_iter = train_set.make_initializable_iterator()
  example = t_iter.get_next()
  v_iter = valid_set.make_initializable_iterator()
  v_example = v_iter.get_next()

  if FLAGS.num_core_per_host > 1:
    # train set
    examples = [{} for _ in range(FLAGS.num_core_per_host)]
    for key in example.keys():
      vals = tf.split(example[key], FLAGS.num_core_per_host, 0)
      for device_id in range(FLAGS.num_core_per_host):
        examples[device_id][key] = vals[device_id]
    
    # validation set
    v_examples = [{} for _ in range(FLAGS.num_core_per_host)]
    for key in v_example.keys():
        vals = tf.split(v_example[key], FLAGS.num_core_per_host, 0)
        for device_id in range(FLAGS.num_core_per_host):
            v_examples[device_id][key] = vals[device_id]
  else:
    examples = [example]
    v_examples = [v_example]

  ##### Create computational graph
  tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []
  v_tower_mems, v_tower_losses, v_tower_new_mems = [], [], []

  for i in range(FLAGS.num_core_per_host):
    reuse = True if i > 0 else None
    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

      # The mems for each tower is a dictionary
      mems_i = {}
      v_mems_i = {}
      if FLAGS.mem_len:
        mems_i["mems"] = create_mems_tf(bsz_per_core)
        v_mems_i["mems"] = create_mems_tf(bsz_per_core)

      loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
          is_training=True,
          features=examples[i],
          mems=mems_i)
      
      v_loss_i, v_new_mems_i = single_core_graph(
          is_training=False,
          features=v_examples[i],
          mems=v_mems_i)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_new_mems.append(new_mems_i)
      tower_grads_and_vars.append(grads_and_vars_i)
      
      v_tower_mems.append(v_mems_i)
      v_tower_losses.append(v_loss_i)
      v_tower_new_mems.append(v_new_mems_i)

  ## average losses and gradients across towers
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
    grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
  else:
    loss = tower_losses[0]
    grads_and_vars = tower_grads_and_vars[0]
    
  if len(v_tower_losses) > 1:
    v_loss = tf.add_n(v_tower_losses) / len(v_tower_losses)
  else:
    v_loss = v_tower_losses[0]

  ## get train op
  train_op, learning_rate, gnorm = model_utils.get_train_op(FLAGS, None,
      num_train_batches, grads_and_vars=grads_and_vars)
  global_step = tf.train.get_global_step()

  ##### Training loop
  # initialize mems
  tower_mems_np = []
  v_tower_mems_np = []
  for i in range(FLAGS.num_core_per_host):
    mems_i_np = {}
    v_mems_i_np = {}
    for key in tower_mems[i].keys():
      mems_i_np[key] = initialize_mems_np(bsz_per_core)
      v_mems_i_np[key] = initialize_mems_np(bsz_per_core)
    tower_mems_np.append(mems_i_np)
    v_tower_mems_np.append(v_mems_i_np)

  saver = tf.train.Saver()

  gpu_options = tf.GPUOptions(allow_growth=True)

  model_utils.init_from_checkpoint(FLAGS, global_vars=True)
        
  # Create performance summaries for Tensorboard logging
  training_performance_summaries, valid_performance_summaries = tb.tensorboard_setup()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
      gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    
    # variables that are run in the session
    fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]
    v_fetches = [v_loss, v_tower_new_mems]
    
    # Create writers for Tensorboard logging
    info_dict = {
          "id": FLAGS.run_id, 
          "n_layers": FLAGS.n_layers, 
          "d_model": FLAGS.d_model, 
          "n_heads": FLAGS.n_head
    }
    train_summary_writer, valid_summary_writer = tb.create_writers(sess, info_dict, logging_dir=FLAGS.tb_logging_dir)
    
    total_loss, prev_step = 0., -1
    for i in range(FLAGS.epochs):

          # Train loop
          try:
                sess.run(t_iter.initializer)
                while True:
                      feed_dict = {}
                      for i in range(FLAGS.num_core_per_host):
                        for key in tower_mems_np[i].keys():
                          for m, m_np in zip(tower_mems[i][key], tower_mems_np[i][key]):
                            feed_dict[m] = m_np
                      
                      fetched = sess.run(fetches, feed_dict=feed_dict)
                      loss_np, tower_mems_np, curr_step = fetched[:3]
                      total_loss += loss_np
                      print(curr_step)

                      # Log training progress
                      if curr_step > 0 and curr_step % FLAGS.log_steps == 0:
                        curr_loss = total_loss / (curr_step - prev_step)
                        summ = tb.run_train(sess, training_performance_summaries, curr_loss)
                        train_summary_writer.add_summary(summ, curr_step)
                        tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
                              "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
                              curr_step, fetched[-3], fetched[-2],
                              curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
                        total_loss, prev_step = 0., curr_step

                      # Save checkpoint
                      if curr_step > 0 and FLAGS.save_steps is not None and curr_step % FLAGS.save_steps == 0:
                        save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
                        saver.save(sess, save_path)
                        tf.logging.info("Model saved in path: {}".format(save_path))

          except tf.errors.OutOfRangeError:
                pass
          
          # Validation loop
          try:
              sess.run(v_iter.initializer)
              v_total_loss, v_steps = 0., 0
              while True:
                  v_feed_dict = {}
                  for i in range(FLAGS.num_core_per_host):
                    for key in v_tower_mems_np[i].keys():
                      for m, m_np in zip(v_tower_mems[i][key], v_tower_mems_np[i][key]):
                       v_feed_dict[m] = m_np
                  
                  v_fetched = sess.run(v_fetches, feed_dict=v_feed_dict)
                  v_loss_np, v_tower_mems_np = v_fetched[:]
                  v_total_loss += v_loss_np
                  v_steps += 1
                
          except tf.errors.OutOfRangeError:
              val_loss = v_total_loss/v_steps
              v_pplx = math.exp(val_loss)
              tf.logging.info("Validation: [{}] | loss {:.2f} | pplx {:>7.2f}"
                                    .format(curr_step, val_loss, v_pplx))
              
              summ_valid = tb.run_valid(sess, valid_performance_summaries, val_loss, v_pplx)
              valid_summary_writer.add_summary(summ_valid, curr_step)
          
          tf.logging.info("------------ Epoch {} ------------".format(i))



def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get corpus info
  FLAGS.n_token = data_utils.VOCAB_SIZE
  tf.logging.info("n_token {}".format(FLAGS.n_token))

  if not tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.MakeDirs(FLAGS.model_dir)
    
  if not tf.gfile.Exists(FLAGS.tb_logging_dir):
    tf.gfile.MakeDirs(FLAGS.tb_logging_dir)

  train("/gpu:0")


if __name__ == "__main__":
  tf.app.run()
