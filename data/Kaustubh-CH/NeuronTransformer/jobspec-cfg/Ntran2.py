from toolbox.Dataloader_H5 import Dataset_h5_neuronInverter
from transformers import  TimeSeriesTransformerModel,Trainer, TrainingArguments
from transformers import AutoFeatureExtractor,AutoModelForAudioClassification
import pandas as pd
import os
import argparse
from torch.utils.data import  DataLoader
# from datasets import Dataset


from torch import nn
import torch
from transformers import Trainer
from typing import Any, Dict, Union

from torch.cuda.amp import autocast
from transformers import AutoConfig
from TransformerModel import Wav2Vec2ForNeuronData

from torch.utils.data import Dataset, DataLoader
import numpy as np
import evaluate


conf={
    'domain':'train'
    
}
conf['world_rank'] = 0
# conf['world_rank']=os.environ['SLURM_PROCID']
conf['world_size']=int(os.environ['SLURM_NTASKS'])
conf['world_size']=1
conf['cell_name']="L5_TTPC1cADpyr0"
conf['shuffle']=True
conf['local_batch_size']=128
conf['data_path']='/pscratch/sd/k/ktub1999/bbp_May_18_8944917/'
conf['h5name']=os.path.join(conf['data_path'],conf['cell_name']+'.mlPack1.h5')
conf['probs_select']=[0]
conf['stims_select']=[0]
conf['max_glob_samples_per_epoch']=5000000


class Training():

  

  def __init__(self):
    conf['domain']='train'
    self.train_data=Dataset_h5_neuronInverter(conf,1)
   
    conf['domain']='valid'
    self.valid_data=Dataset_h5_neuronInverter(conf,1)



  

 
  def train(self):
    print(len(self.train_data))
    
    
    loss_fct=nn.MSELoss()
    training_args = TrainingArguments(
    # output_dir='./results/',
    output_dir='./results/'+str(os.environ['SLURM_JOB_ID']),          # output directory
    num_train_epochs=100,              # total number of training epochs
    evaluation_strategy = "epoch",
    per_device_train_batch_size=128,  # batch size per device during training
    per_device_eval_batch_size=128,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    report_to="wandb",
    metric_for_best_model = 'mse',
    save_strategy ="epoch",
    save_steps =1
    )
    
    pooling_mode ="mean"
     
    config = AutoConfig.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=19,
    )
    setattr(config, 'pooling_mode', pooling_mode)

    model = Wav2Vec2ForNeuronData.from_pretrained(
    "facebook/wav2vec2-base",
    config=config
    )
    
    # def compute_metrics(eval_pred):
    
    #     predictions, labels = eval_pred
    #     # predictions = np.argmax(logits, axis=-1)
    #     return self.metric.compute(predictions=predictions, references=labels)
    def compute_metrics(eval_pred):
      from sklearn.metrics import mean_squared_error

      predictions, labels = eval_pred
      rmse = mean_squared_error(labels, predictions)
      return {"mse": rmse}

    
    # model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=19)
    

        
    # TimeSeriesTransformerModel.from_pretrained("huggingface/time-series-transformer-tourism-monthly")

    print("Training")
    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=self.train_data,         # training dataset
    eval_dataset=self.valid_data, # evaluation dataset
    compute_metrics=compute_metrics
            
    )
    trainer.train()
    trainer.save_model()


def get_parser():  
  parser = argparse.ArgumentParser()
  parser.add_argument("-dp","--data_path", default='/pscratch/sd/k/ktub1999/bbp_May_18_8944917', type=str)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_parser()
  conf['data_path']=args.data_path
  print("Config",conf)
  T = Training()
  T.train()
