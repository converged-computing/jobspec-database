import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import socket

import warnings
warnings.filterwarnings("ignore")

from parrot import Parrot

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

df = pd.read_csv(sys.argv[1])

all_paraphrases = []

hostname = socket.gethostname()
if "gv" in hostname:
  batch_size = 24
elif "gr" in hostname:
  batch_size = 36
else:
  batch_size = 48

chunks = (len(df) - 1) // batch_size + 1
data = df['SEG'].to_list()
for i in tqdm(range(chunks)):
  phrase = data[i*batch_size:(i+1)*batch_size]
#[["Can you recommed some upscale restaurants in Rome?", "This is great."]]
#for phrase in df['SEG']:
  #print("-"*100)
  #print(phrase)
  #print("-"*100)
  para_phrases = parrot.augment(input_phrase_lst=phrase,
                                use_gpu=True,
                                diversity_ranker="levenshtein",
                                do_diverse=False, 
                                max_return_phrases = 40, 
                                max_length=32, 
                                adequacy_threshold = 0.85, 
                                fluency_threshold = 0.8)

  #for orig, paraphrase in zip(para_phrases[0], para_phrases[1]):
      #print(orig)
      #print(paraphrase)
  all_paraphrases += para_phrases[1]

df['paraphrases'] = all_paraphrases
df.to_csv(sys.argv[1].replace('.csv','_para.csv'), index=False)

  
 
  
