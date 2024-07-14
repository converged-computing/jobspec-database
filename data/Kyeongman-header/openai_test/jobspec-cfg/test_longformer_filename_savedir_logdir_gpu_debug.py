import torch
import pickle
from tqdm import tqdm, trange
from dataset_consts import *
import copy
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig,LongformerModel,GPT2Model
import sys


print("gpu : ")
print(torch.cuda.is_available())

testfile_name=sys.argv[1] # 예제 : wp_all_generations_outputs
save_dir=sys.argv[2] #all.tar
log_dir=sys.argv[3] # coh1
gpu=sys.argv[4] # cuda:0 or cpu
PARA=int(sys.argv[5])
debug=int(sys.argv[6]) # 1 or 0

if debug==1:
    debug=True
else:
    debug=False

print("test file : " + testfile_name + ".csv")
print("save dir : " + save_dir)
print("log dir : " + log_dir)
print("gpu or cpu : " + gpu)
print("debug mode : " + str(debug))



CONTINUOUSLY_TRAIN=True


createFolder('longformer')
PATH = './longformer/'+save_dir

writer = SummaryWriter('./runs/'+log_dir)

# tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
if "nextsentenceprediction" in save_dir:
    tokenizer.add_tokens(["[SEP]"],special_tokens=True)

class MyLongformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.config=AutoConfig.from_pretrained('allenai/longformer-base-4096')
        # self.bert = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        self.config=AutoConfig.from_pretrained('gpt2')
        self.gpt = GPT2Model.from_pretrained("gpt2")
        # self.rogistic=torch.nn.Linear(self.config.hidden_size,1)
        if "nextsentenceprediction" in save_dir:
            self.gpt.resize_token_embeddings(len(tokenizer))
        self.rogistic=torch.nn.Linear(self.config.n_embd,1)
        self.sigmoid=torch.nn.Sigmoid()
        self.loss=torch.nn.BCELoss()

    def forward(self, input_ids,attention_mask,global_attention_mask,labels=None):
        

        output=self.gpt(input_ids, attention_mask=attention_mask)
        pooler_output=torch.mean(output.last_hidden_state,dim=-2)
        

        prob=self.rogistic(pooler_output)
        prob=self.sigmoid(prob)
        loss=0
        if labels is not None:
            loss=self.loss(prob,labels)
        return prob, loss

# outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
# print(outputs.pooler_output.shape)
from transformers import get_scheduler
import torch.optim as optim
mylongformer=MyLongformer()
# print(mylongformer(input_ids,attention_mask,global_attention_mask,label=torch.FloatTensor([[1]])))
if torch.cuda.is_available():
     mylongformer=mylongformer.to(gpu)

if CONTINUOUSLY_TRAIN:
    checkpoint= torch.load(PATH)
    mylongformer.load_state_dict(checkpoint['model_state_dict'],strict=False)




def eval(fake_outputs,real_outputs):
    mylongformer.eval()
    with torch.no_grad():
        fake=tokenizer(fake_outputs,max_length=500,padding="max_length",
                    truncation=True,return_tensors="pt")
        input_ids=fake['input_ids'].to(gpu)
        attention_mask=fake['attention_mask'].to(gpu)
        global_attention_mask=torch.zeros_like(attention_mask).to(gpu)
        global_attention_mask[:,0]=1
        if debug:
            print(input_ids)
            print(tokenizer.batch_decode(input_ids,skip_special_tokens=True))
        fake_probs,_=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,)
    
        real=tokenizer(real_outputs,max_length=500,padding="max_length",
                truncation=True,return_tensors="pt")
        input_ids=real['input_ids'].to(gpu)
        attention_mask=real['attention_mask'].to(gpu)
        global_attention_mask=torch.zeros_like(attention_mask).to(gpu)
        global_attention_mask[:,0]=1
        if debug:
            print(input_ids)
            print(tokenizer.batch_decode(input_ids,skip_special_tokens=True))
        real_probs,_=mylongformer(input_ids=input_ids,attention_mask=attention_mask,global_attention_mask=global_attention_mask,)

        del fake
        del real
        del input_ids
        del attention_mask
        del global_attention_mask
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
    return fake_probs.to('cpu'), real_probs.to('cpu')

import csv
import ctypes as ct
import math
import numpy as np
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

f = open(testfile_name+'.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
num_whole_steps=sum(1 for row in rdr)
f.seek(0)
rdr = csv.reader(f)
first=True

count=0
last_keywords=""
cumul_fake_outputs=""
cumul_real_outputs=""
f_score=0
r_score=0
f_scores=[]
r_scores=[]
not_last_fake_scores=[]
not_last_real_scores=[]
step=0
para_count=0
progress_bar = tqdm(range(num_whole_steps))

last_fake=""
last_real=""
not_last_fake=[]
not_last_real=[]

paragraphs=int(testfile_name[-1])
print(paragraphs)
list_of_splitter=[".","!","?",'"',"'"]
for line in rdr:
    
    if first:
        first=False
        continue
    count+=1
    #if count==paragraphs*80:
    #    break
    
    
    progress_bar.update(1)
    
    keywords=line[2].replace('[','').replace(']','')
    fake=line[4].replace('[','').replace(']','')
    #fake=' '.join(sent_tokenize(fake))
    # print(fake)
    sentences=sent_tokenize(fake)
    temp_sentences=copy.deepcopy(sentences)
    for sentence in sentences:
        
        if (sentence[-1] in list_of_splitter) is not True:
            temp_sentences.remove(sentence)
    fake=' '.join(temp_sentences)
    # 완료되지 않은 마지막 문장 삭제해주기.
    real=line[3].replace('[','').replace(']','')
    real=' '.join(sent_tokenize(real))
    #print(para_count)
    if debug:
        print("keywords : " + line[2].replace('[','').replace(']',''))
        print("fake outputs : " + fake)
        print("real outputs : " + line[3].replace('[','').replace(']',''))
        input()
    
    if 'nextsentence' in save_dir : # nextsentenceprediction은 아예 다른 방식이다
        
        if keywords==last_keywords and para_count<(PARA-1):
            cumul_fake_outputs+="[SEP]"+" " + fake+" "
            cumul_real_outputs+="[SEP]"+" " + real+" "
            
            f_score,r_score=eval(cumul_fake_outputs,cumul_real_outputs)

            f_scores.append(f_score.item())
            r_scores.append(r_score.item())
            
            step+=1
            para_count+=1
            #writer.add_scalar("fake score", f_score.item(), step)
            #writer.add_scalar("real score", r_score.item(), step)
            if debug:
                print("eval results : " )
                print("fake : " + cumul_fake_outputs)
                print("real : " + cumul_real_outputs)
                print("keywords : " + last_keywords)
                print("fake score : ")
                print(f_score.item())
                print("real score : ")
                print(r_score.item())
                print("###############")

            cumul_fake_outputs=fake+" "
            cumul_real_outputs=real+" "
        else:
            cumul_fake_outputs=fake+" "
            cumul_real_outputs=real+" "
            last_keywords=keywords
            para_count=0

    else:
        if keywords==last_keywords and para_count<(PARA-1):
            
            cumul_fake_outputs+=fake+" "
            cumul_real_outputs+=real+" "

            not_last_real.append(real+" ")
            not_last_fake.append(fake+" ")
            
            last_real=real+" "
            last_fake=fake+" "
            para_count+=1
            
            continue
        else:
            if count!=1:
                if 'coherence' in save_dir or 'logical' in save_dir:
                    f_score,r_score=eval(cumul_fake_outputs,cumul_real_outputs)
                    f_scores.append(f_score.item())
                    r_scores.append(r_score.item())
                elif 'completeness' in save_dir:
                    not_last_r_scores=0
                    not_last_f_scores=0

                    f_score,r_score=eval(last_fake,last_real)
                    #print(para_count) 
                    for i in range(len(not_last_real)-1):
                        not_last_f_score,not_last_r_score=eval(not_last_fake[i],not_last_real[i])
                        not_last_f_scores+=not_last_f_score.item()
                        not_last_r_scores+=not_last_r_score.item()
                    
                    if len(not_last_real)-1 !=0:
                        not_last_f_scores=not_last_f_scores/(len(not_last_real)-1)
                        not_last_r_scores=not_last_r_scores/(len(not_last_real)-1)
                    
                    not_last_fake_scores.append(not_last_f_scores)
                    not_last_real_scores.append(not_last_r_scores)

                    f_scores.append(f_score.item())
                    r_scores.append(r_score.item())
                
                
                step+=1
                #writer.add_scalar("fake score", f_score.item(), step)
                #writer.add_scalar("real score", r_score.item(), step)

                if debug:
                    print("eval results : " )
                    if 'coherence' in save_dir or 'logical' in save_dir:
                        print("fake : " + cumul_fake_outputs)
                        print("real : " + cumul_real_outputs)
                    elif 'completeness' in save_dir:
                        print("fake : " + last_fake)
                        print("real : " + last_real)
                        print("not last fake : ")
                        print(not_last_fake)
                        print("not last real : ")
                        print(not_last_real)
                        print("not last fake score : ")
                        print(not_last_f_scores)
                        print("not last real score : ")
                        print(not_last_r_scores)

                    print("keywords : " + last_keywords)
                    print("fake score : ")
                    print(f_score.item())
                    print("real score : ")
                    print(r_score.item())
                    
                    print("###############")
                
            cumul_fake_outputs=fake+" "
            cumul_real_outputs=real+" "
            last_keywords=keywords
            last_real=real+" "
            last_fake=fake+" "
            not_last_real=[]
            not_last_fake=[]
            not_last_real.append(real+" ")
            not_last_fake.append(fake+" ")
            para_count=0
            #print(para_count)

if 'coherence' in save_dir or 'logical' in save_dir:
    f_score,r_score=eval(cumul_fake_outputs,cumul_real_outputs)
    f_scores.append(f_score.item())
    r_scores.append(r_score.item())
elif 'completeness' in save_dir:
    not_last_r_scores=0
    not_last_f_scores=0

    f_score,r_score=eval(last_fake,last_real)

    for i in range(len(not_last_real)-1):
        
        not_last_f_score,not_last_r_score=eval(not_last_fake[i],not_last_real[i])
        not_last_f_scores+=not_last_f_score.item()
        not_last_r_scores+=not_last_r_score.item()
    
    if len(not_last_real)-1!=0:
        not_last_f_scores=not_last_f_scores/(len(not_last_real)-1)
        not_last_r_scores=not_last_r_scores/(len(not_last_real)-1)
        
    not_last_fake_scores.append(not_last_f_scores)
    not_last_real_scores.append(not_last_r_scores)
        
    f_scores.append(f_score.item())
    r_scores.append(r_score.item())


f_scores=np.array(f_scores)
r_scores=np.array(r_scores)

if len(not_last_fake_scores)>0:
    not_last_fake_scores=np.array(not_last_fake_scores)
    not_last_real_scores=np.array(not_last_real_scores)

writer.add_scalar("mean fake score", np.mean(f_scores), 0)
writer.add_scalar("mean real score", np.mean(r_scores), 0)
writer.add_scalar("var fake score", np.var(f_scores), 0)
writer.add_scalar("var real score", np.var(r_scores), 0)
if len(not_last_fake_scores)>0:
    writer.add_scalar("mean not last fake score", np.mean(not_last_fake_scores), 0)
    writer.add_scalar("mean not last real score", np.mean(not_last_real_scores), 0)
    writer.add_scalar("var not last fake score", np.var(not_last_fake_scores), 0)
    writer.add_scalar("var not last real score", np.var(not_last_real_scores), 0)


print(testfile_name + "'s " + save_dir + " mean score : " + str(np.mean(f_scores)) + "\n var : " + str(np.var(f_scores)))
print("and this is baseline (original dataset)'s same mean score : " + str(np.mean(r_scores))+ "\n var : " + str(np.var(r_scores)))

if len(not_last_fake_scores)>0:
    print(testfile_name + "'s " + save_dir + " not last mean score : " + str(np.mean(not_last_fake_scores)) + "\n var : " + str(np.var(not_last_fake_scores)))
    print("and this is baseline (original dataset)'s same not last mean score : " + str(np.mean(not_last_real_scores))+ "\n var : " + str(np.var(not_last_real_scores)))

writer.close()
print("writer close")
