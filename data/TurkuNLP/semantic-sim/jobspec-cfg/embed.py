import sys
print("Pre-import",file=sys.stderr,flush=True)
import torch
print("torch done",file=sys.stderr,flush=True)
import transformers
print("transf done",file=sys.stderr,flush=True)
import embed_data
print("edata done",file=sys.stderr,flush=True)
from torch.utils.data import DataLoader
import tqdm
import pickle
print("all done",file=sys.stderr,flush=True)

def embed_batch(batch,bert_model):
    input_ids=batch["enc"].cuda()
    attention_mask=batch["attention_mask"].cuda()
    token_type_ids=batch["token_type_id"].cuda()
    spec_token_mask=batch["spec_token_mask"].cuda()
    emb=bert_model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    #emb.pooler_output ## CLS + tanh
    last_hidden=emb.last_hidden_state # batch x token x embdim
    #multiply last_hidden with the attention mask and inversed special token mask (special token mask is 1 for CLS and SEP, 0 otherwise, I need the opposite
    #this basically says we are taking a sum over all real tokens
    mask_of_interest=attention_mask*(spec_token_mask*-1+1) #batch x token
    mask_of_interest_sum=torch.sum(mask_of_interest,dim=-1) #batch ... count of unmasked tokens

    last_hidden_masked=last_hidden.mul(mask_of_interest.unsqueeze(-1))
    last_hidden_masked_sum=torch.sum(last_hidden_masked,dim=1) #
    del emb, last_hidden_masked
    #last_hidden_masked_sum=torch.sum(last_hidden_masked,dim=1) #sum over words

    last_hidden_mean=torch.div(last_hidden_masked_sum,mask_of_interest_sum.unsqueeze(-1))
    #These are the embeddings!
    return last_hidden_mean



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-model", default=None, help="BERT model name or path")
    parser.add_argument("--out",default=None,help="File to save batches into")
    parser.add_argument("--thisjob",default=0,type=int,help="Set to the number which this job is out of jobs, zero based. It will embed lines where line id modulo jobs equals thisjob Default: %(default)s")
    parser.add_argument("--jobs",default=1,type=int,help="Set to N if you are splitting the work among N workers and each should take every Nth line to embed. Default: %(default)s")
    args = parser.parse_args()

    print("Load tokenizer model",file=sys.stderr,flush=True)
    bert_tokenizer=transformers.BertTokenizer.from_pretrained(args.bert_model)
    print("Load model",file=sys.stderr,flush=True)
    bert_model=transformers.BertModel.from_pretrained(args.bert_model).eval().cuda()
    print("Done loading",file=sys.stderr,flush=True)
    
    s_dataset=embed_data.SentenceDataset(sys.stdin,bert_tokenizer,args.thisjob,args.jobs)
    print("Done creating dataset",file=sys.stderr, flush=True)
    s_datareader=embed_data.fluid_batch(s_dataset,12000)#DataLoader(sp_dataset,collate_fn=embed_data.collate,batch_size=15)

    with tqdm.tqdm() as pbar, torch.no_grad(), open(args.out,"wb") as fout:
        for batch in s_datareader:
            emb_src=embed_batch(batch,bert_model)
            emb_src=emb_src.cpu()
            bsize=emb_src.shape[0]
            
            pickle.dump((batch["line_idx"],emb_src),fout)
            pbar.update(bsize)
    

