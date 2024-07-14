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
from sentence_transformers import SentenceTransformer
print("all done",file=sys.stderr,flush=True)

def embed_batch(batch,sbert_model):
    emb=sbert_model.encode(batch["txt"])
    return torch.tensor(emb)



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", default=None, help="Input file(jsonlines)")
    parser.add_argument("--bert-tokenizer", default=None, help="BERT tokenizer name or path")
    parser.add_argument("--sbert-model", default=None, help="SBERT model name or path")
    parser.add_argument("--out",default=None,help="File to save batches into")
    parser.add_argument("--thisjob",default=0,type=int,help="Set to the number which this job is out of jobs, zero based. It will embed lines where line id modulo jobs equals thisjob Default: %(default)s")
    parser.add_argument("--jobs",default=1,type=int,help="Set to N if you are splitting the work among N workers and each should take every Nth line to embed. Default: %(default)s")
    args = parser.parse_args()

    print("Load tokenizer model",file=sys.stderr,flush=True)
    bert_tokenizer=transformers.AutoTokenizer.from_pretrained(args.bert_tokenizer)
    print("Load model",file=sys.stderr,flush=True)
    sbert_model=SentenceTransformer(args.sbert_model).eval().cuda()
    print("Done loading",file=sys.stderr,flush=True)
    
    s_dataset=embed_data.SentenceDataset(args.in_file,bert_tokenizer,args.thisjob,args.jobs)
    print("Done creating dataset",file=sys.stderr, flush=True)
    s_datareader=embed_data.fluid_batch(s_dataset,12000)#DataLoader(sp_dataset,collate_fn=embed_data.collate,batch_size=15)

    with tqdm.tqdm() as pbar, torch.no_grad(), open(args.out,"wb") as fout:
        for batch in s_datareader:
            emb_src=embed_batch(batch,sbert_model)
            emb_src=emb_src.cpu()
            bsize=emb_src.shape[0]
            
            pickle.dump((batch["line_idx"],emb_src),fout)
            pbar.update(bsize)
    

