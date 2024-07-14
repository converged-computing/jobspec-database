#!/usr/bin/env python3

import json
import torch
import sklearn
import argparse
import numpy as np
from math import ceil

def retrieve_most_similar(vectors1, vectors2):
    """
    Code modified from: https://github.com/TurkuNLP/Deep_Learning_in_LangTech_course/blob/master/laser.ipynb
    Given two vectors, return the most similar for every item in the first vector
    """
    #assert len(vectors1)==len(vectors2)
    all_dist = sklearn.metrics.pairwise_distances(vectors1, vectors2)
    nearest = all_dist.argmin(axis=-1)
    return nearest

def eval_tatoeba_retrieval(nearest, positive_dict):
    """
    The tatoeba aligned corpus has many to many mapping
    i.e. multiple source sentences are parallel to multiple target sentences
    """
    correct = 0
    for i, selected_index in enumerate(nearest): # nearest: <class 'numpy.ndarray'>
        # selected_index <class 'numpy.int64'>
        #if i==0:
        #    print("selected index type", type(selected_index)) #<class 'numpy.int64'>
        #    print("positive dict list item type", type(positive_dict[str(i)][0])) #int
        if selected_index in positive_dict[str(i)]:
            correct += 1
        #    print("correct", selected_index, positive_dict[str(i)])
    print("Correct predictions", correct, "Total predictions", len(nearest), flush=True)
    return correct/len(nearest)

def eval_tatoeba_retrieval_print_predictions(nearest, positive_dict, src_sentences, trg_sentences, n=None):
    """
    The tatoeba aligned corpus has many to many mapping
    i.e. multiple source sentences are parallel to multiple target sentences
    print n predictions, if n not given, print all
    """
    if not n: # print all predictions
        n = len(nearest)
    prnt = True
    correct = 0
    for i, selected_index in enumerate(nearest): # nearest: <class 'numpy.ndarray'>
        if selected_index in positive_dict[str(i)]:
            correct += 1
            if prnt:
                print("CORRECT:", src_sentences[i], trg_sentences[selected_index])
        else:
            if prnt:
                print("INCORRECT:", src_sentences[i], trg_sentences[selected_index])
        n = n-1
        if n==0:
            prnt = False
    print("Correct predictions", correct, "Total predictions", len(nearest))
    return correct/len(nearest)

def load_and_evaluate(model, pos_dict_path, src_path, trg_path, device):
    with open(pos_dict_path, "r") as f:
        pos_dict = json.load(f)
    with open(src_path, "r") as f:
        src_sentences = f.readlines()
    src_sentences = [line.strip() for line in src_sentences]
    with open(trg_path, "r") as f:
        trg_sentences = f.readlines()
    trg_sentences = [line.strip() for line in trg_sentences]

    acc1 = evaluate(model, pos_dict, src_sentences, trg_sentences, device)
    return acc1

def evaluate(model, pos_dict, src_sentences, trg_sentences, device):
    model.eval()
    model.to(device)
    
    embeddings = embed_by_batch(model, src_sentences, trg_sentences, batch_size=64)
    selected_indices = retrieve_most_similar(embeddings["src"], embeddings["trg"])
    acc1 = eval_tatoeba_retrieval(selected_indices, pos_dict)
    #print("Accuracy@1", acc1)
    return acc1

def embed_by_batch(model, src_sentences, trg_sentences, batch_size=128):
    # embedding sentences by batch
    embeddings = None
    for i in range(ceil(len(src_sentences)/batch_size)):
        embedded_segment = model({"src": src_sentences[i*batch_size:(i+1)*batch_size], "trg": trg_sentences[i*batch_size:(i+1)*batch_size]})
        if not isinstance(embeddings, dict): #torch.Tensor):
            embeddings = {k: v.cpu().detach().numpy() for k,v in embedded_segment.items()}
        else:
            #embeddings = {k: torch.cat((v, embedded_segment[k].to("cpu"))) for k, v in embeddings.items()}
            embeddings = {k: np.append(v, embedded_segment[k].cpu().detach().numpy(), axis=0) for k, v in embeddings.items()}
    assert len(embeddings["src"])==len(src_sentences)
    return embeddings

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--src-sentences", help="File containing source sentences", required=True)
    argparser.add_argument("--trg-sentences", help="File containing target sentences", required=True)
    argparser.add_argument("--ckpt-path", help="Path to the AlignLangNet checkpoint to be evaluated", required=True)
    argparser.add_argument("--positive-dict", help="Path to the json positive dictionary", required=True)
    args = argparser.parse_args()
    return args

def main():
    """
    First argument, path to checkpoint for evaluation
    """
    args = parse_args()

    #model = AlignLangNet(args.ckpt_path+"/src", trg_model_path=args.ckpt_path+"/trg")
    model = torch.load(args.ckpt_path, map_location=torch.device("cpu"))
    model.eval()

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # The dictionary of positives
    with open(args.positive_dict, "r") as f:
        pos_dict = json.load(f)

    # src sentences in text form
    with open(args.src_sentences, "r") as f:
        src_sentences = f.readlines()
    src_sentences = [line.strip() for line in src_sentences]

    # trg sentences in text form
    with open(args.trg_sentences, "r") as f:
        trg_sentences = f.readlines()
    trg_sentences = [line.strip() for line in trg_sentences]

    model.device = device
    model.to(model.device)

    # embed source and target sentences
    embeddings = embed_by_batch(model, src_sentences, trg_sentences, batch_size=64)

    # find the nearest neighbor for all the source sentences
    selected_indices = retrieve_most_similar(embeddings["src"], embeddings["trg"])
    acc1 = eval_tatoeba_retrieval_print_predictions(selected_indices, pos_dict, src_sentences, trg_sentences, n=None)
    #acc1 = eval_tatoeba_retrieval(selected_indices, pos_dict)
    print("Accuracy@1", acc1)

if __name__ == "__main__":
    from train import AlignLangNet
    exit(main())
