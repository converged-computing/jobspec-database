import sys
import tqdm
import math
import numpy as np
import os

import torch
from annoy import AnnoyIndex
import pickle
import re

from biobert import getscibertmodel
from buckets import Buckets


# MAX_CONTEXT = 20

def get_relations_embeddings(s):
    # Get all relations including no relation
    relset = set([(a, b) for a, b, _ in s['relations']])
    norels = []
    if len(s['entities']) <= 1:
        return None, None, None, None, None, None, None, None, None
    for i in range(len(s['entities'])):
        for j in range(len(s['entities'])):
            if i == j or (i, j) in relset:
                continue
            norels.append((i, j, 'No relation'))
    all_rels = [(a, b, rel_dic[re.sub('\d+$', '', c)]) for a, b, c in s['relations'] + norels]
    head = np.stack([s['entities'][x[0]][2] for x in all_rels])
    head_text = [s['entities'][x[0]][0] for x in all_rels]
    head_type = np.stack([ent_dic[s['entities'][x[0]][1]] for x in all_rels])
    tail = np.stack([s['entities'][x[1]][2] for x in all_rels])
    tail_text = [s['entities'][x[1]][0] for x in all_rels]
    tail_type = np.stack([ent_dic[s['entities'][x[1]][1]] for x in all_rels])
    labels = np.stack([x[2] for x in all_rels])
    posdiff = np.asarray([buckets.get_bucket(abs(s['entities'][t][3][0] -
                                                 s['entities'][h][3][0])) \
                          for h, t, _ in all_rels])
    return head_text, head, head_type, tail_text, tail, tail_type, labels, posdiff, all_rels


annoy_file = sys.argv[1]
annoy_dir = sys.argv[2]
vectorizer_file = sys.argv[3]  # not used in case of bert retriever
bert_data = sys.argv[4]
bert_data_target = sys.argv[5]
retriever = sys.argv[7]
num_buckets = sys.argv[8] # =4 for wetlabs data, =8 for materials data
K = int(sys.argv[9])  # nearest neighbors
label_annoys = {}

with open(bert_data, 'rb') as f:
    data_src = pickle.load(f)

with open(bert_data_target, 'rb') as f:
    data_tgt = pickle.load(f)

if retriever == 'tfidf':
    with open(vectorizer_file, 'rb') as f:
        vect = pickle.load(f)
    t = AnnoyIndex(len(vect.get_feature_names()), 'angular')

else:
    # vect file not required
    t = t = AnnoyIndex(768, "angular")  # BERT: num_features = 768
    tokenizer, model = getscibertmodel()
    print('Loaded scibert model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for fil in os.listdir(annoy_dir):
        if not fil.endswith('annoy'):
            continue
        label = fil.split('_')[1].split('.')[0]
        t_l = AnnoyIndex(768, "angular")
        t_l.load(os.path.join(annoy_dir, fil))
        label_annoys[label] = t_l
    with open(os.path.join(annoy_dir, 'id_map.pkl'), 'rb') as f:
        id_map = pickle.load(f)


t.load(annoy_file)


bad_relations = set(['Misc-Link'])

if os.path.exists('relations.txt') and os.path.exists('entity_types.txt'):
    with open('relations.txt', 'r') as f:
        relations = f.read().splitlines()
    with open('entity_types.txt', 'r') as f:
        entity_types = f.read().splitlines()
else:
    relations = set()
    entity_types = set()
    for s in data_src:
        for r in s['relations']:
            print(r)
            relation_without_trailing = re.sub('\d+$', '', r[2])
            relations.add(relation_without_trailing)
        for e in s['entities']:
            entity_types.add(e[1])
    relations = list(relations - bad_relations)
    entity_types = list(entity_types)
    with open('relations.txt', 'w') as f:
        f.write('\n'.join(relations))
    with open('entity_types.txt', 'w') as f:
        f.write('\n'.join(entity_types))

if os.path.exists('buckets.pkl'):
    with open('buckets.pkl', 'rb') as f:
        buckets = pickle.load(f)
else:
    lengths = [s['length'] for s in data_src]
    sorted_lengths = sorted(lengths)
    # take 90th percentile to avoid outlier sentences
    mx = sorted_lengths[int(len(sorted_lengths) * 0.9)]
    mn = 1
    buckets = Buckets(mn, mx, bucket_size=int(num_buckets))
    with open('buckets.pkl', 'wb') as f:
        pickle.dump(buckets, f)

print(relations)
print(len(relations))
relations.append('No relation')
rel_dic = {r: i for i, r in enumerate(relations)}
for r in bad_relations:
    rel_dic[r] = len(relations) - 1  # norelation
print(entity_types)
print(len(entity_types))
ent_dic = {e: i for i, e in enumerate(entity_types)}

dataset = []
cnt = 0


def getbertemb(sent):
    bert_tokens_sentence = tokenizer.encode(sent, add_special_tokens=True)
    with torch.no_grad():
        bert_embeddings = \
            model(torch.tensor([bert_tokens_sentence]).to(device))[0].squeeze(0)
        f_emb_avg = torch.mean(bert_embeddings, axis=0).cpu().numpy()
        return f_emb_avg


def add_relations(nns, src, vector, label_annoys, id_map, istrain):
    seen_labels = set()
    num_relations = len(relations)

    # for nn_id in nns:
    #     seen_labels |= set([x[2] for x in src[nn_id]['relations']])
    for l, index in label_annoys.items():
        # if l not in seen_labels:
            if istrain:
                # first sentence could be the same sentence
                label_nn = index.get_nns_by_vector(vector, 2)[1]
            else:
                label_nn = index.get_nns_by_vector(vector, 1)[0]
            sent_id = id_map[l][label_nn]
            nns.append(sent_id)
    return nns



for s in tqdm.tqdm(data_tgt):
    # print('%d / %d done' % (cnt, len(data_tgt)))
    cnt += 1
    if retriever == 'tfidf':
        vector = vect.transform([s['replaced']]).toarray()[0]
    else:
        vector = getbertemb(s['sentence'])
    if bert_data == bert_data_target:  # on training data, we are bound to find
        # same sentence, so we exclude the top
        # match
        nns = t.get_nns_by_vector(vector, K+1)[1:]  # 1st one will be the same
        # sentence
    else:  # on held out data, we needn't exclude the top match
        nns = t.get_nns_by_vector(vector, K)

    nns = add_relations(nns, data_src,vector, label_annoys, id_map, \
                        bert_data==bert_data_target) # add sentences for rare
                                                     # labels
    # print(len(nns))

    context_head = []
    context_head_text = []
    context_head_type = []
    context_tail = []
    context_tail_text = []
    context_tail_type = []
    context_label = []
    context_posdiffs = []
    num_context = 0
    for nn_id in nns:
        cs = data_src[nn_id]
        head_text, head, head_type, tail_text, tail, tail_type, labels,\
            posdiff,_ = get_relations_embeddings(cs)
        if head is not None:
            num_context += 1
            context_head.append(head)
            context_head_text.extend(head_text)
            context_tail.append(tail)
            context_tail_text.extend(tail_text)
            context_label.append(labels)
            context_head_type.append(head_type)
            context_tail_type.append(tail_type)
            context_posdiffs.append(posdiff)

    if num_context > 0:
        context_head = np.concatenate(context_head, axis=0)
        context_tail = np.concatenate(context_tail, axis=0)
        context_label = np.concatenate(context_label, axis=0)
        context_head_type = np.concatenate(context_head_type, axis=0)
        context_tail_type = np.concatenate(context_tail_type, axis=0)
        context_posdiffs = np.concatenate(context_posdiffs, axis=0)
    else:
        context_head, context_head_type, context_tail, context_tail_type, \
        context_label, context_posdiffs = None, None, None, None, None, None

    query_head_text, query_head, query_head_type, query_tail_text, query_tail, \
    query_tail_type, query_labels, query_posdiff, all_rels = get_relations_embeddings(s)
    if query_head is None:
        continue
    dataset.append({
        'query_sent': s['sentence'],
        'context_sents': [data_src[x]['sentence'] for x in \
                          nns],
        'query_head': query_head,
        'query_head_text': query_head_text,
        'query_head_type': query_head_type,
        'query_tail': query_tail,
        'query_tail_text': query_tail_text,
        'query_tail_type': query_tail_type,
        'query_posdiff': query_posdiff,
        'query_labels': query_labels,
        'context_head': context_head,
        'context_head_text': context_head_text,
        'context_head_type': context_head_type,
        'context_tail': context_tail,
        'context_tail_text': context_tail_text,
        'context_tail_type': context_tail_type,
        'context_labels': context_label,
        'context_posdiff': context_posdiffs,
        'entities' : s['entities'],
        'relations': all_rels
    })

with open(sys.argv[6], 'wb') as f:
    pickle.dump(dataset, f)
