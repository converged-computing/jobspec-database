
import json
import argparse
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import spacy
import en_core_web_sm
from common_utils import *

nlp = en_core_web_sm.load()
#nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

'''
Scorer for argument extraction on ACE & KAIROS.
For the RAMS dataset, the official scorer is used.

Outputs:
Head F1
Coref F1
'''


def clean_span(ex, span):
    tokens = ex['tokens']
    if tokens[span[0]].lower() in {'the', 'an', 'a'}:
        if span[0] != span[1]:
            return (span[0]+1, span[1])
    return span


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-file', type=str,
                        default='checkpoints/gen-all-ACE-freq-pred/predictions.jsonl')
    parser.add_argument('--test-file', type=str,
                        default='data/ace/zs-freq-10/test.oneie.json')
    parser.add_argument('--coref-file', type=str)
    parser.add_argument('--head-only', action='store_true')
    parser.add_argument('--coref', action='store_true')
    parser.add_argument('--dataset', type=str, default='KAIROS',
                        choices=['ACE', 'KAIROS', 'AIDA'])
    args = parser.parse_args()

    ontology_dict = load_ontology(dataset=args.dataset)

    if args.dataset == 'KAIROS' and args.coref and not args.coref_file:
        print('coreference file needed for the KAIROS dataset.')
        raise ValueError
    if args.dataset == 'AIDA' and args.coref:
        raise NotImplementedError

    examples = {}
    doc2ex = defaultdict(list)  # a document contains multiple events
    with open(args.gen_file, 'r') as f:
        # this solution relies on keeping the exact same order
        for lidx, line in enumerate(f):
            pred = json.loads(line.strip())
            examples[lidx] = {
                'predicted': pred['predicted'],
                'gold': pred['gold'],
                'doc_id': pred['doc_key']
            }
            doc2ex[pred['doc_key']].append(lidx)

    with open(args.test_file, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            if 'sent_id' in doc.keys():
                doc_id = doc['sent_id']
                # print('evaluating on sentence level')
            else:
                doc_id = doc['doc_id']
                # print('evaluating on document level')
            for idx, eid in enumerate(doc2ex[doc_id]):
                examples[eid]['tokens'] = doc['tokens']
                examples[eid]['event'] = doc['event_mentions'][idx]
                examples[eid]['entity_mentions'] = doc['entity_mentions']

    # span to canonical entity_id mapping for each doc
    coref_mapping = defaultdict(dict)
    if args.coref:
        if args.dataset == 'KAIROS' and args.coref_file:
            with open(args.coref_file, 'r') as f, open(args.test_file, 'r') as test_reader:
                for line, test_line in zip(f, test_reader):
                    coref_ex = json.loads(line)
                    ex = json.loads(test_line)
                    doc_id = coref_ex['doc_key']

                    for cluster, name in zip(coref_ex['clusters'], coref_ex['informative_mentions']):
                        canonical = cluster[0]
                        for ent_id in cluster:
                            ent_span = get_entity_span(ex, ent_id)
                            ent_span = (ent_span[0], ent_span[1]-1)
                            coref_mapping[doc_id][ent_span] = canonical
                    # this does not include singleton clusters
        else:
            # for the ACE dataset
            with open(args.test_file) as f:
                for line in f:
                    doc = json.loads(line.strip())
                    doc_id = doc['sent_id']
                    for entity in doc['entity_mentions']:
                        mention_id = entity['id']
                        ent_id = '-'.join(mention_id.split('-')[:-1])
                        # all indexes are inclusive
                        coref_mapping[doc_id][(
                            entity['start'], entity['end']-1)] = ent_id

    # import ipdb; ipdb.set_trace()
    pred_arg_num = 0
    gold_arg_num = 0
    arg_idn_num = 0
    arg_class_num = 0

    arg_idn_coref_num = 0
    arg_class_coref_num = 0

    for ex in tqdm(examples.values()):
        context_words = ex['tokens']
        doc_id = ex['doc_id']
        doc = None
        if args.head_only:
            doc = nlp(' '.join(context_words))

        # get template
        evt_type = ex['event']['event_type']

        if evt_type not in ontology_dict:
            continue
        # extract argument text
        predicted_args = extract_args_from_template(
            ontology_dict, ex['event']['event_type'], ex['predicted'])
        # import ipdb; ipdb.set_trace()
        # get trigger
        # extract argument span
        trigger_start = ex['event']['trigger']['start']
        trigger_end = ex['event']['trigger']['end']

        predicted_set = set()
        for argname in predicted_args:
            # this argument span is inclusive, FIXME: this might be problematic
            for entity in predicted_args[argname]:
                arg_span = find_arg_span(entity, context_words,
                                         trigger_start, trigger_end, head_only=args.head_only, doc=doc)

                if arg_span:  # if None means hullucination

                    predicted_set.add(
                        (arg_span[0], arg_span[1], evt_type, argname))

                else:
                    new_entity = []
                    for w in entity:
                        if w == 'and' and len(new_entity) > 0:
                            arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end,
                                                     head_only=args.head_only, doc=doc)
                            if arg_span:
                                predicted_set.add(
                                    (arg_span[0], arg_span[1], evt_type, argname))
                            new_entity = []
                        else:
                            new_entity.append(w)

                    if len(new_entity) > 0:  # last entity
                        arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end,
                                                 head_only=args.head_only, doc=doc)
                        if arg_span:
                            predicted_set.add(
                                (arg_span[0], arg_span[1], evt_type, argname))

        # import ipdb; ipdb.set_trace()
        # get gold spans
        gold_set = set()
        # set of canonical mention ids, singleton mentions will not be here
        gold_canonical_set = set()
        for arg in ex['event']['arguments']:
            argname = arg['role']
            entity_id = arg['entity_id']
            span = get_entity_span(ex, entity_id)
            span = (span[0], span[1]-1)
            span = clean_span(ex, span)
            # clean up span by removing `a` `the`
            if args.head_only and span[0] != span[1]:
                span = find_head(span[0], span[1], doc=doc)

            gold_set.add((span[0], span[1], evt_type, argname))
            if args.coref:
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_canonical_set.add((canonical_id, evt_type, argname))

        pred_arg_num += len(predicted_set)
        gold_arg_num += len(gold_set)
        # check matches
        for pred_arg in predicted_set:
            arg_start, arg_end, event_type, role = pred_arg
            gold_idn = {item for item in gold_set
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == event_type}
            if gold_idn:
                arg_idn_num += 1
                gold_class = {item for item in gold_idn if item[-1] == role}
                if gold_class:
                    arg_class_num += 1
            elif args.coref:  # check coref matches
                arg_start, arg_end, event_type, role = pred_arg
                span = (arg_start, arg_end)
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_idn_coref = {item for item in gold_canonical_set
                                      if item[0] == canonical_id and item[1] == event_type}
                    if gold_idn_coref:
                        arg_idn_coref_num += 1
                        gold_class_coref = {item for item in gold_idn_coref
                                            if item[2] == role}
                        if gold_class_coref:
                            arg_class_coref_num += 1

    if args.head_only:
        print('Evaluation by matching head words only....')

    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    print('Role identification: Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: Precision:: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

    if args.coref:
        role_id_prec, role_id_rec, role_id_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_idn_num + arg_idn_coref_num)
        role_prec, role_rec, role_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_class_num + arg_class_coref_num)

        print('Coref Role identification: Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(
            role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
        print('Coref Role: Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.2f}'.format(
            role_prec * 100.0, role_rec * 100.0, role_f * 100.0))
