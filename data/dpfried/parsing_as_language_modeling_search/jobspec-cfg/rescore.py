from __future__ import absolute_import, division, print_function
from utils import PTBModel, MediumConfig, _build_vocab

import time
import pickle
import sys
import tensorflow as tf
import numpy as np
import copy

from score import score_all_trees, score_single_tree, score_trees_separate_batching

from rnng_output_to_nbest import parse_rnng_file

import reader
import random

# flags = tf.flags
logging = tf.logging

def rerank(train_traversed_path,
           candidate_path,
           model_path,
           output_path,
           sent_limit=None,
           likelihood_file=None,
           best_file=None,
           scramble=False,
           reverse=False,
           scramble_keep_top_k=None,
           batching='default',
           num_steps=MediumConfig.num_steps):
    # candidate path: file in ix ||| candidate score ||| parse format
    config = pickle.load(open(model_path + '.config', 'rb'))
    config.batch_size = 10
    config.num_steps = num_steps

    with open(candidate_path) as f:
        candidates_by_sent = list(parse_rnng_file(f))

    if sent_limit is not None:
        candidates_by_sent = candidates_by_sent[:sent_limit]

    if scramble:
        print("scrambling")
        if scramble_keep_top_k:
            print("keeping top k", scramble_keep_top_k)
        for i in range(len(candidates_by_sent)):
            if not scramble_keep_top_k:
                random.shuffle(candidates_by_sent[i])
            else:
                cp = candidates_by_sent[i][scramble_keep_top_k:]
                random.shuffle(cp)
                candidates_by_sent[i][scramble_keep_top_k:] = cp

    if reverse:
        print("reversing")
        for i in range(len(candidates_by_sent)):
            candidates_by_sent[i] = candidates_by_sent[i][::-1]

    print("loading parses")
    parses_by_sent = [
        ["(S1 %s)" % parse.replace("*HASH*", "#") for (ix, score, parse) in candidates]
        for candidates in candidates_by_sent
    ]
    print("loading vocab")
    word_to_id = _build_vocab(train_traversed_path)

    print("running rescore")
    with tf.Graph().as_default(), tf.Session() as session:
        if batching=='none':
            config = copy.copy(config)
            config.batch_size = 1
            config.num_steps = 1
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=False, config=config)

        saver = tf.train.Saver()
        saver.restore(session, model_path)

        print("loading data")
        if batching == 'separate' or batching == 'none':
            xs_by_sent = [[reader.ptb_to_word_ids(parse, word_to_id) for parse in parses]
                          for parses in parses_by_sent]
            id_to_word = {v:k for k, v in word_to_id.items()}
            with open('/tmp/linearized.candidates', 'w') as f:
                for sent_ix, xs in enumerate(xs_by_sent):
                    for cand_ix, x in enumerate(xs):
                        s = ' '.join([id_to_word[id_] for id_ in x])
                        f.write("%d\t%d\t%s\n" % (sent_ix, cand_ix, s))

            losses_by_sent = []
            for i, xs in enumerate(xs_by_sent):
                sys.stderr.write("\r%d / %d" % (i, len(xs_by_sent)))
                if batching == 'separate':
                    losses_by_sent.append(score_trees_separate_batching(session, m, xs, tf.no_op(), eos_index=word_to_id['<eos>']))
                    # if i == 0:
                    #     print(losses_by_sent)
                elif batching == 'none':
                    losses_by_sent.append([score_single_tree(session, m, x) for x in xs])
                    # if i == 0:
                    #     print(losses_by_sent)
                else:
                    raise ValueError("bad batching %s" % batching)

        else:
            test_nbest_data = reader.ptb_list_to_word_ids(parses_by_sent,
                                                        word_to_id,
                                                        remove_duplicates=False,
                                                        sent_limit=None)

            assert(len(test_nbest_data['trees']) == len(parses_by_sent))
            assert(all(len(x) == len(y) for x, y in zip(parses_by_sent, test_nbest_data['trees'])))
            losses_by_sent = score_all_trees(session, m, test_nbest_data, tf.no_op(), word_to_id['<eos>'], likelihood_file=likelihood_file, output_nbest=best_file)

    assert(len(losses_by_sent) == len(candidates_by_sent))
    with open(output_path, 'w') as f:
        for sent_ix, (candidates, losses) in enumerate(zip(candidates_by_sent, losses_by_sent)):
            assert(len(candidates) == len(losses))
            for (ix, old_score, parse), loss in zip(candidates, losses):
                f.write("%s ||| %s ||| %s\n" % (ix, -loss, parse))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_traversed_path", help="e.g. wsj/train_02-21.txt.traversed")
    parser.add_argument("model_path")
    parser.add_argument("candidate_path")
    parser.add_argument("output_path")
    parser.add_argument("--sent_limit", type=int)
    parser.add_argument("--likelihood_file", help="additionally output scores to this file")
    parser.add_argument("--best_file", help="output best parses to this file")
    parser.add_argument("--scramble", action='store_true')
    parser.add_argument("--scramble_keep_top_k", type=int)
    parser.add_argument("--reverse", action='store_true')
    parser.add_argument("--batching", choices=['default', 'none', 'separate'])
    parser.add_argument("--num_steps", type=int, default=MediumConfig.num_steps)
    args = parser.parse_args()

    rerank(args.train_traversed_path,
           args.candidate_path,
           args.model_path,
           args.output_path,
           args.sent_limit,
           args.likelihood_file,
           args.best_file,
           scramble=args.scramble,
           reverse=args.reverse,
           scramble_keep_top_k=args.scramble_keep_top_k,
           batching=args.batching,
           num_steps=MediumConfig.num_steps)
