import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import defaultdict
import string
import csv
import json

from tqdm import tqdm
import numpy as np
# from constants import *
import stanza
import spacy_stanza
from sklearn.metrics import f1_score


import torch, nltk
import numpy as np
import benepar

import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import defaultdict
import string
import csv
import json

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification

from constants import *

def tw_topic_eval(sentences, tw_dir, content, cap=None):
    # num matches of distinct words
    words = []
    with open(os.path.join(tw_dir + content + '.txt'), 'r') as rf:
        for line in rf:
            words.append(line.strip().lower())
    num_match = 0
    num_total = 0
    for sent in sentences:
        num_total += 1
        sent_match = 0
        sent = sent.strip().lower().split()
        sent = [tok.strip(string.punctuation) for tok in sent]
        for word in words:
            if word in sent:
                sent_match += 1
                break
        if cap is None:
            num_match += sent_match
        else:
            num_match += min(cap, sent_match)
    return num_match, num_total


def perplexity(sentences, tokenizer, model, device='cuda'):
    # calculate perplexity
    with torch.no_grad():
        ppl = []
        sos_token = tokenizer.decode([0])
        for sentence in tqdm(sentences, total=len(sentences)):
            full_tensor_input = tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
            full_loss = model(full_tensor_input, labels=full_tensor_input)[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return np.mean(ppl)/len(sentences), np.std(ppl)/len(sentences)


def grammaticality(sentences, tokenizer, model, device='cuda'):
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(model(tokenizer.encode(sent, return_tensors='pt').to(device))[0].flatten(), dim=0)[1]
            total_good += good_prob
        return total_good / len(sentences) # avg probability of grammaticality according to model


def distinctness(results):
    d1, d2, d3 = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(lambda: set())
    total_words = defaultdict(lambda: 0)
    for cw, outputs in results.items():
        for o in outputs:
            o = o.replace(EOT_TOKEN, ' ').strip().split(' ')
            o = [str(x) for x in o]
            total_words[cw] += len(o)
            d1[cw].update(o)
            for i in range(len(o) - 1):
                d2[cw].add(o[i] + ' ' + o[i+1])
            for i in range(len(o) - 2):
                d3[cw].add(o[i] + ' ' + o[i+1] + ' ' + o[i+2])
    return_info = []
    avg_d1, avg_d2, avg_d3 = 0, 0, 0
    for cw in total_words.keys():
        return_info.append((cw, 'DISTINCTNESS', len(d1[cw]) / total_words[cw], len(d2[cw]) / total_words[cw], len(d3[cw]) / total_words[cw]))
        avg_d1 += len(d1[cw]) / total_words[cw]
        avg_d2 += len(d2[cw]) / total_words[cw]
        avg_d3 += len(d3[cw]) / total_words[cw]
    avg_d1, avg_d2, avg_d3 = avg_d1 / len(total_words.keys()), avg_d2 / len(total_words.keys()), avg_d3 / len(total_words.keys())
    return return_info, (avg_d1, avg_d2, avg_d3)



def collapse_unary_strip_pos(tree, strip_top=True):
    """Collapse unary chains and strip part of speech tags."""

    def strip_pos(tree):
        if len(tree) == 1 and isinstance(tree[0], str):
            return tree[0]
        else:
            return nltk.tree.Tree(tree.label(), [strip_pos(child) for child in tree])

    collapsed_tree = strip_pos(tree)
    collapsed_tree.collapse_unary(collapsePOS=True, joinChar="::")
    if collapsed_tree.label() in ("TOP", "ROOT", "S1", "VROOT"):
        if strip_top:
            if len(collapsed_tree) == 1:
                collapsed_tree = collapsed_tree[0]
            else:
                collapsed_tree.set_label("")
        elif len(collapsed_tree) == 1:
            collapsed_tree[0].set_label(
                collapsed_tree.label() + "::" + collapsed_tree[0].label())
            collapsed_tree = collapsed_tree[0]
    return collapsed_tree


def _get_labeled_spans(tree, spans_out, start):
    if isinstance(tree, str):
        return start + 1

    assert len(tree) > 1 or isinstance(
        tree[0], str
    ), "Must call collapse_unary_strip_pos first"
    end = start
    for child in tree:
        end = _get_labeled_spans(child, spans_out, end)
    # Spans are returned as closed intervals on both ends
    spans_out.append((start, end - 1, tree.label()))
    return end


def get_labeled_spans(tree):
    """Converts a tree into a list of labeled spans.
    Args:
        tree: an nltk.tree.Tree object
    Returns:
        A list of (span_start, span_end, span_label) tuples. The start and end
        indices indicate the first and last words of the span (a closed
        interval). Unary chains are collapsed, so e.g. a (S (VP ...)) will
        result in a single span labeled "S+VP".
    """
    tree = collapse_unary_strip_pos(tree)
    spans_out = []
    _get_labeled_spans(tree, spans_out, start=0)
    return spans_out

def padded_chart_from_spans(label_vocab, tree):
    # num_words = 64
    num_words = len(tree.leaves())
    spans = get_labeled_spans(tree)
    chart = np.full((num_words, num_words), -100, dtype=int)
    chart = np.tril(chart, -1)
    # Now all invalid entries are filled with -100, and valid entries with 0
    for start, end, label in spans:
        if label in label_vocab:
            chart[start, end] = label_vocab[label]
    return chart

def chart_from_tree(label_vocab, tree):
    spans = get_labeled_spans(tree)
    num_words = len(tree.leaves())
    chart = np.full((num_words, num_words), -100, dtype=int)
    chart = np.tril(chart, -1)
    # Now all invalid entries are filled with -100, and valid entries with 0
    # print(tree)
    for start, end, label in spans:
        # Previously unseen unary chains can occur in the dev/test sets.
        # For now, we ignore them and don't mark the corresponding chart
        # entry as a constituent.
        # print(start, end, label)
        if label in label_vocab:
            chart[start, end] = label_vocab[label]
    return chart, spans


def pad_charts(charts, padding_value=-100):
    """
    Our input text format contains START and END, but the parse charts doesn't.
    NEED TO: update the charts, so that we include these two, and set their span label to 0.

    :param charts:
    :param padding_value:
    :return:
    """
    max_len = 64
    padded_charts = torch.full(
        (len(charts), max_len, max_len),
        padding_value,
    )
    padded_charts = np.tril(padded_charts, -1)
    # print(padded_charts[-2:], padded_charts.shape)
    # print(padded_charts[1])
    for i, chart in enumerate(charts):
        # print(chart, len(chart), len(chart[0]))
        chart_size = len(chart)
        padded_charts[i, 1:chart_size + 1, 1:chart_size + 1] = chart

    # print(padded_charts[-2:], padded_charts.shape)
    return padded_charts


def remove_leaves(tree_):
    # simple_increm = 0
    for s in tree_.subtrees(lambda t: t.height() == 2):
        s[0] = '*'
        s._label = ''
    return tree_



if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--log_file', type=str, default='tree_bnn.json', help='where to load results from')
    parser.add_argument('--tw_dir', type=str, default='./datasets/wordlists/', help='test wordlists')
    parser.add_argument('--batch_size', type=int, default=8, help='max samples at a time')
    parser.add_argument('--cap_per_example', type=int, default=100, help='max matches to count per sentence')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--control', type=str, default='control_tree')

    args = parser.parse_args()

    tw_topic_match_c_total = 0
    category_totals_c = 0 #defaultdict(lambda:0)
    # results = defaultdict(lambda: [])
    # with open(args.log_file, 'r') as rf:
    #     data = list(csv.DictReader(rf))
    #     for line in data:
    #         for line_temp in line:
    #             print(line_temp)
    #             results[line['category']].append(line['generation'])

    results = []
    with open(args.log_file, 'r') as rf:
        data = rf.readlines()

        if args.control == 'control_attribute':
            # print(data[0])

            # data = list(data)
            # data = " ".join(data)
            # data = data.split('END START ')

            # print(data[0].split(':')[2])
            for line in data:
                if line is None:
                    continue
                #################### For The Text Contribute Task
                line = line.split(':')[1]
                # print(line.split(':'))
                sents = line.split('START')

                for sent in sents:
                    # print('***********************************************')
                    # print('The first sentence is : {' + line + '} !:')
                    sent = sent.split()

                    try:
                        while ('END' in sent):
                            sent.remove('END')
                        while ('START' in line):
                            sent.remove('START')
                        sent = " ".join(sent)
                    except:
                        sent = " ".join(sent)
                    # print('After Processing: {' + sent + '} !:')
                    # print('***********************************************')
                    results.append(sent)

            print('#########################################################')
            tw_topic_match_c_total, words_num_total = tw_topic_eval(results[0:100], args.tw_dir, 'name',
                                                                    cap=args.cap_per_example)
            print('Test wordlist matcheson name:', 100 * tw_topic_match_c_total / words_num_total)
            tw_topic_match_c_total, words_num_total = tw_topic_eval(results, args.tw_dir, 'food',
                                                                    cap=args.cap_per_example)
            print('Test wordlist matcheson food:', 100 * tw_topic_match_c_total / words_num_total)
            tw_topic_match_c_total, words_num_total = tw_topic_eval(results, args.tw_dir, 'area',
                                                                    cap=args.cap_per_example)
            print('Test wordlist matcheson area:', 100 * tw_topic_match_c_total / words_num_total)
            tw_topic_match_c_total, words_num_total = tw_topic_eval(results, args.tw_dir, 'price',
                                                                    cap=args.cap_per_example)
            print('Test wordlist matcheson price:', 100 * tw_topic_match_c_total / words_num_total)
            tw_topic_match_c_total, words_num_total = tw_topic_eval(results, args.tw_dir, 'rating',
                                                                    cap=args.cap_per_example)
            print('Test wordlist matcheson rating:', 100 * tw_topic_match_c_total / words_num_total)
            tw_topic_match_c_total, words_num_total = tw_topic_eval(results, args.tw_dir, 'total',
                                                                    cap=args.cap_per_example)
            print('Test wordlist matcheson total:', 100 * tw_topic_match_c_total / words_num_total)
            print('#########################################################')


        elif args.control == 'control_pos':
            data = list(data)
            # data = " ".join(data)
            # data = data.split('END START ')

            stanza.download('en')
            nlp = spacy_stanza.load_pipeline("en", processors={"tokenize": "spacy"})
            print('***********************************************')
            # print('The first sentence is : {' + line + '} !:')
            flag_num = 0
            for line in data:
                flag_num += 1
                if flag_num>5:
                    break
                if line is None:
                    continue

                line = line.split(':')

                pos_line = line[0][2:-1].replace("'", "")
                pos_line = pos_line.split(',')
                pos_lable = [pos_i.replace(" ", "") for pos_i in pos_line]

                match_score = []
                tex_line = line[1].replace("START", "")[3:-3].split('END')
                # sent_lst = [nlp(seq).dep_ for seq in tex_pos]
                for seq in tex_line:
                    text_input = seq.split("',")[-1][1:]
                    if ('UNK' in text_input) or (len(text_input)<15):
                        continue
                    print(text_input)
                    doc = nlp(seq.split("',")[-1])
                    for token in doc:
                        tex_pos =[token.pos_ for token in doc]
                        # print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
                    tex_pos.insert(0, 'START')
                    tex_pos.append('END')
                    len_pos = max(len(pos_lable), len(tex_pos))
                    if len(pos_lable) < len_pos:
                        for i in range(len_pos - len(pos_lable)):
                            pos_lable.append('PAD')
                    elif len(tex_pos) < len_pos:
                        for i in range(len_pos - len(tex_pos)):
                            tex_pos.append('PAD')

                    m_score = len([j for i, j in zip(pos_lable, tex_pos) if i == j]) / len_pos
                    match_score.append(m_score)
                    print(m_score)
            print('maximum match score of POS is: {' + str(np.max(match_score)) + '} !')
            print('average match score of POS is: {' + str(np.mean(match_score)) + '} !')
            print('***********************************************')
            results.append(line)




        # elif args.control == 'control_spans':
        #     data = list(data)
        #
        #     import benepar, spacy
        #     import nltk
        #     benepar.download('benepar_en3')
        #
        #     parser = benepar.Parser("benepar_en3")
        #     tree_vocab = parser._parser.config["label_vocab"]
        #
        #     nlp = spacy.load('en_core_web_md')
        #     if spacy.__version__.startswith('2'):
        #         nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        #     else:
        #         nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        #
        #     print('***********************  Spans  ************************')
        #     # print('The first sentence is : {' + line + '} !:')
        #     flag_num = 0
        #     for line in data:
        #         flag_num += 1
        #         if flag_num > 5:
        #             break
        #         if line is None:
        #             continue
        #
        #         line = line.split(':')
        #
        #         spans_line = line[0][4:-5].replace("'", "").split(',')
        #         spans_label = spans_line[2]
        #         spans_position = [spans_line[0], spans_line[1]]
        #
        #
        #         match_score = []
        #         tex_line = line[1].replace("START", "")[3:-3].split('END')
        #
        #
        #         sent_lst = []
        #         for sent in tex_line:
        #             text_input = sent.split("',")[-1][1:]
        #             if ('UNK' in text_input) or (len(text_input) < 15):
        #                 continue
        #             input_sentence1 = benepar.InputSentence(
        #                 words=text_input[:63],
        #             )
        #             sent_lst.append(input_sentence1)
        #         parse_lst = list(parser.parse_sents(sent_lst))
        #
        #         chart_lst = []
        #         for parse, seq in zip(parse_lst, tex_line):
        #             chart, spans = chart_from_tree(tree_vocab, parse)
        #
        #             print(chart)
        #
        #
        #             # if tex_tag[spans_line[0]]
        #             #     m_score = len([j for i, j in zip(pos_lable, tex_pos) if i == j]) / len_pos
        #         # match_score.append(m_score/len(tex_line))
        #         # print(m_score)
        #     print('maximum match score of POS is: {' + str(np.max(match_score)) + '} !')
        #     print('average match score of POS is: {' + str(np.mean(match_score)) + '} !')
        #     print('***********************  Spans  ************************')




        elif args.control == 'control_tree':
            data = list(data)

            import benepar, spacy
            import nltk
            benepar.download('benepar_en3')

            parser = benepar.Parser("benepar_en3")
            tree_vocab = parser._parser.config["label_vocab"]
            print(tree_vocab)
            nlp = spacy.load('en_core_web_md')
            if spacy.__version__.startswith('2'):
                nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
            else:
                nlp.add_pipe("benepar", config={"model": "benepar_en3"})

            print('***********************  Tree  ************************')
            # print('The first sentence is : {' + line + '} !:')
            flag_num = 0
            for line in data:
                flag_num += 1
                if flag_num > 5:
                    break
                if line is None:
                    continue

                line = line.split(':')

                print(line[0][3:-3])
                spans_line = line[0][3:-2].replace("'", "").split(',')
                spans_label = spans_line[2]
                spans_position = [spans_line[0], spans_line[1]]


                F1_score = []
                tex_line = line[1].replace("START", "")[3:-3].split('END')


                sent_lst = []
                for sent in tex_line:
                    text_input = sent.split("',")[-1][1:]
                    if ('UNK' in text_input) or (len(text_input) < 15):
                        continue
                    input_sentence1 = benepar.InputSentence(
                        words=text_input[:63],
                    )
                    sent_lst.append(input_sentence1)
                parse_lst = list(parser.parse_sents(sent_lst))

                chart_lst = []
                for parse, seq in zip(parse_lst, tex_line):
                    chart, spans = chart_from_tree(tree_vocab, parse)


                    # for (a, b, c) in spans:
                    #     spans_ids = [vocab_dict.get(x, vocab_dict['UNK']) for x in f"{a} {b} {c}".split()]
                    #     spans_lst.append(spans_ids)


                    # doc = nlp(text_input)
                    # sent = list(doc.sents)[0]
                    # print(sent._.parse_string)
                    # chart = padded_chart_from_spans(tree_vocab, sent)

                    # for token in doc:
                    #     tex_tag = [token.pos_ for token in doc]
                    #     # print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
                    # tex_tag.insert(0, 'START')
                    # tex_tag.append('END')

                    # if tex_tag[spans_line[0]]
                    #     m_score = len([j for i, j in zip(pos_lable, tex_pos) if i == j]) / len_pos
                # match_score.append(m_score/len(tex_line))
                # print(m_score)
            print('maximum match score of POS is: {' + str(np.max(match_score)) + '} !')
            print('average match score of POS is: {' + str(np.mean(match_score)) + '} !')
            print('***********************  Tree  ************************')
