import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from pathlib import Path
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge
from news_tls import utils, data, datewise, clust, summarizers
from pprint import pprint
from argparse import ArgumentParser
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from itertools import count
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from collections import Counter, defaultdict
from news_tls.rl import Critic
from news_tls.environment import Environment

import os
import sys
import json
import torch
import pickle
import pathlib
import numpy as np



import pathlib
import json
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer


# utils functions
def extract_keywords(ref_timeline, metric = 'tfidf'):
    timeline = ref_timeline.date_to_summaries
    text = [item["text"][0].lower() for item in timeline]
    if metric == "tfidf":
        vectorizer = TfidfVectorizer(stop_words = 'english')
        tfidf = vectorizer.fit_transform(text)
        vocab = vectorizer.get_feature_names() # vocabulary
        weight = tfidf.toarray() # weight[i][j] -- vocab[j]'s tf-idf value in text[i]
        keywords = []
        # take words with n-largest tfidf value
        for i in range(len(text)):
            idx = heapq.nlargest(10, range(len(weight[i])), weight[i].take)
            for id in idx:
                keywords.append(vocab[id])
        return set(keywords)

def concatenate(timeline):
    ret = ""
    for item in timeline:
        ret += item["text"]
    return ret




# Summariser
nfirst = 5
model_name = "google/pegasus-multi_news"
#model_name = 'google/pegasus-cnn_dailymail'

# RL
top_k = 5
test_size = 10
epsilon = 0.01

#weights = [1, 0, 0, 0]

def extract_keywords(ref_timeline, metric='tfidf'):
    text = [' '.join(summary) for summary in ref_timeline.values()]
    keywords = []
    if metric == "tfidf":
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform(text)
        vocab = vectorizer.get_feature_names_out()  # vocabulary
        weight = tfidf.toarray()  # weight[i][j] -- vocab[j]'s tf-idf value in text[i]
        # take words with n-largest tfidf value
        for i in range(len(text)):
            idx = heapq.nlargest(3, range(len(weight[i])), weight[i].take)
            for id in idx:
                keywords.append(vocab[id])
    return set(keywords)


def train(args, dataset, env, trunc_timelines=False, time_span_extension=0, dataset_name=None):
    average_rewards = []
    metric = 'align_date_content_costs_many_to_one'
    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    n_topics = len(dataset.collections)
    result_path = Path(args.output)
    for i, collection in enumerate(dataset.collections):

        ref_timelines = [TilseTimeline(tl.date_to_summaries) for tl in collection.timelines]
        topic = collection.name
        n_ref = len(ref_timelines)

        if trunc_timelines:
            ref_timelines = data.truncate_timelines(ref_timelines, collection)

        for j, ref_timeline in enumerate(ref_timelines):
            print(f'topic {i+1}/{n_topics}: {topic}, ref timeline {j+1}/{n_ref}')

            keywords = extract_keywords(ref_timeline.dates_to_summaries)
            env.update(keywords, i)
            print('keywords extracted...')

            ref_dates = sorted(ref_timeline.dates_to_summaries)

            start, end = data.get_input_time_span(ref_dates, time_span_extension)

            collection.start = start
            collection.end = end

            #utils.plot_date_stats(collection, ref_dates)

            l = len(ref_dates)
            k = data.get_average_summary_length(ref_timeline)

            tokenizer = PegasusTokenizer.from_pretrained(args.model)
            state_size = tokenizer.vocab_size
            actor = PegasusForConditionalGeneration.from_pretrained(args.model).to(args.device)
            critic = Critic(state_size).to(args.device)
            critic_loss_fct = torch.nn.MSELoss()
            optimizerA = torch.optim.Adam(actor.lm_head.parameters(), lr=args.lr)
            optimizerC = torch.optim.Adam(critic.parameters(), lr=args.lr)

            if args.method == 'rl-datewise':
                resources = Path(args.resources)
                models_path = resources / 'supervised_date_ranker.{}.pkl'.format(dataset_name)
                # load regression models for date ranking
                key_to_model = None
                date_ranker = datewise.MentionCountDateRanker()
                sent_collector = datewise.PM_Mean_SentenceCollector(clip_sents=5, pub_end=2)
                summarizer = summarizers.PegasusSummariser(model=actor,
                                                           critic=critic,
                                                           tokenizer=tokenizer,
                                                           critic_loss_fct=critic_loss_fct,
                                                           optimizerA=optimizerA,
                                                           optimizerC=optimizerC,
                                                           device=args.device)
                system = datewise.DatewiseRLGenerator(
                    date_ranker=date_ranker,
                    summarizer=summarizer,
                    sent_collector=sent_collector,
                    key_to_model=key_to_model)
            elif args.method == 'rl-clust':
                cluster_ranker = clust.ClusterDateMentionCountRanker()
                clusterer = clust.TemporalMarkovClusterer()
                summarizer = summarizers.PegasusSummariser(model=actor,
                                                           critic=critic,
                                                           tokenizer=tokenizer,
                                                           critic_loss_fct=critic_loss_fct,
                                                           optimizerA=optimizerA,
                                                           optimizerC=optimizerC,
                                                           device=args.device)
                system = clust.ClusteringRLGenerator(
                    cluster_ranker=cluster_ranker,
                    clusterer=clusterer,
                    summarizer=summarizer,
                    clip_sents=5,
                    unique_dates=True)
            else:
                raise ValueError(f'Method not found: {args.method}')

            topic_rewards = []
            for epoch in range(args.epochs):
                print('epoch {}/{}...'.format(epoch+1, args.epochs))
                rewards = system.rl(args, collection, env, max_dates=l, max_summary_sents=k, ref_tl=ref_timeline)
                topic_rewards.append(rewards)

                print('generating timelines...')
                pred_timeline_ = system.predict(
                    collection,
                    max_dates=l,
                    max_summary_sents=k,
                    ref_tl=ref_timeline)

                reference = [[str(date), ref_timeline[date]] for date in ref_timeline]
                with open("{}/{}/{}_{}.json".format(result_path, epoch+1, topic, j), "w") as fp:
                    json.dump(pred_timeline_.to_dict(), fp)
                with open("{}/{}_{}_ref.json".format(result_path, topic, j), "w") as fp:
                    json.dump(reference, fp)

            average_rewards.append(topic_rewards)

            # print('*** PREDICTED ***')
            # utils.print_tl(pred_timeline_)

            print('timeline done')
            #pred_timeline = TilseTimeline(pred_timeline_.date_to_summaries)
            #sys_len = len(pred_timeline.get_dates())
            #ground_truth = TilseGroundTruth([ref_timeline])

            #rouge_scores = get_scores(metric, pred_timeline, ground_truth, evaluator)
            #date_scores = evaluate_dates(pred_timeline, ground_truth)

            #print('sys-len:', sys_len, 'gold-len:', l, 'gold-k:', k)
            continue

            print('Alignment-based ROUGE:')
            pprint(rouge_scores)
            print('Date selection:')
            pprint(date_scores)
            print('-' * 100)
            results.append((rouge_scores, date_scores, pred_timeline_.to_dict()))

    print('average rewards among topics: ', average_rewards)

    average_rewards = np.mean(average_rewards, axis=0)
    print('average rewards across topics from epoch 1 to end ', average_rewards)
    with open("{}/rewards.json".format(result_path), "w") as fp:
        json.dump(average_rewards, fp)
        #save models after every epoch
        #model_dir = Path(args.model) / '{}.pt'.format(epoch)
        #torch.save({'epoch': epoch,
        #            'actor_state_dict': actor.state_dict(),
        #            'critic_state_dict': critic.state_dict(),
        #            'optimizerA_state_dict': optimizerA.state_dict(),
        #            'optimizerC_state_dict': optimizerC.state_dict(),
        #            'rewards': rewards}, model_dir)
        #print('epoch:{}/{} models are saved in {}.'.format(epoch, args.epochs, model_dir))

    #avg_results = get_average_results(results)
    #print('Average results:')
    #pprint(avg_results)
    #output = {
    #    'average': avg_results,
    #    'results': results,
    #}
    #utils.write_json(output, result_path)
    print("-------------------------------summarisation done-------------------------------")


def main(args):
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    dataset = data.Dataset(dataset_path)
    dataset_name = dataset_path.name

    # Configuration
    result_path = Path(args.output)
    if args.preference:
        prefdata_path = Path(args.preference)
    device = args.device
    valid_on = True
    valid_set = ['libya']

#    if args.model != 'google/pegasus-multi_news':
#        checkpoint = torch.load(args.model)
#        actor.load_state_dict(checkpoint['actor_state_dict'])
#        critic.load_state_dict(checkpoint['critic_state_dict'])
#        optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
#        optimizerC.load_state_dict(checkpoint['optimizerC_state_dict'])
#        epoch = checkpoint['epoch']
#        rewards = checkpoint['rewards']

    env = Environment(args, device)

    if dataset_name == 'entities':
        train(args=args,
              dataset=dataset,
              env=env,
              trunc_timelines=True,
              time_span_extension=7,
              dataset_name=dataset_name)
    else:
        train(args=args,
              dataset=dataset,
              env=env,
              trunc_timelines=False,
              time_span_extension=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset dir')
    parser.add_argument('--output', required=True, help='result dir')
    parser.add_argument('--method', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--alpha', required=False, default=15.0)
    parser.add_argument('--lr', required=False, default=1e-3)
    parser.add_argument('--epochs', required=False, default=15, type=int)
    parser.add_argument('--gamma', required=False, default=0.99)
    parser.add_argument('--model', required=False, default='google/pegasus-multi_news', help='model dir/name')
    parser.add_argument('--preference', required=False, default=None, help='preference data dir')
    parser.add_argument('--resources', default=None,
                        help='model resources for tested method')
    args = parser.parse_args()
    main(args)
