import os
from collections import Counter
from pathlib import Path

import click
import spacy
from sklearn.datasets import fetch_20newsgroups
import tqdm
import re
import multiprocessing as mp

from models.evaluator import Evaluator
import matplotlib.pyplot as plt


def evaluate(args):
    alpha, beta, gamma, delta, num_iter, dataset, num_topics, num_classes, iteration, docs = args
    print(f"Evaluating for iteration {iteration}", flush=True)
    evaluator = Evaluator(alpha, beta, gamma, delta, dataset, num_topics, num_classes, num_iter, iteration)
    log_probability, log_perplexity = evaluator.calculate_corpus_likelihood(docs)
    print(f"Finished evaluating for iteration: {iteration}", flush=True)
    return iteration, log_probability, log_perplexity


def pre_process_docs_before_vocab(nlp, unprocessed_docs):
    docs = []
    patterns_and_replacements = {
        '<EMAIL>' : re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    }
    stopwords = nlp.Defaults.stop_words

    for udoc in tqdm.tqdm(nlp.pipe(unprocessed_docs, batch_size=64), total=len(unprocessed_docs)):
        doc = []
        for token in udoc:
            if token.text.lower() in stopwords:
                continue
            elif token.is_alpha:
                doc.append(token.text.lower())
            elif token.is_punct:
                # since punctuation would be one of the syntactic classes
                doc.append(token.text[0]) # why just text[0]? to handle cases like '!!!' or '...'
            elif token.is_space:
                # all space char including '\n' provides no meaning
                continue
            elif token.is_digit:
                doc.append('<NUM>')
            elif token.is_currency:
                doc.append('<CUR>')
            else:
                for replacement, pattern in patterns_and_replacements.items():
                    if pattern.match(token.text):
                        doc.append(replacement)
                        break
                    else:
                        doc.append('<UNK>')

        docs.append(doc)
    return docs

def build_vocab(docs, rare_words_threshold):
    vocab = Counter()
    for doc in tqdm.tqdm(docs):
        vocab.update(doc)

    # ignore words that are rare
    vocab = Counter({key: count for key, count in vocab.items() if count > rare_words_threshold})
    return vocab

def remove_out_of_vocab_tokens(docs, vocab):
    oov_count = 0
    for doc in docs:
        for ind, token in enumerate(doc):
            if token not in vocab:
                del doc[ind]
                oov_count += 1
    return docs, vocab


def plot_iteration_logs(iteration_logs):
    # Unpack the list of tuples into separate lists for iterations, log probabilities, and log perplexities.
    iterations, log_probs, log_perplexities = zip(*iteration_logs)


    # Create two subplots, one for log probabilities and one for log perplexities.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot log probabilities.
    ax1.plot(iterations, log_probs, marker='o', linestyle='-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log Probability')
    ax1.set_title('Log Probabilities vs. Iterations')

    plt.legend()
    plt.xticks(rotation=45)

    # Plot log perplexities.
    ax2.plot(iterations, log_perplexities, marker='o', linestyle='-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log Perplexity')
    ax2.set_title('Log Perplexities vs. Iterations')

    # Adjust spacing between subplots.
    plt.tight_layout()
    plt.legend()
    plt.xticks(rotation=45)

    return plt

@click.command()

@click.option("--alpha", type=float, default=0.02)
@click.option("--beta", type=float, default=0.02)
@click.option("--gamma", type=float, default=0.02)
@click.option("--delta", type=float, default=0.02)
@click.option("--num_iter", type=int, default=10)
@click.option("--num_topics", type=int, default=5)
@click.option("--num_classes", type=int, default=5)
@click.option("--dataset", type=str)
@click.option("--test_dataset", type=str)
@click.option("--test_dataset_size", type=int)
@click.option("--results_dir", type=str, default="out")
def plot(alpha, beta, gamma, delta, num_iter, num_topics, num_classes, dataset, test_dataset, test_dataset_size, results_dir):
    print(f"Started plot task for alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}, num_iter={num_iter}, num_topics={num_topics}, num_classes={num_classes}, test dataset={test_dataset}, test dataset size={test_dataset_size}", flush=True)
    dir_path = os.path.join(results_dir, f"{alpha}_{beta}_{gamma}_{delta}_{num_topics}_{num_classes}_{num_iter}_{dataset}")
    iterations = []
    for file_path in os.listdir(dir_path):
        # check if current file_path is a file
        if os.path.isdir(os.path.join(dir_path, file_path)):
            # add filename to list
           iterations.append(file_path)


    data = fetch_20newsgroups(subset=test_dataset, remove=('headers', 'footers', 'quotes'))
    nlp = spacy.load("en_core_web_sm")
    unprocessed_docs = data['data'][:test_dataset_size]
    docs = pre_process_docs_before_vocab(nlp, unprocessed_docs)
    vocab = build_vocab(docs, 2)
    docs, vocab = remove_out_of_vocab_tokens(docs, vocab)




    pool = mp.Pool(mp.cpu_count())
    arguments = [(alpha, beta, gamma, delta, num_iter, dataset, num_topics, num_classes, it, docs) for it in iterations]
    results = pool.map(evaluate, arguments)
    # results_async = [pool.apply_async(evaluate, args=(alpha, beta, gamma, delta, num_iter, dataset, num_topics, num_classes, iteration, docs)) for iteration in iterations]
    pool.close()
    pool.join()
    #results = [x.get() for x in results_async]
    plot_iteration_logs(results)
    plots_path = os.path.join(dir_path, f"plots_{test_dataset}.pdf")
    print(f"saving to {plots_path}")
    plt.savefig(plots_path)

if __name__ == "__main__":
    plot()


