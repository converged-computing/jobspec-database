import click
from sklearn.datasets import fetch_20newsgroups
import spacy
import tqdm
from collections import Counter
import pandas as pd
import re
import os
import pathlib


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


def save_data(docs, words, size, dset):
    dir_name = os.path.join(".", f"{dset}_{size}")
    pathlib.Path(dir_name).mkdir()
    with open(os.path.join(dir_name, "documents.txt"), "x") as documents_file:
        documents = ''
        for doc in docs:
            doc_str = [str(w) for w in doc]
            documents += f"{' '.join(doc_str)}\n"
        documents_file.writelines(documents)

    with open(os.path.join(dir_name, 'vocab.txt'), "x") as vocab_file:
        vocabulary = ' '.join(words)
        vocab_file.write(vocabulary)

@click.command()
@click.option("--size", type=int, default=5100)
@click.option("--dset", type=str, default='news')
def preprocess_data(size, dset=None):
    if dset=='news':
        data = fetch_20newsgroups(subset="train", remove=('headers', 'footers', 'quotes'),
                                  categories=['alt.atheism','sci.med','sci.space','sci.electronics',
                                              'talk.politics.guns','comp.graphics','rec.motorcycles',
                                              'comp.graphics','soc.religion.christian','comp.os.ms-windows.misc'])
        unprocessed_docs = data['data'][:size]
    elif dset=='nips':
        papers = pd.read_csv("datasets/nips/papers.csv")
        # remove columns
        papers = papers.drop(columns=['id', 'title', 'abstract', 
                              'event_type', 'pdf_name', 'year'], axis=1)
        # sample only 100 papers
        papers = papers.sample(size)
        unprocessed_docs = papers.paper_text.values.tolist()
        

    nlp = spacy.load("en_core_web_sm")
    docs = pre_process_docs_before_vocab(nlp, unprocessed_docs)
    processed_docs = []

    vocab = build_vocab(docs, rare_words_threshold=2)
    #docs, vocab = remove_out_of_vocab_tokens(docs, vocab)
    words = list(vocab.keys())
    oov_idx = words.index('<UNK>')
    for doc in tqdm.tqdm(docs):
        doc_words = []
        for word in doc:
            if word in words:
                word_idx = words.index(word)
                doc_words.append(word_idx)
            else:
                doc_words.append(oov_idx)

        processed_docs.append(doc_words)

    save_data(processed_docs, words, size, dset)


if __name__ == "__main__":
    preprocess_data()



