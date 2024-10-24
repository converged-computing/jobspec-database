import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from utils.data_utils import RawDataset, MorphFeaturizer, create_gender_embeddings
from utils.data_utils import Vocabulary, SeqVocabulary
from utils.metrics import accuracy
import json
import random
import re
import numpy as np
import argparse
from gensim.models import KeyedVectors
from seq2seq import Seq2Seq
from greedy_decoder import BatchSampler
from beam_decoder import BeamSampler
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Vectorizer:
    """Vectorizer Class"""
    def __init__(self, src_vocab_char, trg_vocab_char,
                 src_vocab_word, src_labels_vocab,
                 trg_labels_vocab, trg_gender_vocab):
        """
        Args:
            - src_vocab_char (SeqVocabulary): source vocab on the char level
            - trg_vocab_char (SeqVocabulary): target vocab on the char level
            - src_vocab_word (SeqVocabulary): source vocab on the word level
            - src_labels_vocab (Vocabulary): source labels vocab on the sentence level
            - trg_labels_vocab (Vocabulary): target labels vocab on the sentence level
            - trg_gender_vocab (Vocabulary): target gender vocab on the sentence level
        """

        self.src_vocab_char = src_vocab_char
        self.trg_vocab_char = trg_vocab_char
        self.src_vocab_word = src_vocab_word
        self.src_labels_vocab = src_labels_vocab
        self.trg_labels_vocab = trg_labels_vocab
        self.trg_gender_vocab = trg_gender_vocab

    @classmethod
    def create_vectorizer(cls, data_examples):
        """Class method which builds the vectorizer
        vocab

        Args:
            - data_examples: list of InputExample

        Returns:
            - Vectorizer object
        """

        src_vocab_char = SeqVocabulary()
        src_vocab_word = SeqVocabulary()
        trg_vocab_char = SeqVocabulary()
        src_labels_vocab = Vocabulary()
        trg_labels_vocab = Vocabulary()
        trg_gender_vocab = Vocabulary()

        for ex in data_examples:
            src = ex.src
            trg = ex.trg
            src_label = ex.src_label
            trg_label = ex.trg_label
            trg_gender = ex.trg_gender

            # splitting by a regex to maintain the space
            src = re.split(r'(\s+)', src)
            trg = re.split(r'(\s+)', trg)

            for word in src:
                src_vocab_word.add_token(word)
                src_vocab_char.add_many(list(word))

            for word in trg:
                trg_vocab_char.add_many(list(word))

            src_labels_vocab.add_token(src_label)
            trg_labels_vocab.add_token(trg_label)
            trg_gender_vocab.add_token(trg_gender)

        return cls(src_vocab_char, trg_vocab_char,
                   src_vocab_word, src_labels_vocab,
                   trg_labels_vocab, trg_gender_vocab)

    def get_src_indices(self, seq):
        """Converts the source sequence chars
        to indices

        Args:
          - seq (str): The source sequence

        Returns:
          - char_level_indices (list): <s> + List of chars to index mapping + </s>
          - word_level_indices (list): <s> + List of words to index mapping + </s>
        """

        char_level_indices = [self.src_vocab_char.sos_idx]
        word_level_indices = [self.src_vocab_word.sos_idx]
        seq = re.split(r'(\s+)', seq)

        for word in seq:
            for c in word:
                char_level_indices.append(self.src_vocab_char.lookup_token(c))
                word_level_indices.append(self.src_vocab_word.lookup_token(word))

        word_level_indices.append(self.src_vocab_word.eos_idx)
        char_level_indices.append(self.src_vocab_char.eos_idx)

        assert len(word_level_indices) == len(char_level_indices)

        return char_level_indices, word_level_indices

    def get_trg_indices(self, seq):
        """Converts the target sequence chars
        to indices

        Args:
          - seq (str): The target sequence

        Returns:
          - trg_x_indices (list): <s> + List of chars to index mapping
          - trg_y_indices (list): List of chars to index mapping + </s>
        """
        indices = [self.trg_vocab_char.lookup_token(t) for t in seq]

        trg_x_indices = [self.trg_vocab_char.sos_idx] + indices
        trg_y_indices = indices + [self.trg_vocab_char.eos_idx]
        return trg_x_indices, trg_y_indices

    def vectorize(self, src, trg, src_label, trg_label, trg_gender):
        """
        Args:
          - src (str): The source sequence
          - trg (str): The target sequence
          - src_label (str): The source sequence label
          - trg_label (str): The target sequence label
          - trg_label (str): The target sequence gender

        Returns:
          - vectorized_src_char (tensor): <s> + vectorized source seq on the char level + </s>
          - vectorized_src_word (tensor): <s> + vectorized source seq on the word level + </s>
          - vectorized_trg_x (tensor): <s> + vectorized target seq on the char level
          - vectorized_trg_y (tensor): vectorized target seq on the char level + </s>
          - vectorized_src_label (tensor): vectorized source label
          - vectorized_trg_label (tensor): vectorized target label
          - vectorized_trg_gender (tensor): vectorized target gender
        """

        vectorized_src_char, vectorized_src_word = self.get_src_indices(src)
        vectorized_trg_x, vectorized_trg_y = self.get_trg_indices(trg)
        vectorized_src_label = self.src_labels_vocab.lookup_token(src_label)
        vectorized_trg_label = self.trg_labels_vocab.lookup_token(trg_label)
        vectorized_trg_gender = self.trg_gender_vocab.lookup_token(trg_gender)

        return {'src_char': torch.tensor(vectorized_src_char, dtype=torch.long),
                'src_word': torch.tensor(vectorized_src_word, dtype=torch.long),
                'trg_x': torch.tensor(vectorized_trg_x, dtype=torch.long),
                'trg_y': torch.tensor(vectorized_trg_y, dtype=torch.long),
                'src_label': torch.tensor(vectorized_src_label, dtype=torch.long),
                'trg_label': torch.tensor(vectorized_trg_label, dtype=torch.long),
                'trg_gender': torch.tensor(vectorized_trg_gender, dtype=torch.long)
               }

    def to_serializable(self):
        return {'src_vocab_char': self.src_vocab_char.to_serializable(),
                'trg_vocab_char': self.trg_vocab_char.to_serializable(),
                'src_vocab_word': self.src_vocab_word.to_serializable(),
                'src_labels_vocab': self.src_labels_vocab.to_serializable(),
                'trg_labels_vocab': self.trg_labels_vocab.to_serializable(),
                'trg_gender_vocab': self.trg_gender_vocab.to_serializable()
               }

    @classmethod
    def from_serializable(cls, contents):
        src_vocab_char = SeqVocabulary.from_serializable(contents['src_vocab_char'])
        src_vocab_word = SeqVocabulary.from_serializable(contents['src_vocab_word'])
        trg_vocab_char = SeqVocabulary.from_serializable(contents['trg_vocab_char'])
        src_labels_vocab = Vocabulary.from_serializable(contents['src_labels_vocab'])
        trg_labels_vocab = Vocabulary.from_serializable(contents['trg_labels_vocab'])
        trg_gender_vocab = Vocabulary.from_serializable(contents['trg_gender_vocab'])

        return cls(src_vocab_char, trg_vocab_char,
                   src_vocab_word, src_labels_vocab,
                   trg_labels_vocab, trg_gender_vocab)


class MT_Dataset(Dataset):
    """MT Dataset as a PyTorch dataset"""
    def __init__(self, raw_dataset, vectorizer):
        """
        Args:
            - raw_dataset (RawDataset): raw dataset object
            - vectorizer (Vectorizer): vectorizer object
        """
        self.vectorizer = vectorizer
        self.train_examples = raw_dataset.train_examples
        self.dev_examples = raw_dataset.dev_examples
        self.test_examples = raw_dataset.test_examples
        self.lookup_split = {'train': self.train_examples,
                             'dev': self.dev_examples,
                             'test': self.test_examples}
        self.set_split('train')

    def get_vectorizer(self):
        return self.vectorizer

    @classmethod
    def load_data_and_create_vectorizer(cls, data_dir):
        raw_dataset = RawDataset(data_dir)
        # Note: we always create the vectorized based on the train examples
        vectorizer = Vectorizer.create_vectorizer(raw_dataset.train_examples)
        return cls(raw_dataset, vectorizer)

    @classmethod
    def load_data_and_load_vectorizer(cls, data_dir, vec_path):
        raw_dataset = RawDataset(data_dir)
        vectorizer = cls.load_vectorizer(vec_path)
        return cls(raw_dataset, vectorizer)

    @staticmethod
    def load_vectorizer(vec_path):
        with open(vec_path) as f:
            return Vectorizer.from_serializable(json.load(f))

    def save_vectorizer(self, vec_path):
        with open(vec_path, 'w') as f:
            return json.dump(self.vectorizer.to_serializable(), f, ensure_ascii=False)

    def set_split(self, split):
        self.split = split
        self.split_examples = self.lookup_split[self.split]
        return self.split_examples

    def __getitem__(self, index):
        example = self.split_examples[index]
        src, trg = example.src, example.trg
        src_label, trg_label = example.src_label, example.trg_label
        trg_gender = example.trg_gender
        vectorized = self.vectorizer.vectorize(src, trg, src_label, trg_label, trg_gender)
        return vectorized

    def __len__(self):
        return len(self.split_examples)

class Collator:
    def __init__(self, char_src_pad_idx, char_trg_pad_idx,
                word_src_pad_idx):
        """
        Args:
            - char_src_pad_idx: source vocab padding index on the char level
            - char_trg_pad_idx: target vocab padding index on the char level
            - word_src_pad_idx: source vocab padding index on the word level
        """
        self.char_src_pad_idx = char_src_pad_idx
        self.word_src_pad_idx = word_src_pad_idx
        self.char_trg_pad_idx = char_trg_pad_idx

    def __call__(self, batch):
        # Sorting the batch by src seqs length in descending order
        sorted_batch = sorted(batch, key=lambda x: x['src_char'].shape[0], reverse=True)

        src_char_seqs = [x['src_char'] for x in sorted_batch]
        src_word_seqs = [x['src_word'] for x in sorted_batch]
        src_seqs_labels = [x['src_label'] for x in sorted_batch]

        assert len(src_word_seqs) == len(src_char_seqs)

        trg_x_seqs = [x['trg_x'] for x in sorted_batch]
        trg_y_seqs = [x['trg_y'] for x in sorted_batch]
        trg_seqs_labels = [x['trg_label'] for x in sorted_batch]
        trg_seqs_genders = [x['trg_gender'] for x in sorted_batch]
        lengths = [len(seq) for seq in src_char_seqs]

        padded_src_char_seqs = pad_sequence(src_char_seqs, batch_first=True, padding_value=self.char_src_pad_idx)
        padded_src_word_seqs = pad_sequence(src_word_seqs, batch_first=True, padding_value=self.word_src_pad_idx)

        padded_trg_x_seqs = pad_sequence(trg_x_seqs, batch_first=True, padding_value=self.char_trg_pad_idx)
        padded_trg_y_seqs = pad_sequence(trg_y_seqs, batch_first=True, padding_value=self.char_trg_pad_idx)

        lengths = torch.tensor(lengths, dtype=torch.long)
        src_seqs_labels = torch.tensor(src_seqs_labels, dtype=torch.long)
        trg_seqs_labels = torch.tensor(trg_seqs_labels, dtype=torch.long)
        trg_seqs_genders = torch.tensor(trg_seqs_genders, dtype=torch.long)

        return {'src_char': padded_src_char_seqs,
                'src_word': padded_src_word_seqs,
                'trg_x': padded_trg_x_seqs,
                'trg_y': padded_trg_y_seqs,
                'src_lengths': lengths,
                'src_label': src_seqs_labels,
                'trg_label': trg_seqs_labels,
                'trg_gender': trg_seqs_genders
                }

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def train(model, dataloader, optimizer, criterion, device='cpu',
          teacher_forcing_prob=1, clip_grad=1.0):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        src_char = batch['src_char']
        src_word = batch['src_word']
        trg_x = batch['trg_x']
        trg_y = batch['trg_y']
        src_lengths = batch['src_lengths']
        trg_gender = batch['trg_gender']

        preds, attention_scores = model(char_src_seqs=src_char,
                                        word_src_seqs=src_word,
                                        src_seqs_lengths=src_lengths,
                                        trg_seqs=trg_x,
                                        trg_gender=trg_gender,
                                        teacher_forcing_prob=teacher_forcing_prob)

        # CrossEntropysLoss accepts matrices always! 
        # the preds must be of size (N, C) where C is the number 
        # of classes and N is the number of samples. 
        # The ground truth must be a Vector of size C!
        preds = preds.contiguous().view(-1, preds.shape[-1])
        trg_y = trg_y.view(-1)

        loss = criterion(preds, trg_y)
        epoch_loss += loss.item()
        # Backprop
        loss.backward()
        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        # Optimizer step
        optimizer.step()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device='cpu', teacher_forcing_prob=0):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            src_char = batch['src_char']
            src_word = batch['src_word']
            trg_x = batch['trg_x']
            trg_y = batch['trg_y']
            src_lengths = batch['src_lengths']
            trg_gender = batch['trg_gender']

            preds, attention_scores = model(char_src_seqs=src_char,
                                            word_src_seqs=src_word,
                                            src_seqs_lengths=src_lengths,
                                            trg_seqs=trg_x,
                                            trg_gender=trg_gender,
                                            teacher_forcing_prob=teacher_forcing_prob)

            # CrossEntropyLoss accepts matrices always! 
            # the preds must be of size (N, C) where C is the number 
            # of classes and N is the number of samples. 
            # The ground truth must be a Vector of size C!
            preds = preds.contiguous().view(-1, preds.shape[-1])
            trg_y = trg_y.view(-1)

            loss = criterion(preds, trg_y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def inference(sampler, beam_sampler, dataloader, args):
    output_inf_file = open(args.preds_dir + '.inf', mode='w', encoding='utf8')
    output_beam_g = open(args.preds_dir + '.beam_greedy', mode='w', encoding='utf8')
    output_beam = open(args.preds_dir + '.beam', mode='w', encoding='utf8')

    greedy_stats = {}
    beam_stats = {}
    greedy_accuracy = 0
    beam_accuracy = 0

    for batch in dataloader:
        sampler.set_batch(batch)
        src = sampler.get_src_sentence(0)
        trg = sampler.get_trg_sentence(0)
        src_label = sampler.get_src_label(0)
        trg_label = sampler.get_trg_label(0)

        if args.embed_trg_gender:
            trg_gender = sampler.get_trg_gender(0)
        else:
            trg_gender = None

        translated = sampler.greedy_decode(sentence=src, trg_gender=trg_gender)
        beam_trans_10 = beam_sampler.beam_decode(sentence=src, trg_gender=trg_gender, topk=1, beam_width=10, max_len=512)
        beam_trans_1 = beam_sampler.beam_decode(sentence=src, trg_gender=trg_gender, topk=1, beam_width=1, max_len=512)

        greedy_accuracy += accuracy(trg=trg, pred=translated)
        beam_accuracy += accuracy(trg=trg, pred=beam_trans_10)

        correct = 'CORRECT!' if trg == translated else 'INCORRECT!'
        different_g = 'SAME!' if translated == beam_trans_1 else 'DIFF!'
        different_10 = 'SAME!' if translated == beam_trans_10 else 'DIFF!'

        if beam_trans_1 == trg:
            greedy_stats[(src_label, trg_label, 'correct')] = 1 + greedy_stats.get((src_label, trg_label, 'correct'), 0)
        else:
            greedy_stats[(src_label, trg_label, 'incorrect')] = 1 + greedy_stats.get((src_label, trg_label, 'incorrect'), 0)
        if beam_trans_10 == trg:
            beam_stats[(src_label, trg_label, 'correct')] = 1 + beam_stats.get((src_label, trg_label, 'correct'), 0)
        else:
            beam_stats[(src_label, trg_label, 'incorrect')] = 1 + beam_stats.get((src_label, trg_label, 'incorrect'), 0)

        output_inf_file.write(translated)
        output_inf_file.write('\n')
        output_beam_g.write(beam_trans_1)
        output_beam_g.write('\n')
        output_beam.write(beam_trans_10)
        output_beam.write('\n')

        logger.info(f'src:\t\t\t{src}')
        logger.info(f'trg:\t\t\t{trg}')
        logger.info(f'greedy:\t\t\t{translated}')
        logger.info(f'beam:\t\t\t{beam_trans_10}')
        logger.info(f'src label:\t\t{src_label}')
        logger.info(f'trg label:\t\t{trg_label}')
        if trg_gender:
            logger.info(f'trg gender:\t\t{trg_gender}')
        logger.info(f'res:\t\t\t{correct}')
        logger.info(f'beam==greedy?:\t\t{different_10}')
        logger.info('\n\n')

    greedy_accuracy /= len(dataloader)
    beam_accuracy /= len(dataloader)

    output_inf_file.close()
    output_beam_g.close()
    output_beam.close()

    logger.info('*******STATS*******')
    assert sum([greedy_stats[x] for x in greedy_stats]) == sum([beam_stats[x] for x in beam_stats])
    total_examples = sum([greedy_stats[x] for x in greedy_stats])
    logger.info(f'TOTAL EXAMPLES: {total_examples}')
    logger.info('\n')

    correct_greedy = {(x[0], x[1]): greedy_stats[x] for x in greedy_stats if x[2] == 'correct'}
    incorrect_greedy = {(x[0], x[1]): greedy_stats[x] for x in greedy_stats if x[2] == 'incorrect'}
    total_correct_greedy = sum([v for k,v in correct_greedy.items()])
    total_incorrect_greedy = sum([v for k, v in incorrect_greedy.items()])

    logger.info('Results using greedy decoding:')
    for x in correct_greedy:
        logger.info(f'{x[0]}->{x[1]}')
        logger.info(f'\tCorrect: {correct_greedy.get(x, 0)}\tIncorrect: {incorrect_greedy.get(x, 0)}')
    logger.info(f'--------------------------------')
    logger.info(f'Total Correct: {total_correct_greedy}\tTotal Incorrect: {total_incorrect_greedy}')
    logger.info(f'Accuracy:\t{greedy_accuracy}')

    logger.info('\n')

    correct_beam = {(x[0], x[1]): beam_stats[x] for x in beam_stats if x[2] == 'correct'}
    incorrect_beam = {(x[0], x[1]): beam_stats[x] for x in beam_stats if x[2] == 'incorrect'}
    total_correct_beam = sum([v for k, v in correct_beam.items()])
    total_incorrect_beam = sum([v for k, v in incorrect_beam.items()])

    logger.info('Results using beam decoding:')
    for x in correct_beam:
        logger.info(f'{x[0]}->{x[1]}')
        logger.info(f'\tCorrect: {correct_beam.get(x, 0)}\tIncorrect: {incorrect_beam.get(x, 0)}')

    logger.info(f'--------------------------------')
    logger.info(f'Total Correct: {total_correct_beam}\tTotal Incorrect: {total_incorrect_beam}')
    logger.info(f'Accuracy:\t{beam_accuracy}')

def get_morph_features(args, data, word_vocab):
    morph_featurizer = MorphFeaturizer(args.analyzer_db_path)
    if args.reload_files:
        morph_featurizer.load_morph_features(args.morph_features_path)
    else:
        morph_featurizer.featurize_sentences(data)
        if args.cache_files:
            morph_featurizer.save_morph_features(args.morph_features_path)

    morph_embeddings = morph_featurizer.create_morph_embeddings(word_vocab)
    return morph_embeddings

def load_fasttext_embeddings(args, vocab):
    set_seed(args.seed, args.use_cuda)
    fasttext_wv = KeyedVectors.load(args.fasttext_embeddings_kv_path, mmap='r')
    pretrained_embeddings = torch.zeros((len(vocab), fasttext_wv.vector_size), dtype=torch.float32)
    oov = 0
    unks = list()
    for word, index in vocab.token_to_idx.items():
        if word in fasttext_wv.vocab:
            pretrained_embeddings[index] = torch.tensor(fasttext_wv[word], dtype=torch.float32)
        else:
            oov += 1
            unks.append(word)

    print(f'# Vocab not in the Embeddings: {oov}', flush=True)
    print(unks, flush=True)
    return pretrained_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the src and trg files."
    )
    parser.add_argument(
        "--vectorizer_path",
        default=None,
        type=str,
        help="The path of the saved vectorizer"
    )
    parser.add_argument(
        "--cache_files",
        action="store_true",
        help="Whether to cache the vocab and the vectorizer objects or not"
    )
    parser.add_argument(
        "--reload_files",
        action="store_true",
        help="Whether to reload the vocab and the vectorizer objects from a cached file"
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--embedding_dim",
        default=32,
        type=int,
        help="The embedding dimensions of the model"
    )
    parser.add_argument(
        "--trg_gender_embedding_dim",
        default=0,
        type=int,
        help="The embedding dimensions of the target gender"
    )
    parser.add_argument(
        "--hidd_dim",
        default=64,
        type=int,
        help="The hidden dimensions of the model"
    )
    parser.add_argument(
        "--num_layers",
        default=1,
        type=int,
        help="The numbers of layers of the model"
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout rate."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Optimizer weight decay"
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-4,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--clip_grad",
        default=1.0,
        type=float,
        help="Gradient clipping norm."
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU"
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Whether to use the gpu or not."
    )
    parser.add_argument(
        "--seed",
        default=21,
        type=int,
        help="Random seed."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default=None,
        help="The directory of the model."
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training or not."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval or not."
    )
    parser.add_argument(
        "--visualize_loss",
        action="store_true",
        help="Whether to visualize the loss during training and evaluation."
    )
    parser.add_argument(
        "--do_inference",
        action="store_true",
        help="Whether to do inference or not."
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="dev",
        help="The dataset to do inference on."
    )
    parser.add_argument(
        "--use_morph_features",
        action="store_true",
        help="Whether to use morphological features or not."
    )
    parser.add_argument(
        "--use_fasttext_embeddings",
        action="store_true",
        help="Whether to use fasttext embeddings or not."
    )
    parser.add_argument(
        "--fasttext_embeddings_kv_path",
        type=str,
        default=None,
        help="The path to the pretrained fasttext embeddings keyedvectors."
    )
    parser.add_argument(
        "--embed_trg_gender",
        action="store_true",
        help="Whether to embed the target gender or not."
    )
    parser.add_argument(
        "--one_hot_trg_gender",
        action="store_true",
        help="Whether to embed the target gender in a zero hot fashion."
    )
    parser.add_argument(
        "--analyzer_db_path",
        type=str,
        default=None,
        help="Path to the anaylzer database."
    )
    parser.add_argument(
        "--morph_features_path",
        type=str,
        default=None,
        help="The path of the saved morphological features."
    )
    parser.add_argument(
        "--preds_dir",
        type=str,
        default=None,
        help="The directory to write the translations to"
    )

    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')
    set_seed(args.seed, args.use_cuda)

    if args.reload_files:
        dataset = MT_Dataset.load_data_and_load_vectorizer(args.data_dir, args.vectorizer_path)
    else:
        dataset = MT_Dataset.load_data_and_create_vectorizer(args.data_dir)

    vectorizer = dataset.get_vectorizer()

    if args.cache_files:
        dataset.save_vectorizer(args.vectorizer_path)

    if args.use_morph_features:
        # we create morph features on the src side of the
        # training data
        logger.info(f'Loading Gender Morph Features...')
        train_src_data = [t.src for t in dataset.train_examples]
        morph_embeddings = get_morph_features(args, train_src_data, vectorizer.src_vocab_word)
    else:
        morph_embeddings = None

    if args.use_fasttext_embeddings:
        logger.info(f'Loading FastText Embeddings...')
        fasttext_embeddings = load_fasttext_embeddings(args, vectorizer.src_vocab_word)
    else:
        fasttext_embeddings = None

    if args.one_hot_trg_gender:
        logger.info(f'Creating one hot gender embeddings...')
        gender_embeddings = create_gender_embeddings(vectorizer.trg_gender_vocab)
        print(vectorizer.trg_gender_vocab.token_to_idx, flush=True)
        print(gender_embeddings, flush=True)
    else:
        gender_embeddings = None

    ENCODER_INPUT_DIM = len(vectorizer.src_vocab_char)
    DECODER_INPUT_DIM = len(vectorizer.trg_vocab_char)
    DECODER_OUTPUT_DIM = len(vectorizer.trg_vocab_char)
    DECODER_TRG_GEN_INPUT_DIM = len(vectorizer.trg_gender_vocab)
    CHAR_SRC_PAD_INDEX = vectorizer.src_vocab_char.pad_idx
    WORD_SRC_PAD_INDEX = vectorizer.src_vocab_word.pad_idx
    TRG_PAD_INDEX = vectorizer.trg_vocab_char.pad_idx
    TRG_SOS_INDEX = vectorizer.trg_vocab_char.sos_idx


    model = Seq2Seq(encoder_input_dim=ENCODER_INPUT_DIM,
                    encoder_embed_dim=args.embedding_dim,
                    encoder_hidd_dim=args.hidd_dim,
                    encoder_num_layers=args.num_layers,
                    decoder_input_dim=DECODER_INPUT_DIM,
                    decoder_embed_dim=args.embedding_dim,
                    decoder_hidd_dim=args.hidd_dim,
                    decoder_num_layers=args.num_layers,
                    decoder_output_dim=DECODER_OUTPUT_DIM,
                    morph_embeddings=morph_embeddings,
                    fasttext_embeddings=fasttext_embeddings,
                    embed_trg_gender=args.embed_trg_gender,
                    gender_input_dim=DECODER_TRG_GEN_INPUT_DIM,
                    gender_embed_dim=args.trg_gender_embedding_dim,
                    gender_embeddings=gender_embeddings,
                    char_src_padding_idx=CHAR_SRC_PAD_INDEX,
                    word_src_padding_idx=WORD_SRC_PAD_INDEX,
                    trg_padding_idx=TRG_PAD_INDEX,
                    trg_sos_idx=TRG_SOS_INDEX,
                    dropout=args.dropout)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_INDEX)
    # lr scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=2, factor=0.5)

    collator = Collator(char_src_pad_idx=CHAR_SRC_PAD_INDEX,
                        char_trg_pad_idx=TRG_PAD_INDEX,
                        word_src_pad_idx=WORD_SRC_PAD_INDEX)

    model = model.to(device)

    if args.do_train:
        logger.info('Training...')
        train_losses = []
        dev_losses = []
        best_loss = 1e10
        teacher_forcing_prob = 0.3
        clip_grad = args.clip_grad
        set_seed(args.seed, args.use_cuda)

        for epoch in range(args.num_train_epochs):
            dataset.set_split('train')
            dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator, drop_last=True)

            train_loss = train(model, dataloader, optimizer, criterion, device, teacher_forcing_prob=teacher_forcing_prob, clip_grad=clip_grad)
            train_losses.append(train_loss)

            dataset.set_split('dev')
            dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator, drop_last=True)
            dev_loss = evaluate(model, dataloader, criterion, device, teacher_forcing_prob=0)
            dev_losses.append(dev_loss)

            #save best model
            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(model.state_dict(), args.model_path)

            scheduler.step(dev_loss)
            logger.info(f'Epoch: {(epoch + 1)}')
            logger.info(f'\tTrain Loss: {train_loss:.4f}   |   Dev Loss: {dev_loss:.4f}')

    if args.do_train and args.visualize_loss:
        plt.plot(range(1, 1 + args.num_train_epochs), np.asarray(train_losses), 'b-', color='blue', label='Training')
        plt.plot(range(1, 1 + args.num_train_epochs), np.asarray(dev_losses), 'b-', color='orange', label='Evaluation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.model_path + '.loss.png')

    if args.do_eval:
        logger.info('Evaluation')
        set_seed(args.seed, args.use_cuda)
        dev_losses = []
        for epoch in range(args.num_train_epochs):
            dataset.set_split('dev')
            dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collator)
            dev_loss = evaluate(model, dataloader, criterion, device, teacher_forcing_prob=0)
            dev_losses.append(dev_loss)
            logger.info(f'Dev Loss: {dev_loss:.4f}')

    if args.do_inference:
        logger.info('Inference')
        set_seed(args.seed, args.use_cuda)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        model = model.to(device)
        dataset.set_split(args.inference_mode)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
        sampler = BatchSampler(model=model,
                               src_vocab_char=vectorizer.src_vocab_char,
                               src_vocab_word=vectorizer.src_vocab_word,
                               trg_vocab_char=vectorizer.trg_vocab_char,
                               src_labels_vocab=vectorizer.src_labels_vocab,
                               trg_labels_vocab=vectorizer.trg_labels_vocab,
                               trg_gender_vocab=vectorizer.trg_gender_vocab)

        beam_sampler =  BeamSampler(model=model,
                                    src_vocab_char=vectorizer.src_vocab_char,
                                    src_vocab_word=vectorizer.src_vocab_word,
                                    trg_vocab_char=vectorizer.trg_vocab_char,
                                    src_labels_vocab=vectorizer.src_labels_vocab,
                                    trg_labels_vocab=vectorizer.trg_labels_vocab,
                                    trg_gender_vocab=vectorizer.trg_gender_vocab)

        inference(sampler, beam_sampler, dataloader, args)


if __name__ == "__main__":
    main()

