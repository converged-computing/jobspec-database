import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
from multiprocessing.pool import ThreadPool
import time

from data_processing import generate_data
from models import load_base_model_and_tokenizer
from plotting import save_roc_curves, save_ll_histograms, save_llr_histograms
from utils import get_roc_metrics, get_precision_recall_metrics

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

# for enrichment
import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def move_base_model_to_gpu():
    print('MOVING BASE (or scoring) MODEL TO GPU...', end='', flush=True)
    start = time.time()

    if mask_model is not None:
        mask_model.cpu()

    if args.openai_model is None:
        base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def move_mask_model_to_gpu():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    if args.openai_model is None:
        base_model.cpu()
    if not args.random_fills:
        mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False, concentration = None):
    """
    Takes raw text and applies random masking for perturbations
    :param text: raw text
    :param span_length:
    :param pct:
    :param ceil_pct:
    :return: masked text
    """
    tokens = text.split(' ')
    if concentration in ["FREQ", "STOP", "NONSTOP"]:
        tokens = text.lower().split()
        if concentration == "FREQ":
            # Selectively masks most frequent non-stop words
            idx = {}
            for i, token in enumerate(tokens):
                if token not in stop_words:
                    if token in idx:
                        idx[token].append(i)
                    else:
                        idx[token] = [i]
            idx = {token:idx[token] for token in idx if len(idx[token])>1}
            sorted_tokens = sorted(idx, key=lambda token:len(idx[token]), reverse=True)
            selected_list = []
            for token in sorted_tokens:
                selected_list.extend(np.random.choice(idx[token], size=len(idx[token])//2, replace=False))
        elif concentration == "STOP":
            # Selectively masks stop words
            selected_list = [i for i, token in enumerate(tokens) if token in stop_words]
        else:
            # Selectively masks non-stop words
            selected_list = [i for i, token in enumerate(tokens) if token not in stop_words]

    elif concentration in ["ALL", "ADJ", "NOUN", "VERB", "PROPN"]:
        # Selectively masks according to part-of-speech
        doc = nlp(text)
        if concentration == "ALL":
            relevant_words = set(
                [word.text for sent in doc.sentences for word in sent.words if word.upos in ["ADJ", "NOUN", "VERB"]])
        else:
            relevant_words = set([word.text for sent in doc.sentences for word in sent.words if word.upos == concentration])

        selected_list = [i for i, token in enumerate(tokens) if token in relevant_words]
        
    elif '+' in concentration:
        assert len(concentration.split('+')) == 2, '2 POS only'
        first, second = concentration.split('+')
        POS = ['ADJ', 'NOUN', 'VERB', 'ADV', 'PROPN']
        assert first in POS and second in POS, 'POS must be one of ["ADJ","NOUN", "VERB", "ADV", "PROPN"]'
        doc = nlp(text)
        relevant_words_first = set([word.text for sent in doc.sentences for word in sent.words if word.upos == first])
        relevant_words_second = set([word.text for sent in doc.sentences for word in sent.words if word.upos == second])
        selected_list = []
        for i in range(len(tokens) - 1):
            if tokens[i] in relevant_words_first and tokens[i+1] in relevant_words_second:
              selected_list.append(i)

        selected_list = [i for i, token in enumerate(tokens[:-1]) if (token in relevant_words_first and tokens[i+1] in relevant_words_second)]
    
        if (first, second) == ("ADV", "VERB") or (second, first) == ("ADV", "VERB"):
          selected_list += [i for i, token in enumerate(tokens[:-1]) if (token in relevant_words_second and tokens[i+1] in relevant_words_first)]

    elif concentration is not None:
        raise ValueError("Concentration not supported")

    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)

    ### ADDED, prevent no selection
    if concentration is not None:
        while len(selected_list) < n_spans: # just making sure that we have other words we want to select too
            x = np.random.randint(0, len(tokens) - span_length)
            if x not in selected_list:
                selected_list.append(x)

    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    count = 0

    while n_masks < n_spans:
        # print(n_masks, n_spans)
        count += 1
        # used to be below
        if concentration is not None and count < 100:
            start = np.random.choice(selected_list)
        else:
            start = np.random.randint(0, len(tokens) - span_length) #this is where we look for figures of speech

        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
        if count > 100:
            print("exceeded tries! Picking randomly")
            # start = np.random.randint(0, len(tokens) - span_length)  # this is where we look for figures of speech

    # print('after while')
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    # print("done", time.time())
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


def replace_masks(texts):
    """
    replace each masked span with a sample from T5 mask_model
    :param texts: masked text
    :return: filled text
    """
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    """
    Take T5 output and convert it back into normal text output
    :param texts: T5 output
    :return: list of extracted fills
    """
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    """
    Takes extracted fills and inserts it back into the masked text
    :param masked_texts:
    :param extracted_fills:
    :return: completely perturbed text
    """
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    """
    Take raw text and perturb with random or T5 model. For single text
    :param texts:
    :param span_length:
    :param pct:
    :param ceil_pct:
    :return: perturbed texts
    """
    if not args.random_fills:
        # print("HERE HERE HERE")

        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct, concentration=args.concentration) for x in texts]
        # print("a")
        raw_fills = replace_masks(masked_texts)
        # print("b")
        extracted_fills = extract_fills(raw_fills)
        # print("c")
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        # print("d")

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts

def perturb_texts(texts, span_length, pct, ceil_pct=False):
    """
    Wrapper function that processes a collection of texts to perturb
    :param texts: a list of strings to perturb
    :param span_length: length of text to perturb
    :param pct: percent perturbation
    :param ceil_pct:
    :return:
    """

    chunk_size = args.chunk_size # basically the batch size for perturbation
    if '11b' in mask_filling_model_name:
        chunk_size //= 2
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def _openai_sample(p):
    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p

    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text



# def get_likelihood(logits, labels):
#     """
#     :param logits:
#     :param labels:
#     :return:
#     """
#     assert logits.shape[0] == 1
#     assert labels.shape[0] == 1
#
#     logits = logits.view(-1, logits.shape[-1])[:-1]
#     labels = labels.view(-1)[1:]
#     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
#     log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
#     return log_likelihood.mean()


def get_ll(text):
    """
    Get the log likelihood of each text under the base_model
    :param text: raw text
    :return: likelihood number
    """
    if args.openai_model:
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts):
    """
    Wrapper function that gets the likelihood of a collection of texts
    :param texts: list of texts
    :return: list of likelihoods
    """
    if not args.openai_model:
        return [get_ll(text) for text in texts]
    else:
        global API_TOKEN_COUNTER

        # use GPT2_TOKENIZER to get total number of tokens
        total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(args.batch_size)
        return pool.map(get_ll, texts)


def get_rank(text, log=False):
    """
    get the average rank of each observed token sorted by model likelihood
    :param text:
    :param log:
    :return:
    """
    assert args.openai_model is None, "get_rank not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


## KEY FUNCTION: runs all the numbers
def get_perturbation_results(span_length=10, n_perturbations=1, n_samples=500):
    """
    Given the provided list of original and sampled data, we run perturbations,
    :param span_length:
    :param n_perturbations:
    :param n_samples:
    :return:
    """
    move_mask_model_to_gpu()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    move_base_model_to_gpu() #return back to normal

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"])
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["original"])
        res["sampled_ll"] = get_ll(res["sampled"])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results

def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    """
    Processes the results after perturbations and llikelihood estimation and gets the stats
    :param results:
    :param criterion:
    :param span_length:
    :param n_perturbations:
    :param n_samples:
    :return:
    """
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "original_crit": criterion_fn(original_text[idx]),
                "sampled": sampled_text[idx],
                "sampled_crit": criterion_fn(sampled_text[idx]),
            })

    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results],
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }

def eval_supervised(data, model):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    real, fake = data['original'], data['sampled']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())

        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }

if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum", help = "dataset type")
    parser.add_argument('--dataset_key', type=str, default="document", help = "key of text in dataset")
    parser.add_argument('--pct_words_masked', type=float, default=0.3,
                        help = "percent words masked. pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))")
    parser.add_argument('--span_length', type=int, default=2, help = "how many words we perturb")
    parser.add_argument('--n_samples', type=int, default=200, help = "how many machine generated texts we use")
    parser.add_argument('--n_perturbation_list', type=str, default="1,10", help = "number of perturbations to try")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1, help = "number of times you run the perturbation on a text")
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium", help = "the model used to generate the text (for testing purposes)")
    parser.add_argument('--scoring_model_name', type=str, default="", help = "The model used to generate DetectGPT score")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large", help = "the model used to perturb the text")
    parser.add_argument('--batch_size', type=int, default=50, help = "how many data points we handle at once")
    parser.add_argument('--chunk_size', type=int, default=20, help = "how many text examples we perturb at once")
    # parser.add_argument('--n_similarity_samples', type=int, default=20)  #NOT USED
    parser.add_argument('--int8', action='store_true') #precision stuff
    parser.add_argument('--half', action='store_true') #precision stuff
    # parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true') # for sampling: enable top k
    parser.add_argument('--top_k', type=int, default=40) # top k parameter
    parser.add_argument('--do_top_p', action='store_true') # for sampling: enable top p
    parser.add_argument('--top_p', type=float, default=0.96) # top p parameter
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None, help = "use this to specify an openai model for calling the API")
    parser.add_argument('--openai_key', type=str, help = "api key for accessing openai models")
    parser.add_argument('--baselines_only', action='store_true', help = "use this if you only want to run baselines")
    parser.add_argument('--skip_baselines', action='store_true', help = "use this if you only want to run DetectGPT")
    parser.add_argument('--buffer_size', type=int, default=1, help = "area between masks")
    parser.add_argument('--mask_top_p', type=float, default=1.0, help = "top-p parameter for the perturbation model")
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0, help = "perturbs text before running; used for ablations")
    parser.add_argument('--pre_perturb_span_length', type=int, default=5, help = "pre-perturbation parameters")
    parser.add_argument('--random_fills', action='store_true', help = "bypass the perturbation model and fill with random text")
    parser.add_argument('--random_fills_tokens', action='store_true', help = "use tokens to fill randomly")
    parser.add_argument('--cache_dir', type=str, default="cache", help = "where we store the models")
    parser.add_argument('--concentration', type=str, default=None, help = "How we pick the words to perturb. None is default")
    parser.add_argument('--prompt', type=str, default=None, help = "Additional Prompt to Model")
    parser.add_argument('--chatgpt', action='store_true', help = "Use ChatGPT for sample generation")
    parser.add_argument('--chatgpt_preset', type=str, default="You are a helpful assistant.", help = "preset system context for ChatGPT")
    parser.add_argument('--chatgpt_temperature', type=float, default=0.0, help = "ChatGPT temperature")
    parser.add_argument('--chatgpt_top_p', type=float, default=0.0, help = "ChatGPT top-p")
    parser.add_argument('--chatgpt_frequency_penalty', type=float, default=0.0, help = "ChatGPT frequency penalty")
    parser.add_argument('--chatgpt_presence-penalty', type=float, default=0.0, help = "ChatGPT presence penalty")

    args = parser.parse_args()
    assert not args.chatgpt or args.scoring_model_name or args.openai_model #chatgpt can't do likelihoods, so it must be scored by a different model
    API_TOKEN_COUNTER = 0
    if args.openai_model is not None or args.chatgpt:
        import openai
        if args.openai_key is None:
            with open("secretkey.txt") as f:
                openai.api_key = f.readline().strip()
        else:
            # assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
            openai.api_key = args.openai_key

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    if args.openai_model is None:
        base_model_name = args.base_model_name.replace('/', '_')
    else:
        base_model_name = "openai-" + args.openai_model.replace('/', '_')
    scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
    SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-" \
                  f"{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-" \
                  f"{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}-{args.span_length}"
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    # write args to file
    with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    # n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

    # generic generative model
    print(args.base_model_name)
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name, args)
    mask_model = None
    # mask filling t5 model
    if not args.baselines_only and not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs,
                                                                        **half_kwargs, cache_dir=cache_dir)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512,
                                                                   cache_dir=cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions,
                                                                cache_dir=cache_dir)
    if args.dataset in ['english', 'german']:
        preproc_tokenizer = mask_tokenizer

    move_base_model_to_gpu()

    print(f'Loading dataset {args.dataset}...')
    if args.openai_model is not None or args.chatgpt:
        print("USING OPENAI. IGNORING BASE MODEL")
        if args.chatgpt:
            print("IGNORING BASE MODEL AND USING CHATGPT TO GENERATE")
        # if chatgpt flag is enabled, we IGNORE the base model during generation
        data = generate_data(args.dataset, args.dataset_key, preproc_tokenizer, base_model, base_tokenizer, args, openai)
    else:
        data = generate_data(args.dataset, args.dataset_key, preproc_tokenizer, base_model, base_tokenizer, args, None)

    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

    if args.scoring_model_name:
        print(f'Loading SCORING model {args.scoring_model_name}...')
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.scoring_model_name, args)
        move_base_model_to_gpu()  # Load again because we've deleted/replaced the old model

    # write the data to a json file in the save folder
    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
        print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
        json.dump(data, f)

    if not args.skip_baselines:
        baseline_outputs = [run_baseline_threshold_experiment(get_ll, "likelihood", n_samples=n_samples)]
        if args.openai_model is None:
            rank_criterion = lambda text: -get_rank(text, log=False)
            baseline_outputs.append(run_baseline_threshold_experiment(rank_criterion, "rank", n_samples=n_samples))
            logrank_criterion = lambda text: -get_rank(text, log=True)
            baseline_outputs.append(
                run_baseline_threshold_experiment(logrank_criterion, "log_rank", n_samples=n_samples))
            entropy_criterion = lambda text: get_entropy(text)
            baseline_outputs.append(
                run_baseline_threshold_experiment(entropy_criterion, "entropy", n_samples=n_samples))

        baseline_outputs.append(eval_supervised(data, model='roberta-base-openai-detector'))
        baseline_outputs.append(eval_supervised(data, model='roberta-large-openai-detector'))

    outputs = []

    #### MAIN INFERENCE ####
    if not args.baselines_only:
        # run perturbation experiments
        for n_perturbations in n_perturbation_list:
            perturbation_results = get_perturbation_results(args.span_length, n_perturbations,
                                                            n_samples)
            for perturbation_mode in ['d', 'z']:
                output = run_perturbation_experiment(
                    perturbation_results, perturbation_mode, span_length=args.span_length,
                    n_perturbations=n_perturbations, n_samples=n_samples)
                outputs.append(output)
                with open(os.path.join(SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"),
                          "w") as f:
                    json.dump(output, f)

    if not args.skip_baselines:
        # write likelihood threshold results to a file
        with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs[0], f)

        if args.openai_model is None:
            # write rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[1], f)

            # write log rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[2], f)

            # write entropy threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[3], f)

        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-2], f)

        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-1], f)

        outputs += baseline_outputs

    save_roc_curves(outputs, SAVE_FOLDER, args)
    save_ll_histograms(outputs, SAVE_FOLDER)
    save_llr_histograms(outputs, SAVE_FOLDER)

    # move results folder from tmp_results/ to results/, making sure necessary directories exist
    new_folder = SAVE_FOLDER.replace("tmp_results", "results")
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)



    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")
