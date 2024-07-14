import random
from pathlib import Path

import pandas as pd
import typer
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
import evaluate
import numpy as np
from torchinfo import summary
import nltk


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


ROOT_FOLDER = Path(__file__).parent if Path(__file__).parent.name == 'content' else Path(__file__).parent.parent.parent  # detect colab run


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)


nltk.download("punkt", quiet=True)


app = typer.Typer(add_completion=False)


def _load_dataset(tokenizer, generation_type='explanation_only', max_length=512, predicted_labels=None):
    """generation_type: one of 'explanation_only', 'explanation_use_label', 'explanation_use_prompt_label', 'explanation_use_flan_prompt_label', 'label_and_explanation'"""
    dataset = load_dataset('esnli')
    cl = dataset['train'].features['label']
    dataset = dataset.rename_column('label', 'raw_label')
    dataset = dataset['test']
    if predicted_labels is not None:
        dataset = dataset.add_column('predicted_label', predicted_labels)

    def _join_with_sep(list_a, list_b):
        # TODO: add new special token instead of using tokenizer.unk_token
        # special_token = tokenizer.additional_special_tokens[0]
        special_token = ':'
        return [f'{h}{special_token} {l}' for h, l in zip(list_a, list_b)]

    def _form_target(list_a, list_b, list_c, list_d, include_label: bool, sep_token = ';', special_token = ':'):
        """Constructs the target sequence for the model.
        In case of training use first explanation, for validation use all explanations connacted with sep.
        """
        if include_label:
            if len(max(list_c)) == 0:  # training
                return [f'{l}{special_token} {e1}' for l, e1 in zip(list_a, list_b)]
            else:
                return [f'{l}{special_token} {e1} {sep_token}{l}{special_token} {e2} {sep_token}{l}{special_token} {e3}'
                        for l, e1, e2, e3 in zip(list_a, list_b, list_c, list_d)]
        else:
            if len(max(list_c)) == 0:  # training
                return list_b
            else:
                return [f'{e1} {sep_token} {e2} {sep_token} {e3}' for e1, e2, e3 in zip(list_b, list_c, list_d)]

    def tokenize_function(examples):
        _text_target = _form_target(cl.int2str(examples['raw_label']), examples['explanation_1'], examples['explanation_2'], examples['explanation_3'], include_label=generation_type=='label_and_explanation')

        if 'predicted_label' in examples:
            examples['raw_label'] = cl.str2int(examples['predicted_label'])

        if generation_type == 'explanation_only':
            # using "[premise] SEP [hypothesis]" generate "[explanation]"
            examples = tokenizer(examples['premise'], examples['hypothesis'], text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'explanation_only_flan_prompt':
            # using "[premise] SEP [hypothesis] SEP Give an explanation." generate "[explanation]"
            examples = tokenizer(examples['premise'], _join_with_sep(examples['hypothesis'], [f'Give an explanation.'] * len(examples['raw_label'])), text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'explanation_use_label':
            # using "[premise] SEP [hypothesis] SEP [label]" generate "[explanation]"
            examples = tokenizer(examples['premise'], _join_with_sep(examples['hypothesis'], cl.int2str(examples['raw_label'])), text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'explanation_use_prompt_label':
            # using "[premise] SEP [hypothesis] SEP It was [label]" generate "[explanation]"
            examples = tokenizer(examples['premise'], _join_with_sep(examples['hypothesis'], [f'It was {i}.' for i in cl.int2str(examples['raw_label'])]), text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'explanation_use_flan_prompt_label':
            # using "[premise] SEP [hypothesis] SEP It is [label]. Give an explanation why." generate "[explanation]"
            examples = tokenizer(examples['premise'], _join_with_sep(examples['hypothesis'], [f'It is {i}. Give an explanation why.' for i in cl.int2str(examples['raw_label'])]), text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        elif generation_type == 'label_and_explanation':
            # using "[premise] SEP [hypothesis]" generate "[label] SEP [explanation]"
            examples = tokenizer(examples['premise'], examples['hypothesis'], text_target=_text_target, truncation=True, padding='do_not_pad', max_length=max_length)
        else:
            raise RuntimeError(f'Unknown generation_type="{generation_type}"')
        return examples

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset


def _get_metrics_function(tokenizer, generation_type='explanation_only'):
    """generation_type: one of 'explanation_only', explanation_only_flan_prompt, 'explanation_use_label', 'explanation_use_prompt_label', 'explanation_use_flan_prompt_label', 'label_and_explanation'"""
    # metrics = evaluate.combine([
    #     evaluate.load('bertscore', lang='en'),
    #     evaluate.load('rouge', use_stemmer=False),
    #     evaluate.load('bleu'),
    # ])
    metric_bs = evaluate.load('bertscore')
    metric_rouge = evaluate.load('rouge')
    metric_bleu = evaluate.load('bleu')

    def _compute_metrics(eval_preds, sep_token=';', special_token=':'):
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if generation_type == 'label_and_explanation':
            raise NotImplementedError('classification metrics not implemented for label_and_explanation')

        # rougeLSum expects newline after each sentence, also split multiple predictions in list of references
        decoded_preds = ['\n'.join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = [['\n'.join(nltk.sent_tokenize(label.strip())) for label in item_labels.split(sep_token)] for item_labels in decoded_labels]

        result = {
            'bertscore_f1': np.mean(metric_bs.compute(predictions=decoded_preds, references=decoded_labels, lang='en', use_fast_tokenizer=True)['f1']),
            # 'bleurt': np.mean(metric_bleurt.compute(predictions=decoded_preds, references=decoded_labels, checkpoint='bleurt-tiny-128')['scores']),
            **metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False),
            **metric_bleu.compute(predictions=decoded_preds, references=decoded_labels),
        }
        return {
            k: result[k]
            for k in ['bertscore_f1', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
        }

    return _compute_metrics


def _get_trainer_args():
    return Seq2SeqTrainingArguments(
        output_dir='tmp',
        report_to='none',
        per_device_eval_batch_size=32,
        no_cuda=not IS_CUDA_AVAILABLE,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=1,  # faster evaluation, but test score will be better
    )


@app.command()
def main(
        model_name: str = typer.Argument(None, help='Pretrained model to finetune: HUB or Path'),
        generation_type: str = typer.Option('explanation_only', help='What we generate'),
        use_label_from: str = typer.Option(None, help='Filename to read labels'),
):
    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')

    # additional labels
    if use_label_from:
        print('Loading labels from', use_label_from)
        predicted_labels = list(pd.read_csv(use_label_from)['predicted_label'])
        print(len(predicted_labels), 'predicted labels loaded')
    else:
        predicted_labels = None

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding='longest',
    )

    # load data
    tokenized_test_dataset = _load_dataset(tokenizer, generation_type=generation_type, max_length=512, predicted_labels=predicted_labels)

    # load new pretrained model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # load metrics
    compute_metrics = _get_metrics_function(tokenizer, generation_type=generation_type)

    # create trainer
    training_args = _get_trainer_args()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print('\n', '-' * 32, 'Predicting...', '-' * 32, '\n')

    test_prediction = trainer.predict(tokenized_test_dataset, num_beams=1)
    predictions = test_prediction.predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # print('decoded_predictions', decoded_predictions)
    print('test_prediction.metrics', test_prediction.metrics)

    df = pd.DataFrame(decoded_predictions, columns=['predicted_explanation'])
    df.to_csv(f'{model_name.replace("/", "-")}.csv', index=False)

    # print('metrics (n_bins=1)', dict(test_prediction.metrics))
    # test_prediction = trainer.predict(tokenized_test_dataset, num_beams=2)
    # print('metrics (n_bins=2)', dict(test_prediction.metrics))
    # test_prediction = trainer.predict(tokenized_test_dataset, num_beams=4)
    # print('metrics (n_bins=4)', dict(test_prediction.metrics))


if __name__ == '__main__':
    app()
