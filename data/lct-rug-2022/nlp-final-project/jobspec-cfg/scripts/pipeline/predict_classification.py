import random
from pathlib import Path

import pandas as pd
import typer
from datasets import load_dataset, ClassLabel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
import evaluate
import numpy as np
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


def _load_dataset(tokenizer, classification_type='nli', predicted_explanation=None):
    """
    :param tokenizer:
    :param classification_type: one of 'nli', 'nli_explanation'
    :return:
    """
    dataset = load_dataset('esnli')
    dataset = dataset['test']
    if predicted_explanation is not None:
        dataset = dataset.add_column('predicted_explanation', predicted_explanation)

    def _join_with_sep(list_a, list_b):
        return [f'{h}{tokenizer.eos_token} {l}' for h, l in zip(list_a, list_b)]

    def tokenize_function(examples):
        if classification_type == 'nli':
            # using "[premise] SEP [hypothesis]" classify on [label]
            examples = tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='do_not_pad')
        elif classification_type == 'nli_explanation':
            # using "[premise] SEP [hypothesis] SEP [explanation]" classify on [label]
            examples = tokenizer(examples['premise'], _join_with_sep(examples['hypothesis'], examples['explanation_1' if predicted_explanation is None else 'predicted_explanation']), truncation=True, padding='do_not_pad')
        else:
            raise RuntimeError(f'Unknown classification_type="{classification_type}"')
        return examples

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset


def _get_metrics_function(tokenizer):
    # metrics = evaluate.combine([
    #     evaluate.load('f1', average='macro'),
    #     evaluate.load('accuracy'),
    # ])
    metric_f1 = evaluate.load('f1')
    metric_acc = evaluate.load('accuracy')

    def _compute_metrics(eval_preds):
        preds, labels = eval_preds
        predictions = np.argmax(preds, axis=-1)
        return {
            **metric_f1.compute(predictions=predictions, references=labels, average='macro'),
            **metric_acc.compute(predictions=predictions, references=labels),
        }

    return _compute_metrics


def _get_trainer_args():
    return TrainingArguments(
        output_dir='tmp',
        report_to='none',
        auto_find_batch_size=False,  # divide by 2 in case of OOM
        per_device_eval_batch_size=32,
        no_cuda=not IS_CUDA_AVAILABLE,
    )


@app.command()
def main(
        model_name: str = typer.Argument(None, help='Pretrained model to finetune: HUB or Path'),
        classification_type: str = typer.Option('nli', help='What we classify'),
        use_explanation_from: str = typer.Option(None, help='Filename to read explanation'),
):
    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')

    # additional explanation
    if use_explanation_from:
        print('Loading explanation from', use_explanation_from)
        predicted_explanation = list(pd.read_csv(use_explanation_from)['predicted_explanation'])
        print(len(predicted_explanation), 'predicted explanation loaded')
    else:
        predicted_explanation = None

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
    )

    # load data
    tokenized_test_dataset = _load_dataset(tokenizer, classification_type=classification_type)

    # load new pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # load metrics
    compute_metrics = _get_metrics_function(tokenizer)

    # create trainer
    training_args = _get_trainer_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print('\n', '-' * 32, 'Predicting', '-' * 32, '\n')

    cl: ClassLabel = tokenized_test_dataset.features['label']
    test_prediction = trainer.predict(tokenized_test_dataset)
    predictions = np.argmax(test_prediction.predictions, axis=-1)
    label_predictions = cl.int2str(predictions)

    # print('label_predictions', label_predictions)
    print('test_prediction.metrics', test_prediction.metrics)

    df = pd.DataFrame(label_predictions, columns=['predicted_label'])
    df.to_csv(f'{model_name.replace("/", "-")}.csv', index=False)


if __name__ == '__main__':
    app()
