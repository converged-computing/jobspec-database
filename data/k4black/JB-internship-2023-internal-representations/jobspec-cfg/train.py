import json
import random
from pathlib import Path

import typer
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
import evaluate
import numpy as np
from torchinfo import summary
import nltk
from transformers.integrations import NeptuneCallback
import neptune.new as neptune


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


ROOT_FOLDER = Path(__file__).parent
with open(ROOT_FOLDER / 'params.json') as f:
    EDOS_EVAL_PARAMS = json.load(f)

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
print('IS_CUDA_AVAILABLE', IS_CUDA_AVAILABLE)
print('IS_BF16_AVAILABLE', IS_BF16_AVAILABLE)


nltk.download("punkt", quiet=True)


app = typer.Typer(add_completion=False)


def _load_dataset(tokenizer, max_length=512):
    # dataset = load_dataset("code_x_glue_tc_text_to_code")  # does not contain test labels/code
    dataset = load_dataset('AhmedSSoliman/CodeXGLUE-CONCODE')
    dataset = dataset.rename_column('nl', 'text')

    def tokenize_function(examples):
        examples = tokenizer(examples['text'], text_target=examples['code'], truncation=True, padding='do_not_pad', max_length=max_length)
        return examples

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    return tokenized_dataset


def _get_metrics_function(tokenizer):
    metrics = evaluate.combine(['exact_match', 'rouge', 'bleu'])

    def _compute_metrics(eval_preds):
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # rougeLSum expects newline after each sentence, but we will skip it
        # decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metrics.compute(predictions=decoded_preds, references=decoded_labels)
        return {
            k: result[k]
            for k in ['exact_match', 'rouge1', 'rouge2', 'rougeL', 'bleu']
        }

    return _compute_metrics


def _get_trainer_args(params, hub_model_name, output_dir, push_to_hub=False, model_support_fp16=True, resume_from_checkpoint=False):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        report_to='none',

        learning_rate=params['learning_rate'],
        lr_scheduler_type='linear',
        weight_decay=params.get('weight_decay', 0.01),
        optim=params.get('optim', 'adafactor'),

        auto_find_batch_size=True,  # divide by 2 in case of OOM
        per_device_train_batch_size=params['batch_size'],
        per_device_eval_batch_size=params['batch_size'],
        num_train_epochs=params['max_epochs'],
        warmup_ratio=params.get('warmup_ratio', 0.05),

        no_cuda=not IS_CUDA_AVAILABLE,
        fp16=IS_CUDA_AVAILABLE and model_support_fp16,  # always use fp16 on gpu, if not a special model
        fp16_full_eval=IS_CUDA_AVAILABLE,
        bf16=IS_BF16_AVAILABLE,

        logging_strategy='steps',
        logging_steps=params['eval_steps'],
        evaluation_strategy='steps',
        eval_steps=params['eval_steps'],
        save_strategy='steps',
        save_steps=params['eval_steps'],

        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=3,

        predict_with_generate=True,
        # TODO: set 256, but all other models was validated with max-len=20, so keep it to be able to compare
        generation_max_length=20,  # 256 - max validation and test, a bit lower than max training sample
        generation_num_beams=1,
        torch_compile=False,  # not working on Tesla T4 for now

        hub_model_id=hub_model_name,
        resume_from_checkpoint=resume_from_checkpoint,
        push_to_hub=push_to_hub,
        hub_strategy='checkpoint',
    )


@app.command()
def main(
        base_model: str = typer.Option('t5-small', help='Pretrained model to finetune: HUB or Path'),
        config_name: str = typer.Option('default', help='Config name to use: see params.json'),
        resume_training_id: str = typer.Option(None, help='Neptune tag to resume training from or None'),
        postfix: str = typer.Option('', help='Model name postfix'),
        push_to_hub: bool = typer.Option(False, help='Push model to HuggingFace Hub'),
        save_model: bool = typer.Option(False, help='Save model locally'),
        results_folder: Path = typer.Option(ROOT_FOLDER / 'results', dir_okay=True, writable=True, help='Folder to save results'),
        save_folder: Path = typer.Option(ROOT_FOLDER / 'models', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    clear_base_model = base_model.replace('/', '-')
    model_name_to_save = f'{clear_base_model}-CodeXGLUE-CONCODE-{config_name}'
    if postfix:
        model_name_to_save += f'{model_name_to_save}-{postfix}'
    output_dir = str(results_folder / model_name_to_save)
    model_save_folder = save_folder / model_name_to_save
    hub_model_name = f'k4black/{model_name_to_save}'

    # load config
    params = EDOS_EVAL_PARAMS[config_name.split('-')[0]]  # read base config
    params.update(EDOS_EVAL_PARAMS[config_name])  # update with specific config
    model_support_fp16 = True and 'flan-t5' not in base_model and 'byt5' not in base_model  # flan-t5 and byt5 do not support fp16
    model_max_length = 512 if 'byt5' not in base_model else 1024
    print('model_support_fp16', model_support_fp16)
    print('model_max_length', model_max_length)

    print('\n', '-' * 32, 'Loading...', '-' * 32, '\n')

    # create neptune run
    neptune_run = neptune.init_run(with_id=resume_training_id, tags=[f'model:{base_model}', f'conf:{config_name}'])
    neptune_callback = NeptuneCallback(run=neptune_run)
    neptune_object_id = neptune_run['sys/id'].fetch()
    print('neptune_object_id', neptune_object_id)
    neptune_run['finetuning/parameters'] = {
        'base_model': base_model,
        'config_name': config_name,
    }

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding='longest',
    )

    # load new pretrained model
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    summary(model)

    # add special tokens (add only if not exist in the model)
    if params.get('add_special_tokens', False):
        if 'byt5' not in base_model:
            tokenizer.add_tokens(['\x00', '~', '^', '}', '<', '`', '{', ';', '.', '=', ')', '(', '+', '-'])
        tokenizer.add_special_tokens({'additional_special_tokens': ['concode_elem_sep', 'concode_field_sep']})
        model.resize_token_embeddings(len(tokenizer))

    # load data
    tokenized_dataset = _load_dataset(tokenizer, max_length=model_max_length)

    # load metrics
    compute_metrics = _get_metrics_function(tokenizer)

    # create trainer
    training_args = _get_trainer_args(
        params, hub_model_name, output_dir,
        push_to_hub=push_to_hub, model_support_fp16=model_support_fp16, resume_from_checkpoint=resume_training_id is not None
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=params.get('early_stopping_patience', 5)),
            neptune_callback,
        ],
    )

    print('\n', '-' * 32, 'Training...', '-' * 32, '\n')

    # train itself
    trainer.train(resume_from_checkpoint=resume_training_id is not None)

    # save model
    if save_model:
        if model_save_folder:
            model_save_folder.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_save_folder))

    print('\n', '-' * 32, 'End', '-' * 32, '\n')

    test_prediction = trainer.predict(tokenized_dataset['test'], num_beams=1)
    print('metrics (n_bins=1)', dict(test_prediction.metrics))
    test_prediction = trainer.predict(tokenized_dataset['test'], num_beams=2)
    print('metrics (n_bins=2)', dict(test_prediction.metrics))
    test_prediction = trainer.predict(tokenized_dataset['test'], num_beams=4)
    print('metrics (n_bins=4)', dict(test_prediction.metrics))

    neptune_callback.run['finetuning/final_metrics'] = dict(test_prediction.metrics)


if __name__ == '__main__':
    app()
