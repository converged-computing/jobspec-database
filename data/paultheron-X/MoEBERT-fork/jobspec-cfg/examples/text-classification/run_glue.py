#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import numpy as np
import os
import random
import sys
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.models.bert.modeling_bert_moe import MoEBertForSequenceClassification
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    preprocess_importance: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to pre-compute model importance."}
    )
    moebert: Optional[str] = field(
        default=None,
        metadata={"help": "Whether to use MoE."}
    )
    moebert_distill: Optional[float] = field(
        default=0.0,
        metadata={"help": "Constant for distillation."}
    )
    moebert_expert_dim: Optional[int] = field(
        default=1024,
        metadata={"help": "Dimension of experts."}
    )
    moebert_expert_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout for FFNs in MoE."}
    )
    moebert_expert_num: Optional[int] = field(
        default=4,
        metadata={"help": "Number of experts in MoE."}
    )
    moebert_load_balance: Optional[float] = field(
        default=0.0,
        metadata={"help": "Weight of load balancing loss, only used with trainable gating mechanism."}
    )
    moebert_load_experts: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to load a MoE model."}
    )
    moebert_load_importance: Optional[str] = field(
        default=None,
        metadata={"help": "Load the pre-computed importance."}
    )
    moebert_route_hash_list: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the saved hash list."}
    )
    moebert_route_method: Optional[str] = field(
        default="hash-random",
        metadata={"help": "Choose from gate-token, gate-sentence, hash-random and hash-balance."},
    )
    moebert_share_importance: Optional[int] = field(
        default=512,
        metadata={"help": "Amount of neurons to share across experts."}
    )
    moebert_gate_gamma: Optional[float] = field(
        default=1,
        metadata={"help": "value of gamma to use in the soft trees gating mechanism."}
    )
    moebert_gate_entropy: Optional[float] = field(
        default=1,
        metadata={"help": "value of entropy to use in the entropy regularizer for gating."}
    )
    moebert_perm_epoch: Optional[float] = field(
        default=1,
        metadata={"help": "value of number of epochs to use for learning permutation for gating."}
    )
    moebert_k: Optional[float] = field(
        default=1,
        metadata={"help": "value of k for the number of experts"}
    )
    moebert_trimmed_lasso_reg: Optional[float] = field(
        default=0.0,
        metadata={"help": "value of the trimmed lasso regularization"}
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default='houlsby',
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
    logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        cls_dropout=training_args.cls_dropout,
        apply_lora=model_args.apply_lora,
        lora_alpha=model_args.lora_alpha,
        lora_r=model_args.lora_r,
        apply_adapter=model_args.apply_adapter,
        adapter_type=model_args.adapter_type,
        adapter_size=model_args.adapter_size,
        reg_loss_wgt=model_args.reg_loss_wgt,
        masking_prob=model_args.masking_prob,
    )

    if model_args.preprocess_importance:
        if training_args.do_train:
            raise ValueError("Only support evaluation mode.")
    config.preprocess_importance = model_args.preprocess_importance

    config.moebert = model_args.moebert
    if config.moebert is not None:
        config.moebert = config.moebert.lower()
    config.moebert_distill = model_args.moebert_distill
    config.moebert_expert_dim = model_args.moebert_expert_dim
    config.moebert_expert_dropout = model_args.moebert_expert_dropout
    config.moebert_expert_num = model_args.moebert_expert_num
    config.moebert_load_balance = model_args.moebert_load_balance
    config.moebert_load_experts = model_args.moebert_load_experts
    config.moebert_load_importance = model_args.moebert_load_importance
    config.moebert_route_hash_list = model_args.moebert_route_hash_list
    config.moebert_route_method = model_args.moebert_route_method
    config.moebert_share_importance = model_args.moebert_share_importance
    config.moebert_gate_gamma = model_args.moebert_gate_gamma
    config.moebert_gate_entropy = model_args.moebert_gate_entropy
    config.moebert_perm_epoch = model_args.moebert_perm_epoch
    config.moebert_k = model_args.moebert_k
    config.moebert_trimmed_lasso_reg = model_args.moebert_trimmed_lasso_reg
    
    print("===================================config")
    print(config)
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if config.moebert == "moe":
        if config.model_type == "bert":
            meta_model = MoEBertForSequenceClassification
        else:
            raise ValueError("Model type not recognized by MoE.")
    else:
        meta_model = AutoModelForSequenceClassification
    model = meta_model.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if config.moebert_distill > 0:
        teacher = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        for param in teacher.parameters():
            param.requires_grad = False
        model.teacher = teacher

    trainable_params = []
    if model_args.apply_lora:
        if model_args.lora_path is not None:
            lora_state_dict = torch.load(model_args.lora_path)
            logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
            logger.info(lora_state_dict.keys())
            model.load_state_dict(lora_state_dict, strict=False)
        trainable_params.append('lora')

    if model_args.apply_adapter:
        if model_args.adapter_path is not None:
            adapter_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_adapter.bin'))
            head_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_model_head.bin'))
            added_state_dict = {}
            for k, v in adapter_state_dict.items():
                new_k = k.replace(data_args.task_name + '.', '').replace('adapter_down.0.', 'adapter_A.').replace('adapter_up.', 'adapter_B.').replace('.adapters.', '.adapter.')
                added_state_dict[new_k] = v
            for k, v in head_state_dict.items():
                new_k = k.replace('heads.' + data_args.task_name + '.1', 'classifier.dense').replace('heads.' + data_args.task_name + '.4', 'classifier.out_proj')
                added_state_dict[new_k] = v
            logger.info(f"Apply adapter state dict from {model_args.adapter_path}.")
            logger.info(added_state_dict.keys())
            missing_keys, unexpected_keys = model.load_state_dict(added_state_dict, strict=False)
            for missing_key in missing_keys:
                assert 'adapter' not in missing_key, missing_key + ' is missed in the model'
            assert len(unexpected_keys) == 0, 'Unexpected keys ' + str(unexpected_keys)
        trainable_params.append('adapter')

    if model_args.apply_bitfit:
        trainable_params.append('bias')

    if len(trainable_params) > 0:
        for name, param in model.named_parameters():
            if name.startswith('deberta') or name.startswith('roberta'):
                param.requires_grad = False
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        param.requires_grad = True
                        break
            else:
                param.requires_grad = True

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        if data_args.task_name == "mnli":
            eval_dataset_mnli_mm = datasets["validation_mismatched"]

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset_mnli_mm=eval_dataset_mnli_mm if data_args.task_name == "mnli" and training_args.do_eval else None,
    )

    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        # Check the config from that potential checkpoint has the right number of labels before using it as a
        # checkpoint.
        if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
            checkpoint = model_args.model_name_or_path

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if config.moebert_load_experts:
            from transformers.file_utils import WEIGHTS_NAME
            import process.utils

            process.utils.process_ffn(model)
            if checkpoint is not None and os.path.isfile(os.path.join(checkpoint, WEIGHTS_NAME)):
                logger.info(f"Loading model from {checkpoint}).")
                state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME), map_location="cpu")
                trainer._load_state_dict_in_model(state_dict)
                del state_dict

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

#     if training_args.do_predict:
#         logger.info("*** Test ***")

#         # Loop to handle MNLI double evaluation (matched, mis-matched)
#         tasks = [data_args.task_name]
#         test_datasets = [test_dataset]
#         if data_args.task_name == "mnli":
#             tasks.append("mnli-mm")
#             test_datasets.append(datasets["test_mismatched"])

#         for test_dataset, task in zip(test_datasets, tasks):
#             # Removing the `label` columns because it contains -1 and Trainer won't like that.
#             test_dataset.remove_columns_("label")
#             predictions = trainer.predict(test_dataset=test_dataset).predictions
#             predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

#             output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
#             if trainer.is_world_process_zero():
#                 with open(output_test_file, "w") as writer:
#                     logger.info(f"***** Test results {task} *****")
#                     writer.write("index\tprediction\n")
#                     for index, item in enumerate(predictions):
#                         if is_regression:
#                             writer.write(f"{index}\t{item:3.3f}\n")
#                         else:
#                             item = label_list[item]
#                             writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()