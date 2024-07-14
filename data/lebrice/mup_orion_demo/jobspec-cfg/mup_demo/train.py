"""Trains a GPT2 model with MuP parametrization on a causal language modeling task.

Adapted from a HuggingFace language modeling example:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

Example command (make sure to run `accelerate config` first):
```
accelerate launch mup_demo/train.py \
    --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
    --output_dir test_run
```
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Callable, Literal

import datasets
import evaluate
import mutransformers
import simple_parsing
import transformers
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from simple_parsing.helpers import flag
from simple_parsing.helpers.serialization.serializable import save_yaml
from torch import Tensor
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers import Trainer as _Trainer
from transformers import TrainingArguments as _TrainingArguments
from transformers import default_data_collator, is_torch_tpu_available, set_seed
from transformers.integrations import WandbCallback
from transformers.testing_utils import CaptureLogger
from transformers.trainer import TrainOutput
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ExplicitEnum
from transformers.utils.logging import get_logger

from mup_demo.model import get_gpt2_model
from mup_demo.mup_trainer_patch import patch_trainer_for_mup
from mup_demo.utils import is_main_process

try:
    import wandb
except ImportError:
    wandb = None
logger = get_logger(__name__)

# Apply the 'patch' to the Trainer class so it uses the mup optimizers.
patch_trainer_for_mup()


@dataclass
class GPT2ConfigArgs:
    """Dataclass containing the fields of the GPT2Config class, so we can change them from the CLI
    more easily than with the 'config_overrides' argument from the example.

    NOTE: Unsure why HuggingFace didn't opt to use a dataclass for the GPT2Config class. It would
    have made this unnecessary.
    """

    vocab_size: int = 50257
    """Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
    """

    n_positions: int = 1024
    """The maximum sequence length that this model might ever be used with. Typically set this to something large
    just in case (e.g., 512 or 1024 or 2048).
    """

    n_embd: int = 768
    """ Dimensionality of the embeddings and hidden states. """

    n_layer: int = 12
    """ Number of hidden layers in the Transformer encoder. """

    n_head: int = 12
    """ Number of attention heads for each attention layer in the Transformer encoder."""

    n_inner: int | None = None
    """Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd """

    activation_function: Literal["relu", "silu", "gelu", "tanh", "gelu_new"] = "gelu"
    """ Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    """

    resid_pdrop: float = 0.1
    """The dropout probability for all fully connected layers in the embeddings, encoder, and
    pooler.
    """

    embd_pdrop: float = 0.1
    """The dropout ratio for the embeddings. """

    attn_pdrop: float = 0.1
    """ The dropout ratio for the attention. """

    layer_norm_epsilon: float = 1e-5
    """ The epsilon to use in the layer normalization layers. """

    initializer_range: float = 0.02
    """The standard deviation of the truncated_normal_initializer for initializing all weight
    matrices.
    """

    # NOTE: Omitting the summary_* fields, since they're not used in the model. They're only used
    # when doing a sentence summarization task.

    scale_attn_weights: bool = True
    """Scale attention weights by dividing by sqrt(hidden_size)."""

    use_cache: bool = True
    """ Whether or not the model should return the last key/values attentions (not used by all
    models).
    """

    scale_attn_by_inverse_layer_idx: bool = False
    """Whether to additionally scale attention weights by `1 / layer_idx + 1`."""

    reorder_and_upcast_attn: bool = False
    """Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
    dot-product/softmax to float() when training with mixed precision.
    """

    attn_mult: float | None = None
    """ "Attention key-projection weight multiplier alpha_{attn}" from the mup paper.
    If set to None, take the value of `(self.hidden_size / self.num_attention_heads)**0.5`.
    """

    readout_zero_init: bool = False
    query_zero_init: bool = False

    def to_config(self) -> mutransformers.GPT2Config:
        """Convert this dataclass to a GPT2Config instance."""
        config_kwargs = asdict(self)
        config_kwargs.pop("readout_zero_init")  # Not a valid argument for GPT2Config
        config_kwargs.pop("query_zero_init")  # Not a valid argument for GPT2Config
        return mutransformers.GPT2Config(**asdict(self))

    def make_model(self) -> mutransformers.GPT2LMHeadModel:
        # NOTE: unused. Currently creating the config and then using it
        # NOTE: This currently doesn't allow for finetuning a model.
        config = self.to_config()
        logger.info(f"Creating a MuP-parametrized GPT2 model with config: {config}")
        return get_gpt2_model(
            config,
            model_type=mutransformers.GPT2LMHeadModel,
            readout_zero_init=self.readout_zero_init,
            query_zero_init=self.query_zero_init,
        )


@dataclass
class ModelArguments:
    """Configuration options about the model/config/tokenizer that will be trained.

    NOTE: In the original example from HF, the script can be used to finetune models from the hub.
    Here, we only care about training a gpt2 model from scratch with the MuP parameterization.
    """

    model: GPT2ConfigArgs = field(default_factory=GPT2ConfigArgs)
    """Configuration options for the model. This corresponds to the parameers of the GPT2Config
    class which is used to create the model.
    """

    tokenizer_name: str = "gpt2"
    """ Pretrained tokenizer name or path."""

    use_fast_tokenizer: bool = flag(True)
    """Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."""

    model_revision: str = "main"
    """ The specific model version to use (can be a branch name, tag name or commit id)."""

    use_auth_token: bool = False
    """Will use the token generated when running `huggingface-cli login` (necessary to use this
    script with private models).
    """


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    dataset_name: str | None = None
    """The name of the dataset to use (via the datasets library)."""

    dataset_config_name: str | None = None
    """The configuration name of the dataset to use (via the datasets library)."""

    train_file: str | None = None
    """The input training data file (a text file)."""

    validation_file: str | None = None
    """An optional input evaluation data file to evaluate the perplexity on (a text file)."""

    max_train_samples: int | None = None
    """For debugging purposes or quicker training, truncate the number of training examples to
    this value if set."""

    max_eval_samples: int | None = None
    """For debugging purposes or quicker training, truncate the number of evaluation examples to
    this value if set.
    """

    block_size: int | None = None
    """Optional input sequence length after tokenization.
    The training dataset will be truncated in block of this size for training.
    Default to the model max input length for single sentence inputs (take into account special
    tokens).
    """

    overwrite_cache: bool = False
    """Overwrite the cached training and evaluation sets"""

    validation_split_percentage: int | None = 5
    """The percentage of the train set used as validation set in case there's no validation split.
    """

    preprocessing_num_workers: int | None = None
    """The number of processes to use for the preprocessing."""

    keep_linebreaks: bool = True
    """Whether to keep line breaks when using TXT files or not."""

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.train_file:
            _raise_if_bad_extension(self.train_file, "train_file")
        if self.validation_file:
            _raise_if_bad_extension(self.validation_file, "validation_file")


@dataclass
class TrainingArguments(_TrainingArguments):
    """The subset of arguments which relate to the training loop."""

    def __post_init__(self):
        # NOTE: Little temporary patch so these fields (which have a `str | Enum` annotation) are
        # parsed properly.
        # BUG: A bug that occurs with ExplicitEnum, where it parses it by name into
        # a 'IntervalStrategy.NO' string, rather than into the 'IntervalStrategy.NO' enum value!
        def _to_str(v: ExplicitEnum | str) -> str:
            if "." in v:
                return v.split(".")[-1].lower()
            return v if not isinstance(v, ExplicitEnum) else v.value

        self.evaluation_strategy = _to_str(self.evaluation_strategy)  # "no"
        self.logging_strategy = _to_str(self.logging_strategy)  # "steps"
        self.save_strategy = _to_str(self.save_strategy)  # "steps"
        self.hub_strategy = _to_str(self.hub_strategy)  # "every_save"
        self.lr_scheduler_type = _to_str(self.lr_scheduler_type)  # "linear"
        self.optim = _to_str(self.optim)  # "adamw_hf"
        if self.sharded_ddp == []:
            self.sharded_ddp = ""
        super().__post_init__()


# Apply a small patch to the Trainer class, to allow the auto_find_batch_size to work properly.
class Trainer(_Trainer):
    def _inner_training_loop(
        self,
        batch_size: int,
        args: TrainingArguments,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        # NOTE: THis fixes a small bug here in the base class:
        args.per_device_train_batch_size = batch_size
        if is_main_process():
            # Also update the wandb config to reflect the batch size that is actually used.
            if wandb and wandb.run:
                wandb.config.update(
                    {
                        "training_args": dataclasses.asdict(args),
                    },
                    allow_val_change=True,
                )
                # TODO: There's a duplicate, non-nested entry for "per-device_train_batch_size".
                # It probably gets set by the `WandbCallback`. Only update it if it's already present.
                if "per_device_train_batch_size" in wandb.config:
                    wandb.config.update(
                        {
                            "per_device_train_batch_size": batch_size,
                        },
                        allow_val_change=True,
                    )

        return super()._inner_training_loop(
            batch_size=batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )


def _raise_if_bad_extension(file_path: str, attr_name: str):
    extension = file_path.split(".")[-1]
    if extension not in ["csv", "json", "txt"]:
        raise ValueError(f"`{attr_name}` should be a csv, json or txt file.")


def parse_args(
    default_model_args: ModelArguments | None = None,
    default_data_args: DataTrainingArguments | None = None,
    default_training_args: TrainingArguments | None = None,
) -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    """Parse the model, data, and training arguments from the command-line.

    Uses [SimpleParsing](https://www.github.com/lebrice/SimpleParsing), an alternative to the
    HFArgumentParser, which also uses dataclasses to create argparse arguments.
    """
    parser = simple_parsing.ArgumentParser(description=__doc__, add_config_path_arg=True)
    parser.add_arguments(ModelArguments, dest="model", default=default_model_args)
    parser.add_arguments(DataTrainingArguments, dest="data", default=default_data_args)
    parser.add_arguments(TrainingArguments, dest="training", default=default_training_args)

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    args = parser.parse_args()

    model_args = args.model
    data_args = args.data
    training_args = args.training

    return model_args, data_args, training_args


def parse_with_good_defaults() -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    return parse_args(
        default_model_args=ModelArguments(
            model=GPT2ConfigArgs(
                n_embd=256,
                n_head=4,
                n_layer=2,
                readout_zero_init=True,
                query_zero_init=True,
            ),
            tokenizer_name="gpt2",
        ),
        default_data_args=DataTrainingArguments(
            dataset_name="wikitext",
            dataset_config_name="wikitext-2-raw-v1",
        ),
        default_training_args=TrainingArguments(
            output_dir="runs/debug",
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_steps=500,
            greater_is_better=False,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            auto_find_batch_size=False,
            ddp_find_unused_parameters=False,
            do_train=True,
            do_eval=True,
            num_train_epochs=1,
            report_to="wandb",
        ),
    )


def main():
    # Get the configurations for the training run from the command-line.
    model_args, data_args, training_args = parse_with_good_defaults()

    # Setup logging for HuggingFace.
    _setup_logging(training_args)

    trainer = setup_trainer(
        model_args=model_args, data_args=data_args, training_args=training_args
    )

    if training_args.do_train:
        train(trainer=trainer, model_args=model_args, data_args=data_args)

    metrics = None
    if training_args.do_eval:
        metrics = evaluation_loop(trainer=trainer, model_args=model_args, data_args=data_args)

    # NOTE: Removed some Optional Post-run stuff: creating a model card, uploading the model to
    # the hub, etc.
    if not is_main_process():
        return metrics

    if training_args.do_eval:
        assert metrics is not None
        from orion.client import report_results

        objective = metrics["eval_loss"]
        report_results(
            [dict(name="objective", type="objective", value=objective)]
            + [dict(name=key, type="statistic", value=value) for key, value in metrics.items()]
        )

    return metrics


def train(
    trainer: Trainer, model_args: ModelArguments, data_args: DataTrainingArguments
) -> dict[str, float]:
    train_dataset = trainer.train_dataset
    assert train_dataset is not None
    training_args = trainer.args

    checkpoint = None
    last_checkpoint = find_last_checkpoint(training_args)
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint or None

    log_dir = Path(training_args.output_dir)
    log_dir.mkdir(exist_ok=True, parents=True)  # Make sure the log dir exists
    save_yaml(training_args, log_dir / "training_args.yaml")
    save_yaml(model_args, log_dir / "model_args.yaml")
    save_yaml(data_args, log_dir / "data_args.yaml")

    train_result: TrainOutput = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logging_to_wandb = (
        wandb is not None
        and is_main_process()
        and bool(training_args.report_to)
        and (
            "all" in training_args.report_to
            or any(log_backend.endswith("wandb") for log_backend in training_args.report_to)
        )
    )
    if logging_to_wandb:
        # TODO: Having trouble getting wandb to save the checkpoint folders in the output dir!
        assert wandb
        wandb.save(training_args.output_dir + "/**", base_path=training_args.output_dir)
        wandb.save(
            training_args.output_dir + "/checkpoint-*/**", base_path=training_args.output_dir
        )
        for subdir in Path(training_args.output_dir).iterdir():
            if subdir.is_dir():
                wandb.save(str(subdir), base_path=training_args.output_dir)

    return metrics


# Evaluation
def evaluation_loop(
    trainer: Trainer, model_args: ModelArguments, data_args: DataTrainingArguments
):
    logger.info("*** Evaluate ***")
    eval_dataset = trainer.eval_dataset
    assert eval_dataset is not None

    metrics = trainer.evaluate()

    max_eval_samples = (
        data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    )
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


def setup_trainer(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
) -> Trainer:
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm", model_args, data_args)

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    # last_checkpoint = find_last_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = get_datasets(data_args=data_args, model_args=model_args)

    # Load pretrained model and tokenizer

    tokenizer = _get_tokenizer(model_args=model_args)

    # Preprocessing the datasets.

    train_dataset, eval_dataset = preprocess_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # logger.info("Non-default config values:\n" + yaml.dump(config.to_diff_dict()))
    model_init: Callable[[], PreTrainedModel] = functools.partial(
        make_model,
        tokenizer=tokenizer,
        model_args=model_args,
    )

    preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None
    compute_metrics: Callable[[EvalPrediction], dict] | None = None
    if training_args.do_eval and not is_torch_tpu_available():

        def _preprocess_logits_for_metrics(logits: Tensor, labels: Tensor) -> Tensor:
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", experiment_id=training_args.run_name)

        def _compute_metrics(eval_preds: EvalPrediction) -> dict:
            preds, labels = eval_preds
            # TODO: Switch to this syntax if it works.
            # preds = eval_preds.predictions
            # labels = eval_preds.label_ids

            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            metrics = {}
            # BUG: Was getting a weird FileNotFoundError: [Errno 2] No such file or directory
            metrics = metric.compute(predictions=preds, references=labels)

            assert isinstance(metrics, dict)
            return metrics

        preprocess_logits_for_metrics = _preprocess_logits_for_metrics
        compute_metrics = _compute_metrics

    assert train_dataset is not None

    logging_to_wandb = (
        wandb is not None
        and is_main_process()
        and bool(training_args.report_to)
        and (
            "all" in training_args.report_to
            or any(log_backend.endswith("wandb") for log_backend in training_args.report_to)
        )
    )

    if logging_to_wandb:
        assert wandb

        run_id = None
        if "ORION_EXPERIMENT_NAME" in os.environ:
            # Create the run id from the Orion info:
            run_id = (
                os.environ["ORION_EXPERIMENT_NAME"]
                + "_"
                + os.environ["ORION_EXPERIMENT_VERSION"]
                + "_"
                + os.environ["ORION_TRIAL_ID"]
            )
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "mup_demo"),
            name=training_args.run_name,
            # TODO: Perhaps we could use the Trial ID of Orion to identify the wandb run?
            id=run_id,
            config={
                "model": dataclasses.asdict(model_args),
                "data": dataclasses.asdict(data_args),
                "training_args": dataclasses.asdict(training_args),
            },
            # TODO: Unsure about this one here.
            # dir=training_args.output_dir,
            allow_val_change=True,
            resume="allow",
        )
        if wandb.run and wandb.run.resumed:
            logger.critical("RUN IS BEING RESUMED!")
        wandb.save(training_args.output_dir + "/**")

    # Prevent the Trainer from typing to create a wandb callback (we're adding it ourselves below).
    # NOTE: can't use other reporting methods because of this line here.
    training_args.report_to = []

    # Initialize our Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        # NOTE: next arg has wrong annotation (should be marked as optional)
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # type: ignore
    )

    if logging_to_wandb:
        assert wandb
        assert wandb.run is not None
        trainer.add_callback(WandbCallback())
    return trainer


def find_last_checkpoint(training_args: _TrainingArguments) -> str | None:
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # NOTE: This is being a little bit annoying.
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.warning(
                RuntimeWarning(
                    f"Output directory ({training_args.output_dir}) already exists and contains "
                    "no checkpoints. Overwriting."
                )
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this "
                f"behavior, change the `--output_dir` or add `--overwrite_output_dir` to train "
                f"from scratch."
            )
    return last_checkpoint


def get_datasets(data_args: DataTrainingArguments, model_args: ModelArguments) -> DatasetDict:
    """Get the datasets.

    You can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    or just provide the name of one of the public datasets available on the hub at
    https://huggingface.co/datasets/ (the dataset will be downloaded automatically from the
    datasets Hub).

    For CSV/JSON files, this script will use the column called 'text' or the first column if no
    column called 'text' is found. You can easily tweak this behavior (see below).

    In distributed training, the load_dataset function guarantee that only one local process can concurrently
    download the dataset.
    """
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            use_auth_token=model_args.use_auth_token,
        )
        assert isinstance(raw_datasets, DatasetDict)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                use_auth_token=model_args.use_auth_token,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                use_auth_token=model_args.use_auth_token,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            use_auth_token=model_args.use_auth_token,
            **dataset_args,
        )
        assert isinstance(raw_datasets, DatasetDict)

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=model_args.use_auth_token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=model_args.use_auth_token,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    return raw_datasets


def _setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )


def _get_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizerBase:
    if not model_args.tokenizer_name:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    return tokenizer


def make_model(
    model_args: ModelArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> mutransformers.GPT2LMHeadModel:
    """Creates the model using the model configuration and the tokenizer."""
    logger.info("Creating a MuP GPT2 model.")
    model = model_args.model.make_model()

    # NOTE: Compared to `model.num_parameters()` this doesn't count shared parameters twice
    # (idk if this is useful for GPT2 though).
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Model size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    return model


def preprocess_datasets(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
) -> tuple[Dataset | None, Dataset | None]:
    shared_fs_between_nodes = False
    keep_in_memory = True

    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be "
                "chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(
        desc="dataset map tokenization", local=not shared_fs_between_nodes
    ):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            keep_in_memory=False,
            desc="Running tokenizer on dataset",
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(
        desc="grouping texts together", local=not shared_fs_between_nodes
    ):
        # NOTE: This is how you can keep the dataset in memory, while also saving/loading from a cache file!
        save_path = Path(os.environ["SLURM_TMPDIR"]) / "temp"
        if training_args.local_process_index == 0:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
                keep_in_memory=False,
            )
            lm_datasets.save_to_disk(str(save_path))
        lm_datasets = DatasetDict.load_from_disk(str(save_path), keep_in_memory=keep_in_memory)

    train_dataset: Dataset | datasets.arrow_dataset.Dataset | None = None
    eval_dataset: Dataset | datasets.arrow_dataset.Dataset | None = None

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train_dataset, eval_dataset


if __name__ == "__main__":
    main()
