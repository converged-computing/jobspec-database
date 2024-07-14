import data.data_reader as data_reader
from modelclass import ModelClass
import logging
import datetime
import os

dir = os.path.dirname(os.path.abspath(__file__))
import argparse

# create main parser
parser = argparse.ArgumentParser(
    description="Transfer adversarial examples to a different model."
)
parser.add_argument("--dataset", type=str, help="Dataset name", required=True)
parser.add_argument("--basemodel", type=str, help="Base Model name", required=True)
parser.add_argument(
    "--transfermodel", type=str, help="Transfer Model name", required=True
)
parser.add_argument("--bs", type=int, help="Batch Size", default=48)
parser.add_argument(
    "--wandb_logging",
    action=argparse.BooleanOptionalAction,
    help="Whether to log to wandb",
    default=False,
)

# parse arguments
args = parser.parse_args()

DATASET_NAME = args.dataset
base_model_name = args.basemodel
transfer_model_name = args.transfermodel
batch_size = args.bs
WANDB_LOGGING = args.wandb_logging

if WANDB_LOGGING:
    import wandb

    # login to wandb with key
    wandb.login(key="KEY")
    wandb.init(
        project="XAI-Attack",
        entity="ENTITY",
        group="adversarial_example_creation",
        name=DATASET_NAME
        + "_"
        + base_model_name
        + "_"
        + transfer_model_name
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )


logging_path = os.path.join(
    dir,
    "../logs/evaluation/adversarial_transfer/" + DATASET_NAME + "/" + base_model_name,
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log",
)
os.makedirs(os.path.dirname(logging_path), exist_ok=True)
# setting up logger to log to file in logs folder with name created from current time and date
logging.basicConfig(
    filename=logging_path,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

x_train, y_train, x_test, y_test, x_dev, y_dev = data_reader.read_data(
    DATASET_NAME, adversarial=False
)
is_sentence_pair_task = type(x_train) is tuple

# read in adversarial examples
x_train_adv, y_train_adv = data_reader.read_adv_examples(
    DATASET_NAME, base_model_name, is_sentence_pair_task
)

# check if the adversarial examples are adversarial for the transfer model
num_labels = len(set(y_train))

model_transfer = ModelClass(transfer_model_name, False, None, num_labels=num_labels)
model_transfer.train(x_train, y_train, x_dev, y_dev, bs_train=batch_size, bs_eval=128)

if is_sentence_pair_task:
    model_transfer.evaluate((x_train_adv[0], x_train_adv[1]), y_train_adv)
else:
    model_transfer.evaluate(x_train_adv, y_train_adv)
